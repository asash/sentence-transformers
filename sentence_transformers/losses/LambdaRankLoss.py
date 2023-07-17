from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from torch import (tile,  
                   unsqueeze, 
                   mean, 
                   topk, 
                   sign, 
                   gather,
                   Tensor)

class LambdarankLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_true_raw, y_pred_raw, instance: LambdaRankLoss):
        ctx.instance = instance
        if instance.remove_batch_dim:
            y_true = y_true_raw.view(instance.batch_size, instance.num_items)
            y_pred = y_pred_raw.view(instance.batch_size, instance.num_items)
        else:
            y_true = y_true_raw
            y_pred = y_pred_raw
        result = torch.mean(torch.abs(y_pred))
        ctx.save_for_backward(y_true, y_pred)
        return result

    @staticmethod
    def backward(ctx, dy):
        y_true, y_pred = ctx.saved_tensors
        instance: LambdaRankLoss  = ctx.instance
        lambdarank_lambdas = instance.get_lambdas(y_true, y_pred)
        bce_lambdas = instance.get_bce_lambdas(y_true, y_pred)
        grad_output = torch.zeros_like(y_true), ((1 - instance.bce_grad_weight) * lambdarank_lambdas + (bce_lambdas * instance.bce_grad_weight)) * dy, None
        return grad_output


class LambdaRankLoss(torch.autograd.Function):
    def __init__(self, 
                 num_items : int = None, 
                 batch_size : int = None, 
                 sigma : float = 1.0, 
                 ndcg_at : int = 50, 
                 dtype : torch.dtype = torch.float32, 
                 lambda_normalization : bool = True,
                 pred_truncate_at : int = None, 
                 bce_grad_weight : float = 0.0, 
                 device='cpu',
                 remove_batch_dim : bool = False):
        
        self.__name__ = 'lambdarank'
        self.num_items = num_items
        self.batch_size = batch_size
        self.sigma = sigma
        self.dtype = dtype
        self.bce_grad_weight = bce_grad_weight
        self.remove_batch_dim = remove_batch_dim
        self.params_truncate_at = pred_truncate_at
        self.device = device
        self.params_ndcg_at = ndcg_at
        self.lambda_normalization = lambda_normalization
        self.less_is_better = False
        self.setup()
    
    def get_pairwise_diffs_for_vector(self, x):
        b, a = torch.meshgrid(x, x[:self.ndcg_at])
        return b - a
    
    def get_pairwise_diff_batch(self, x):
        x_top_tile = tile(unsqueeze(x[:, :self.ndcg_at], 1), [1, self.pred_truncate_at, 1])
        x_tile = tile(unsqueeze(x, 2), [1, 1, self.ndcg_at])
        result = x_tile - x_top_tile
        return result

    def setup(self):
        if self.batch_size is None or self.num_items is None:
            return
        
        if self.params_truncate_at == None:
            self.pred_truncate_at = self.num_items
        else:
            self.pred_truncate_at = self.params_truncate_at

        self.ndcg_at = min(self.params_ndcg_at, self.num_items)
        self.dcg_position_discounts = (1. / torch.log2((torch.range(0,self.pred_truncate_at-1) + 2).type(self.dtype))).to(self.device)
        self.top_position_discounts = self.dcg_position_discounts[:self.ndcg_at].view(self.ndcg_at, 1)
        self.swap_importance = torch.abs(self.get_pairwise_diffs_for_vector(self.dcg_position_discounts))
        self.batch_indices = tile(unsqueeze(torch.range(0, self.batch_size-1), 1), [1, self.pred_truncate_at]).view(self.pred_truncate_at * self.batch_size, 1).to(self.device)
        self.mask = (1 - F.pad(torch.ones(self.ndcg_at), (0, self.pred_truncate_at - self.ndcg_at)).view(1, self.pred_truncate_at)).type(self.dtype).to(self.device)
        
    
    #in pytorch y_pred is the first argument
    def __call__(self, y_pred, y_true):
        return LambdarankLossFn.apply(y_true, y_pred, self)

    def get_bce_lambdas(self, y_true, y_pred):
        s_x = torch.sigmoid(y_pred)  # calculate sigmoid of y_pred
        logits_loss_lambdas = (s_x - y_true) / self.num_items
        return logits_loss_lambdas

    def bce_lambdas_len(self, y_true, y_pred):
        bce_lambdas = self.get_bce_lambdas(y_true, y_pred)
        norms = torch.norm(bce_lambdas , axis=1)
        return self.bce_grad_weight * mean(norms)

    def get_lambdas(self, y_true, y_pred):
        sorted_by_score = topk(y_pred.type(self.dtype), self.pred_truncate_at)
        col_indices_reshaped = sorted_by_score.indices.view(self.pred_truncate_at * self.batch_size, 1)
        pred_ordered = sorted_by_score.values
        true_ordered = y_true.gather(1, sorted_by_score.indices).type(self.dtype)
        inverse_idcg = self.get_inverse_idcg(true_ordered)
        true_gains = 2 ** true_ordered - 1
        true_gains_diff = self.get_pairwise_diff_batch(true_gains)
        S = sign(true_gains_diff)
        delta_ndcg = true_gains_diff * self.swap_importance * inverse_idcg
        pairwise_diffs = self.get_pairwise_diff_batch(pred_ordered) * S

        #normalize dcg gaps - inspired by lightbm
        if self.lambda_normalization:
            best_score = pred_ordered[:, 0]
            worst_score = pred_ordered[:, -1]

            range_is_zero = torch.eq(best_score, worst_score).type(self.dtype).view(self.batch_size, 1, 1)
            norms = (1 - range_is_zero) * (torch.abs(pairwise_diffs) + 0.01) + (range_is_zero)
            delta_ndcg = torch.where(norms != 0, delta_ndcg / norms, torch.tensor(float('nan')))


        sigmoid = -self.sigma / (1 + torch.exp(self.sigma * (pairwise_diffs)))
        lambda_matrix =  delta_ndcg * sigmoid

        #calculate sum of lambdas by rows. For top items - calculate as sum by columns.
        lambda_sum_raw = torch.sum(lambda_matrix, axis=2)
        top_lambda_sum = torch.sum(lambda_matrix, dim=1)
        pad = (0, self.pred_truncate_at - self.ndcg_at)  # Padding for the last dimension
        top_lambda_sum = torch.nn.functional.pad(-top_lambda_sum, pad, mode='constant', value=0)
        lambda_sum_raw_top_masked = lambda_sum_raw * self.mask
        lambda_sum_result = lambda_sum_raw_top_masked + top_lambda_sum

        if self.lambda_normalization:
            #normalize results - inspired by lightbm
            all_lambdas_sum = torch.reshape(torch.sum(torch.abs(lambda_sum_result), axis=(1)), (self.batch_size, 1))
            norm_factor = torch.where(all_lambdas_sum != 0, torch.log2(all_lambdas_sum + 1) / all_lambdas_sum, torch.zeros_like(all_lambdas_sum))
            lambda_sum = lambda_sum_result * norm_factor
        else:
            lambda_sum = lambda_sum_result

        indices = torch.concat([self.batch_indices, col_indices_reshaped], axis=1)
        reshaped_lambda_sum = lambda_sum.reshape(self.pred_truncate_at * self.batch_size)
        result_lambdas = torch.zeros(self.batch_size, self.num_items, dtype=reshaped_lambda_sum.dtype, device=reshaped_lambda_sum.device)
        indices = indices.long()
        result_lambdas[indices[:, 0], indices[:, 1]] = reshaped_lambda_sum
        return result_lambdas.type(torch.float32)

    def get_inverse_idcg(self, true_ordered):
        top_k_values = topk(true_ordered, self.ndcg_at).values
        top_k_discounted = torch.linalg.matmul(top_k_values, self.top_position_discounts)
        no_nan_division = torch.where(top_k_discounted != 0, 1.0 / top_k_discounted, torch.zeros_like(top_k_discounted))
        reshaped = no_nan_division.reshape(self.batch_size, 1, 1)
        return reshaped
    
class LambdasSumWrapper:
    def __init__(self, lambdarank_loss : LambdaRankLoss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "lambdarank_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_lambdas(y_true, y_pred)
        return (1 - self.lambdarank_loss.bce_grad_weight) * torch.sum(torch.abs(lambdas))

class BCELambdasSumWrapper:
    def __init__(self, lambdarank_loss : LambdaRankLoss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "bce_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_bce_lambdas(y_true, y_pred)
        norms = torch.sum(lambdas, axis=1)
        return (self.lambdarank_loss.bce_grad_weight) * torch.sum(torch.abs(lambdas))
