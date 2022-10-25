import torch
import torch.nn as nn

# weight norm but with epsilon added to the norm to avoid division by zero
# https://github.com/pytorch/pytorch/blob/852f8526c52190125446adc9a6ecbcc28fb66182/aten/src/ATen/native/WeightNorm.cpp
class WeightNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_g, weight_v, eps, dim):
        dim = weight_v.ndim + dim if dim < 0 else dim
        norm = torch.norm_except_dim(weight_v, 2, dim)
        norm = torch.clamp_min(norm, eps)
        ctx.save_for_backward(weight_g, weight_v, norm)
        ctx.dim = dim
        return weight_g / norm * weight_v
    
    @staticmethod
    def backward(ctx, grad_output):
        weight_g, weight_v, norm = ctx.saved_tensors
        dim = ctx.dim
        per_dim_sums = weight_v * grad_output
        if dim == 0:
            per_dim_sums = per_dim_sums.view(weight_v.size(0), -1).sum(1).view_as(weight_g)
        else: # dim == last_dim
            per_dim_sums = per_dim_sums.view(-1, weight_v.size(-1)).sum(0).view_as(weight_g)
        dydg = per_dim_sums / norm
        dydv = (weight_g / norm) * (grad_output - weight_v * (per_dim_sums / norm ** 2))
        return dydg, dydv, None, None

class WeightNorm(nn.Module):
    def __init__(self, eps=1e-8, dim=0):
        super().__init__()
        self.eps = eps
        self.dim = dim
    
    def forward(self, weight_g, weight_v):
        # return weight_g / norm_except_dim(weight_v, self.eps, self.dim) * weight_v
        return WeightNormFunction.apply(weight_g, weight_v, self.eps, self.dim)
    
    def right_inverse(self, weight):
        dim = weight.ndim + self.dim if self.dim < 0 else self.dim
        norm = torch.norm_except_dim(weight, 2, dim)
        norm = torch.clamp_min(norm, self.eps)
        return norm, weight
