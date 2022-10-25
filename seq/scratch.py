import torch

import triton
import triton.language as tl

@triton.jit
def apply_dropout(x, offsets, p, seed, mask_val=0.):
    scale = 1 / (1 - p)
    rand = tl.rand(seed, offsets)
    return tl.where(rand > p, x * scale, mask_val)

@triton.jit
def _dropout(X, O,
             stride_x1, stride_x2,
             stride_o1, stride_o2,
             dropout_prob, dropout_seed,
             M, N, BLOCK: tl.constexpr):
    offs_m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    X = X + offs_m[:, None] * stride_x1 + offs_n[None, :] * stride_x2
    x = tl.load(X, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    
    offsets = offs_m[:, None] * M + offs_n[None, :]
    x = apply_dropout(x, offsets, dropout_prob, dropout_seed)
    
    O = O + offs_m[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(O, x, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def dropout(x, p, seed):
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK = 16
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    _dropout[grid](x, o, 
                   x.stride(0), x.stride(1), 
                   o.stride(0), o.stride(1),
                   p, seed, M, N, BLOCK=BLOCK)
    return o

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(100, 100, device='cuda')
    out = dropout(x, 0.5, 0)
    print()
