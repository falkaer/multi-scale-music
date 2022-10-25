import math
from functools import partial

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    
    # streaming logsumexp
    @triton.jit
    def _logsumexp(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
        rm = tl.program_id(0)
        alpha = tl.zeros((1,), tl.float32) + -float('inf')
        res = tl.zeros((1,), tl.float32)
        for bn in range(0, N, BLOCK_N):
            rn = bn + tl.arange(0, BLOCK_N)
            Xmn = X + rm * xm_stride + rn * xn_stride
            x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
            c = tl.max(x, axis=0)
            # correct the current sum and update the max
            res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
            alpha = tl.where(c > alpha, c, alpha)
            res += tl.sum(tl.exp(x - alpha), axis=0)
        out = tl.log(res) + alpha
        rm = tl.program_id(0) + tl.arange(0, 1)
        OUT = OUT + rm * out_stride
        tl.store(OUT, out)
    
    def logsumexp(input):
        assert input.is_cuda
        *dims, N = input.shape
        input = input.view(-1, N)
        out = input.new_empty(*dims).view(-1)
        M = input.shape[0]
        _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                         BLOCK_N=4096, num_warps=4)
        return out.view(*dims)
    
    # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    def softmax_(x):
        if not x.is_cuda:
            return torch.softmax(x, dim=-1, out=x)
        c = logsumexp(x)
        return x.sub_(c[..., None]).exp_()
    
except ImportError:
    logsumexp = partial(torch.logsumexp, dim=-1)
    softmax_ = lambda x: torch.softmax(x, dim=-1, out=x)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps)) if torch.is_tensor(t) else math.log(max(t, eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)

def top_k_logits_(logits, thres=0.5):
    if thres == 0:
        return logits
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    # return torch.topk(logits, k, dim=-1)
    top_logits, _ = torch.topk(logits, k, dim=-1)
    logits[logits < top_logits[..., [-1]]] = -float('inf')
    return logits

def top_p_probs_(probs, p):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1, out=sorted_probs)
    
    sorted_idx_remove_cond = cum_probs >= p
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill_(indices_to_remove, 0.0)
    return probs.div_(torch.sum(probs, dim=-1, keepdim=True))

@autocast(enabled=False)
def sample_logits(logits, temperature=1., filter_top_k=None, filter_top_p=None, bsize=128):
    if logits.shape[1] > bsize:
        s = partial(sample_logits, temperature=temperature, 
                    filter_top_k=filter_top_k, filter_top_p=filter_top_p, 
                    bsize=bsize)
        return torch.cat([s(x) for x in torch.split(logits, bsize, dim=1)], dim=1)
    logits = logits.to(dtype=torch.float, copy=True)
    if temperature == 0:
        return logits.argmax(dim=-1)
    if filter_top_k is not None:
        logits = top_k_logits_(logits, thres=filter_top_k)
    if filter_top_p is not None:
        # probs = F.softmax(logits / temperature, dim=-1)
        probs = softmax_(logits.mul_(1 / temperature))
        probs = top_p_probs_(probs, p=filter_top_p)
        *dims, C = probs.shape
        return torch.multinomial(probs.view(-1, C), num_samples=1).view(*dims)
    else:
        return gumbel_sample(logits, temperature, dim=-1)

@torch.no_grad()
def scheduled_sample(model, codes, sample_proba=0.1, temperature=1, filter_top_k=None, filter_top_p=None, passes=1, inplace=False):
    training = model.training
    model.eval()
    if not inplace:
        codes = codes.clone()
    for _ in range(passes):
        sample = sample_logits(model(codes),
                               temperature=temperature,
                               filter_top_k=filter_top_k,
                               filter_top_p=filter_top_p)
        p = torch.rand(sample.shape, device=codes.device)
        mask = p < sample_proba
        codes[mask] = sample[mask]
    model.train(training)
    return codes

if __name__ == '__main__':
    torch.manual_seed(0)
    logits = torch.randn(5, 1024, 16384, device='cuda')
    print(sample_logits(logits, temperature=0, filter_top_p=0.92))
    print(sample_logits(logits, temperature=0, filter_top_p=0.92, bsize=1000000))