from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math

# https://github.com/ofirpress/attention_with_linear_biases/issues/5
class AliBi(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        ratio = (2 ** (-2 ** -(math.log2(num_heads) - 3)))
        slopes = ratio ** torch.arange(1, num_heads + 1)
        self.register_buffer('slopes', slopes)
        self.register_buffer('bias', None)
    
    def forward(self, m, n):
        if self.bias is None or m > self.bias.size(1) or n > self.bias.size(2):
            M = max(m, n)
            offs_m = torch.arange(M, device=self.slopes.device)
            offs_n = torch.arange(M, device=self.slopes.device)
            bias = rearrange(offs_n, 'n -> 1 1 n') - rearrange(offs_m, 'm -> 1 m 1')
            bias = -torch.abs(bias)  # use symmetric alibi
            slopes = rearrange(self.slopes, 'h -> h 1 1')
            self.register_buffer('bias', slopes * bias)
        return self.bias[:, -m:, -n:]

alibi_cache = {}

def cached_alibi(h, device='cuda', dtype=torch.float32):
    if (h, device) in alibi_cache:
        return alibi_cache[(h, device, dtype)]
    a = AliBi(h).to(device=device, dtype=dtype)
    alibi_cache[(h, device, dtype)] = a
    return a

def causal_mask(m, n, device):
    triu_mask = torch.ones((m, n), dtype=torch.bool, device=device).triu_(n - m + 1)
    return triu_mask

def to_causal(mask):
    m, n = mask.shape[-2:]
    return mask.masked_fill(causal_mask(m, n, mask.device), float('-inf'))

mask_cache = {}

# reference implementation
def simple_attention(q, k, v, scale, causal=True, use_alibi=True, dropout=0.):
    sim = torch.einsum('b h m d, b h n d -> b h m n', q * scale, k)
    b, h, m, n = sim.shape
    mask = None
    if causal or use_alibi:
        mask_key = (m, n, h, q.device, q.dtype, causal, use_alibi)
        if mask_key in mask_cache:
            mask = mask_cache[mask_key]
        else:
            mask = sim.new_zeros(m, n, dtype=q.dtype)
            if use_alibi:
                alibi = cached_alibi(h, q.device)
                mask = mask + alibi(m, n)
            if causal:
                mask = to_causal(mask)
            mask_cache[mask_key] = mask
    if mask is not None:
        sim = sim + mask
    attn = sim.softmax(dim=-1)
    if dropout > 0:
        attn = F.dropout(attn, p=dropout)  # TODO: inplace?
    return torch.einsum('b h i j, b h j d -> b h i d', attn, v)

from util import default
from seq.flash_attention import flash_attention
from seq.rotary import RotaryEmbedding

class MultiheadAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, attn_dropout=0., postattn_dropout=0., attention=flash_attention,
                 prenorm=None, postnorm=None, softmax_scale=None, causal=True, use_rotary=False, use_alibi=False):
        super().__init__()
        self.scale = default(softmax_scale, dim_head ** -0.5)
        self.heads = heads
        self.attn_dropout = attn_dropout
        
        # TODO: attention dropout?
        self.attention = attention
        
        inner_dim = dim_head * heads
        self.causal = causal
        self.use_alibi = use_alibi
        self.prenorm = default(prenorm, nn.Identity)(dim)
        self.postnorm = default(postnorm, nn.Identity)(dim)
        self.rotary = RotaryEmbedding(dim_head) if use_rotary else lambda q, k: (q, k)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.drop_out = nn.Dropout(postattn_dropout, inplace=True) # funny
    
    def forward(self, x, context=None, cache=None):
        x = self.prenorm(x)
        context = default(context, x)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        if cache is not None:
            k, v = cache(k, v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, k = self.rotary(q, k)
        out = self.attention(q, k, v, self.scale, self.causal, self.use_alibi, self.attn_dropout)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.drop_out(self.to_out(out))
        return self.postnorm(out)

class AttentionCache(nn.Module):
    def __init__(self, target_len):
        super().__init__()
        self.target_len = target_len
        self.cur_len = 0
        self.register_buffer('k', None)
        self.register_buffer('v', None)
    
    def forward(self, k, v):
        B, L, D = k.shape
        if self.k is None:
            self.k = k.new_empty(B, self.target_len, D)
            self.v = v.new_empty(B, self.target_len, D)
        self.k[:, self.cur_len:self.cur_len + L] = k
        self.v[:, self.cur_len:self.cur_len + L] = v
        self.cur_len += L
        assert self.cur_len <= self.target_len
        return self.k[:, :self.cur_len], self.v[:, :self.cur_len]

# if __name__ == '__main__':
#     torch.manual_seed(0)
#     att1 = MultiheadAttention(256, attention=simple_attention, causal=True, use_alibi=True).cuda()
#     att2 = MultiheadAttention(256, attention=flash_attention, causal=True, use_alibi=True).cuda()
#     att2.prenorm = att1.prenorm
#     att2.to_q = att1.to_q
#     att2.to_k = att1.to_k
#     att2.to_v = att1.to_v
#     att2.to_out = att1.to_out
#     x = torch.randn(10, 8, 256, device='cuda', requires_grad=True)
#     
#     with torch.cuda.amp.autocast():
#         print(att1(x))
#         print(att2(x))

if __name__ == '__main__':
    torch.manual_seed(0)
    B, N, D = 1, 4, 2
    att = MultiheadAttention(D, attention=simple_attention, causal=True, use_alibi=True).cuda()
    seq = torch.randn(B, N, D, device='cuda', requires_grad=True)
    
    sample_len = 100
    cache = AttentionCache(N + sample_len)
    out1 = seq
    out2 = seq.new_empty(B, 0, D)
    sample2 = seq
    
    with torch.cuda.amp.autocast():
        for _ in range(sample_len):
            sample1 = att(out1)[:, [-1]]
            out1 = torch.cat((out1, sample1), dim=1)
            
            sample2 = att(sample2, cache=cache)[:, [-1]]
            out2 = torch.cat((out2, sample2), dim=1)
    
    out1 = out1[:, -sample_len:]
    print(out1)
    print(out1.shape)
    print(out2)
    print(out2.shape)
    
    print(torch.allclose(out1, out2, atol=1e-6))
    