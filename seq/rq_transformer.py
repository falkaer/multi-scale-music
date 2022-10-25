from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import init as init

from einops import rearrange, repeat

from seq.attention import AttentionCache, simple_attention, flash_attention, MultiheadAttention
from seq.sampling import sample_logits
from util import OversamplingDataset, default, exists, overridekwargs

class ScaleNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones())
    
    def forward(self, x):
        return self.scale * F.normalize(x, dim=-1)

# rezero
class Scale(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
    
    def forward(self, x):
        return self.scale * x

def FeedForward(*, dim, mult=4, dropout=0., prenorm=None, postnorm=None, activation=None):
    activation = default(activation, lambda _: nn.GELU())
    return nn.Sequential(
            default(prenorm, nn.Identity)(dim),
            nn.Linear(dim, dim * mult),
            Rearrange('b n d -> b d n'),
            activation(dim * mult),
            Rearrange('b d n -> b n d'),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim * mult, dim),
            default(postnorm, nn.Identity)(dim)
    )

class TransformerBlock(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 dim_head=64,
                 heads=8,
                 attention=flash_attention,
                 attn_dropout=0.,
                 postattn_dropout=0.,
                 attn_prenorm=nn.LayerNorm,
                 attn_postnorm=None,
                 attn_softmax_scale=None,
                 ff_dropout=0.,
                 ff_mult=4,
                 ff_prenorm=nn.LayerNorm,
                 ff_postnorm=None,
                 activation=None,
                 prenorm=None,
                 midnorm=None,
                 postnorm=None,
                 causal=True,
                 use_rotary=False,
                 use_alibi=True):
        super().__init__()
        self.attn = MultiheadAttention(dim=dim, dim_head=dim_head, heads=heads, attn_dropout=attn_dropout,
                                       attention=attention, prenorm=attn_prenorm, postnorm=attn_postnorm,
                                       softmax_scale=attn_softmax_scale, postattn_dropout=postattn_dropout,
                                       causal=causal, use_rotary=use_rotary, use_alibi=use_alibi)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, activation=activation,
                              prenorm=ff_prenorm, postnorm=ff_postnorm)
        self.prenorm = default(prenorm, nn.Identity)(dim)
        self.midnorm = default(midnorm, nn.Identity)(dim)
        self.postnorm = default(postnorm, nn.Identity)(dim)
    
    def forward(self, x, cache=None):
        x = self.prenorm(x)
        x = self.attn(x, cache=cache) + x
        x = self.midnorm(x)
        x = self.ff(x) + x
        return self.postnorm(x)

class Transformer(nn.Module):
    def __init__(
            self,
            *,
            layers,
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(**kwargs) for _ in range(layers)])
    
    def forward(self, x, caches=None):
        if caches is None:
            caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, caches):
            x = block(x, cache=cache)
        return x

# main class
class RQTransformer(nn.Module):
    def __init__(
            self,
            *,
            quantizer,
            inner_dim,
            time_layers,
            resolution_layers,
            embed_dropout=0.,
            **kwargs
    ):
        super().__init__()
        first_vq, *rest_vq = quantizer.layers
        num_quantizers = len(quantizer.layers)
        codebook_size, dim = first_vq.codebook.shape
        
        self.quantizer = quantizer.eval()
        
        self.embed_dropout = nn.Dropout(embed_dropout, inplace=True)
        self.time_start_token = nn.Parameter(torch.randn(inner_dim))
        self.time_proj = nn.Linear(dim, inner_dim)
        self.time_transformer = Transformer(
                dim=inner_dim,
                layers=time_layers,
                **kwargs
        )
        
        self.resolution_pos_emb = nn.Embedding(num_quantizers, inner_dim)
        self.resolution_proj = nn.Linear(dim, inner_dim)
        self.resolution_transformer = Transformer(
                dim=inner_dim,
                layers=resolution_layers,
                **overridekwargs(kwargs,
                                 attention=simple_attention,
                                 use_rotary=False,
                                 use_alibi=False)
        )
        
        self.to_logits = nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, codebook_size)
        )
    
    def train(self, mode=True):
        super().train(mode)
        self.quantizer.eval()
        return self
    
    @torch.no_grad()
    def generate(self, sample_len, prime, temperature=1.,
                 filter_top_k=0.9, filter_top_p=None):
        B, L, resolution = prime.shape
        device = prime.device
        seq = prime.new_empty(B, sample_len, resolution, dtype=torch.long)  # b n q
        time_caches = [AttentionCache(L + sample_len) for _ in self.time_transformer.blocks]
        resolution_pos = self.resolution_pos_emb(torch.arange(resolution, device=device))
        quantized = self.quantizer.embed_codes(prime, dim=-1)
        for i in range(sample_len):
            time_tokens = self.time_proj(quantized)
            time_tokens = torch.cat((
                repeat(self.time_start_token, 'd -> b 1 d', b=B),
                time_tokens
            ), dim=1) if i == 0 else time_tokens
            time_tokens = self.time_transformer(time_tokens, caches=time_caches)[:, [-1]]
            time_tokens = self.embed_dropout(time_tokens)
            quantized = 0.
            resolution_caches = [AttentionCache(resolution) for _ in self.resolution_transformer.blocks]
            for r, vq, pos_emb in zip(range(resolution), self.quantizer.layers, resolution_pos):
                resolution_tokens = time_tokens if r == 0 else self.resolution_proj(quantized)
                resolution_tokens = resolution_tokens + pos_emb
                resolution_tokens = self.resolution_transformer(resolution_tokens, caches=resolution_caches)
                codes = sample_logits(self.to_logits(resolution_tokens), temperature, filter_top_k, filter_top_p)
                quantized = quantized + vq.embed_codes(codes)
                seq[:, i, r] = rearrange(codes, 'b 1 -> b')
        return seq
    
    # keep for timing later
    @torch.no_grad()
    def generate_no_cache(self, sample_len, prime, temperature=1.,
                          filter_top_k=0.9, filter_top_p=None):
        B, L, resolution = prime.shape
        device = prime.device
        seq = prime.new_empty(B, sample_len, resolution, dtype=torch.long)  # b n q
        resolution_pos = self.resolution_pos_emb(torch.arange(resolution, device=device))
        quantized = self.quantizer.embed_codes(prime, dim=-1)
        for i in range(sample_len):
            time_tokens = self.time_proj(quantized)
            time_tokens = torch.cat((
                repeat(self.time_start_token, 'd -> b 1 d', b=B),
                time_tokens
            ), dim=1)
            time_tokens = self.time_transformer(time_tokens)[:, [-1]]
            time_tokens = self.embed_dropout(time_tokens)
            quantized_out = 0.
            partial_embeds = quantized.new_empty(B, 0, quantized.shape[2])
            for r, vq in zip(range(resolution), self.quantizer.layers):
                resolution_tokens = torch.cat((
                    time_tokens,
                    self.resolution_proj(partial_embeds)
                ), dim=1)
                resolution_tokens = resolution_tokens + resolution_pos[:r + 1]
                resolution_tokens = self.resolution_transformer(resolution_tokens)[:, [-1]]
                codes = sample_logits(self.to_logits(resolution_tokens), temperature, filter_top_k, filter_top_p)
                quantized_out = quantized_out + vq.embed_codes(codes)
                partial_embeds = torch.cat((partial_embeds, quantized_out), dim=1)
                seq[:, i, r] = rearrange(codes, 'b 1 -> b')
            quantized = torch.cat((quantized, quantized_out), dim=1)
        
        return seq
    
    def forward(self, codes):
        # codes: B x N x Nq
        assert codes.ndim == 3
        assert not self.quantizer.training
        
        b, time, resolution, device = *codes.shape, codes.device
        assert resolution <= len(self.quantizer.layers), 'resolution dimension must be <= number of quantizers'
        
        # get code embeddings
        quantized, partial_embeds = self.quantizer.embed_codes(codes, dim=-1, return_partials=True)
        partial_embeds = torch.cumsum(partial_embeds, dim=2)[..., :-1, :]
        quantized = self.time_proj(quantized)
        
        time_tokens = torch.cat((
            repeat(self.time_start_token, 'd -> b 1 d', b=b),
            quantized[:, :-1, :]  # drop last embedding
        ), dim=1)
        time_tokens = self.embed_dropout(time_tokens)
        time_tokens = self.time_transformer(time_tokens)
        time_tokens = rearrange(time_tokens, 'b n d -> b n 1 d')
        
        # time tokens become the start of the resolution transformer input
        resolution_pos = self.resolution_pos_emb(torch.arange(resolution, device=device))
        resolution_tokens = torch.cat((time_tokens, self.resolution_proj(partial_embeds)), dim=2)
        resolution_tokens = resolution_tokens + resolution_pos
        resolution_tokens = rearrange(resolution_tokens, 'b n q d -> (b n) q d')
        resolution_tokens = self.resolution_transformer(resolution_tokens)
        resolution_tokens = rearrange(resolution_tokens, '(b n) q d -> b n q d', b=b)
        return self.to_logits(resolution_tokens)  # b n q C

def tfixup_init_(transformer):
    def init_fun_(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            D = module.weight.shape[-1]
            std = D ** -1 / 2
            init.normal_(module.weight.data, mean=0, std=std)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight.data)
            init.zeros_(module.bias.data)
    
    def scale_fun_(module, coeff):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.mul_(coeff)
    
    ts = (transformer.time_transformer, transformer.resolution_transformer)
    coeffs = [(9 * len(t.blocks)) ** -1 / 4 for t in ts]
    
    D = transformer.time_start_token.shape[-1]
    std = D ** -1 / 2
    init.normal_(transformer.time_start_token, mean=0, std=std)
    init.normal_(transformer.time_proj.weight.data, mean=0, std=std)
    init.normal_(transformer.resolution_proj.weight.data, mean=0, std=std)
    
    init_fun_(transformer.resolution_pos_emb)
    scale_fun_(transformer.resolution_pos_emb, coeffs[1])
    
    for t, coeff in zip(ts, coeffs):
        t.apply(init_fun_)
        _scale_fun_ = partial(scale_fun_, coeff=coeff)
        for block in t.blocks:
            _scale_fun_(block.attn.to_v)
            _scale_fun_(block.attn.to_out)
            block.ff.apply(_scale_fun_)
            if hasattr(block, 'pre_ff'):
                block.pre_ff.apply(_scale_fun_)
            if hasattr(block, 'conv'):
                block.conv.apply(_scale_fun_)

def small_init_(transformer):
    def init_fun_(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            init.normal_(module.weight.data, mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                init.zeros_(module.bias.data)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight.data)
            init.zeros_(module.bias.data)
    
    transformer.apply(init_fun_)

if __name__ == '__main__':
    from seq.attention import flash_attention
    from vq import ResidualVQ
    from util import count_parameters
    from embedding_dataset import EmbeddingDataset
    from maestro_dataset import MAESTRO
    
    torch.manual_seed(0)
    quantizer = ResidualVQ(dim=256, num_quantizers=4, codebook_size=16384).eval()
    model = RQTransformer(inner_dim=256, quantizer=quantizer, time_layers=6, resolution_layers=4,
                          attention=simple_attention,
                          use_alibi=False, use_rotary=False).cuda()
    
    # print('Parameter counts:\n',
    #       'RQ-transformer: {}\n'.format(count_parameters(model)),
    #       'Time transformer: {}\n'.format(count_parameters(model.time_transformer)),
    #       'Resolution transformer: {}\n'.format(count_parameters(model.resolution_transformer)))
    
    print('Parameter counts:\n',
          'RQ-transformer: {}\n'.format(count_parameters(model)),
          'Time transformer: {}\n'.format(count_parameters(model.time_transformer)))
    
    train_data = MAESTRO(split='train')
    train_data = EmbeddingDataset(train_data, 'checkpoints', 'fiery-snow-315', 1679001, num_embeddings=1024)
    train_data = OversamplingDataset(train_data, oversampling_factor=100)
    
    e = torch.stack([train_data[0][0][:5 * 50] for i in range(10)], dim=0).cuda()
    print(e.shape)
    _, codes, _, _, dists_sq = quantizer(e, return_distances=True)
    # with torch.cuda.amp.autocast():
    #     y = model(codes)
    print(codes.shape)
    model.eval()
    gen1 = model.generate(100, prime=codes, temperature=0)
    gen2 = model.generate_no_cache(100, prime=codes, temperature=0)
    print(gen1)
    print(gen2)
    print(torch.all(gen1 == gen2))
