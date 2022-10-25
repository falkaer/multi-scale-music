from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from seq.rq_transformer import FeedForward, RQTransformer, Scale, Transformer, TransformerBlock

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
    
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.):
        super().__init__()
        
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        
        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                Rearrange('b n c -> b c n'),
                nn.Conv1d(dim, inner_dim * 2, 1),
                nn.GLU(dim=1),
                DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
                Swish(),
                nn.Conv1d(inner_dim, dim, 1),
                Rearrange('b c n -> b n c'),
                nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

def calc_same_padding_dilated(kernel_size, dilation):
    pad = kernel_size // 2 * dilation
    return (pad, pad - (kernel_size + 1) % 2)

class DilatedDepthWiseConv1d(DepthWiseConv1d):
    def __init__(self, chan_in, chan_out, kernel_size, padding, dilation=1):
        super().__init__(chan_in, chan_out, kernel_size, padding)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, dilation=dilation, groups=chan_in)
    
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class DilatedConformerConvModule(ConformerConvModule):
    def __init__(self,
                 dim,
                 causal=False,
                 dilation=1,
                 expansion_factor=2,
                 kernel_size=31,
                 dropout=0.):
        super().__init__(dim, causal, expansion_factor, kernel_size, dropout)
        
        inner_dim = dim * expansion_factor
        padding = calc_same_padding_dilated(kernel_size, dilation) if not causal else (
            dilation * (kernel_size - 1), 0)
        
        # replace DepthWiseConv1d
        self.net[4] = DilatedDepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                                             padding=padding, dilation=dilation)

class ConformerBlock(TransformerBlock):
    def __init__(self,
                 *,
                 dim,
                 ff_mult=4,
                 ff_dropout=0.,
                 ff_prenorm=None,
                 ff_postnorm=Scale,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 conv_dropout=0.,
                 causal=True,
                 **kwargs):
        super().__init__(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout,
                         ff_prenorm=ff_prenorm, ff_postnorm=ff_postnorm,
                         causal=causal, **kwargs)
        self.pre_ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, prenorm=ff_prenorm, postnorm=ff_postnorm)
        self.conv = ConformerConvModule(dim=dim,
                                        expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size,
                                        dropout=conv_dropout,
                                        causal=causal)
    
    def forward(self, x, cache=None):
        x = self.prenorm(x)
        x = self.pre_ff(x) + x
        x = self.attn(x, cache=cache) + x
        x = self.midnorm(x)
        x = self.conv(x) + x
        x = self.ff(x) + x
        return self.postnorm(x)

class DilatedConformerBlock(ConformerBlock):
    def __init__(self,
                 *,
                 dim,
                 dilation,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 conv_dropout=0.,
                 causal=True,
                 **kwargs):
        super().__init__(dim=dim,
                         conv_expansion_factor=conv_expansion_factor,
                         conv_kernel_size=conv_kernel_size,
                         conv_dropout=conv_dropout,
                         causal=causal, **kwargs)
        self.conv = DilatedConformerConvModule(dim=dim, dilation=dilation,
                                               expansion_factor=conv_expansion_factor,
                                               kernel_size=conv_kernel_size,
                                               dropout=conv_dropout,
                                               causal=causal)

class DilatedConformer(Transformer):
    def __init__(self,
                 *,
                 layers,
                 max_dilation=5,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 conv_dropout=0.,
                 **kwargs):
        super().__init__(layers=layers, **kwargs)
        Block = partial(DilatedConformerBlock,
                        conv_expansion_factor=conv_expansion_factor,
                        conv_kernel_size=conv_kernel_size,
                        conv_dropout=conv_dropout)
        self.blocks = nn.ModuleList([Block(dilation=2 ** (i % max_dilation), **kwargs) for i in range(layers)])

class RQDilatedConformer(RQTransformer):
    def __init__(self,
                 *,
                 quantizer,
                 inner_dim,
                 time_layers,
                 resolution_layers,
                 max_dilation=5,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 conv_dropout=0.,
                 embed_dropout=0.,
                 **kwargs):
        super().__init__(quantizer=quantizer, inner_dim=inner_dim, time_layers=time_layers,
                         resolution_layers=resolution_layers, embed_dropout=embed_dropout, **kwargs)
        self.time_transformer = DilatedConformer(
                dim=inner_dim,
                layers=time_layers,
                max_dilation=max_dilation,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_dropout=conv_dropout,
                **kwargs
        )

Conformer = partial(DilatedConformer, max_dilation=1)
RQConformer = partial(RQDilatedConformer, max_dilation=1)