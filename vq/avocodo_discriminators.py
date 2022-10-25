import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from pqmf import PQMFAnalysis
from util import init_module

from itertools import repeat

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class CoMBD(nn.Module):
    def __init__(self, filters, kernels, groups, strides, activation):
        super().__init__()
        self.blocks = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.blocks.append(nn.Sequential(
                    nn.Conv1d(init_channel, f, k, s, padding=get_padding(k, 1), groups=g),
                    activation(f)
            ))
            init_channel = f
        self.post_conv = nn.Conv1d(filters[-1], 1, 3, 1, padding=get_padding(3, 1))
    
    def reset_parameters(self, init_fun):
        for block in self.blocks:
            init_module(block[0], init_fun)
        init_module(self.post_conv, init_fun)
    
    def forward(self, x):
        fmaps = []
        for l in self.blocks:
            x = l(x)
            fmaps.append(x)
        x = self.post_conv(x)
        x = rearrange(x, 'b ... -> b (...)')
        return x, fmaps

# class MDC(nn.Module):
#     def __init__(self, in_channel, channel, kernel, stride, dilations, activation):
#         super().__init__()
#         self.kernel = kernel
#         self.dilations = dilations
#         
#         num_dilations = len(dilations)
#         self.weights = nn.Parameter(torch.ones(num_dilations) / num_dilations)
#         self.weight = nn.Parameter(torch.empty(channel, in_channel, kernel))
#         self.post_conv = nn.Sequential(
#                 nn.Conv1d(channel, channel, 3, stride=stride, padding=get_padding(3, 1)),
#                 activation(channel)
#         )
#         
#         # standard torch init
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#     
#     def reset_parameters(self, init_fun):
#         self.weights.data.fill_(1 / len(self.dilations))
#         init_fun(self.weight)
#         # self.weight.data *= len(self.dilations)
#         init_module(self.post_conv[0], init_fun)
#     
#     def forward(self, x):
#         xs = []
#         # in case of weight norm, materialize only once
#         weight = self.weight
#         for d in self.dilations:
#             pad = get_padding(weight.size(-1), d)
#             xs.append(F.conv1d(x, weight, bias=None, padding=pad, dilation=d))
#         x = torch.sum(self.weights * torch.stack(xs, dim=-1), dim=-1)
#         return self.post_conv(x)

class MDC(nn.Module):
    def __init__(self, in_channel, channel, kernel, stride, dilations, activation):
        super().__init__()
        self.kernel = kernel
        self.dilations = dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channel, channel, kernel, stride=1,
                      padding=get_padding(kernel, d),
                      dilation=d) for d in dilations
        ])
        self.post_conv = nn.Sequential(
                nn.Conv1d(channel, channel, 3, stride=stride, padding=get_padding(3, 1)),
                activation(channel)
        )
    
    def reset_parameters(self, init_fun):
        init_module(self.post_conv[0], init_fun)
        for conv in self.convs:
            init_module(conv, init_fun)
    
    def forward(self, x):
        xs = [conv(x) for conv in self.convs]
        x = torch.sum(torch.stack(xs, dim=-1), dim=-1)
        return self.post_conv(x)

class SubBandDiscriminator(torch.nn.Module):
    def __init__(self, init_channel, channels, kernel, strides, dilations, activation):
        super().__init__()
        self.mdcs = torch.nn.ModuleList()
        for c, s, d in zip(channels, strides, dilations):
            self.mdcs.append(MDC(init_channel, c, kernel, s, d, activation))
            init_channel = c
        self.post_conv = nn.Conv1d(init_channel, 1, 3, padding=get_padding(3, 1))
    
    def reset_parameters(self, init_fun):
        for mdc in self.mdcs:
            mdc.reset_parameters(init_fun)
        init_module(self.post_conv, init_fun)
    
    def forward(self, x):
        fmap = []
        for l in self.mdcs:
            x = l(x)
            fmap.append(x)
        x = self.post_conv(x)
        x = rearrange(x, 'b ... -> b (...)')
        return x, fmap

class MultiCoMBDiscriminator(torch.nn.Module):
    def __init__(self, C, fmap_channels, kernels=None, channels=None, groups=None, strides=None, activation=None):
        super().__init__()
        # C must be at least 4
        if kernels is None:
            kernels = [[7, 11, 11, 11, 11, 5], [11, 21, 21, 21, 21, 5], [15, 41, 41, 41, 41, 5]]
        if channels is None:
            channels = [C, 4 * C, 16 * C, 64 * C, 64 * C, 64 * C]
        if groups is None:
            groups = [1, 4, 16, 64, 256, 1]
        if strides is None:
            strides = [1, 1, 4, 4, 4, 1]
        if activation is None:
            activation = lambda _: nn.LeakyReLU(negative_slope=0.1)
        
        self.combds = torch.nn.ModuleList([
            CoMBD(filters=channels, kernels=kernel, groups=groups, strides=strides, activation=activation)
            for kernel in kernels
        ])
        
        # self.gen_proj = nn.ModuleList([
        #     nn.Conv1d(in_channels=fmap_channels[0], out_channels=1, kernel_size=7, stride=1, padding=3),
        #     nn.Conv1d(in_channels=fmap_channels[1], out_channels=1, kernel_size=7, stride=1, padding=3)
        # ])
        
        # TODO: are these optimal?
        self.pqmf_2 = PQMFAnalysis(subbands=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_8 = PQMFAnalysis(subbands=8, taps=192, cutoff=0.13, beta=10.0)
    
    def reset_parameters(self, init_fun=None):
        if init_fun is None:
            init_fun = lambda x, *args: init.kaiming_uniform_(x)
        for combd in self.combds:
            combd.reset_parameters(init_fun)
        # init_module(self.gen_proj[0], init_fun)
        # init_module(self.gen_proj[1], init_fun)
    
    def forward(self, x, gen_fmaps=None):
        all_scores = []
        all_fmaps = []
        
        x = rearrange(x, 'b n -> b 1 n')
        x8down = self.pqmf_8(x)[:, :1, :]
        x2down = self.pqmf_2(x)[:, :1, :]
        
        x8_score, x8_fmaps = self.combds[0](x8down)
        all_scores.append(x8_score)
        all_fmaps.extend(x8_fmaps)
        
        x2_score, x2_fmaps = self.combds[1](x2down)
        all_scores.append(x2_score)
        all_fmaps.extend(x2_fmaps)
        
        x_score, x_fmaps = self.combds[2](x)
        all_scores.append(x_score)
        all_fmaps.extend(x_fmaps)
        
        # feature match the downsampled waveforms to 
        # the internal activations in the generator
        # if gen_fmaps is not None:
        #     for gen_fmap, proj, combd in zip(gen_fmaps, self.gen_proj, self.combds):
        #         score, fmaps = combd(proj(gen_fmap))
        #         all_scores.append(score)
        #         all_fmaps.extend(fmaps)
        # else:
        #     all_scores.append(x8_score)
        #     all_fmaps.extend(x8_fmaps)
        #     all_scores.append(x2_score)
        #     all_fmaps.extend(x2_fmaps)
        
        return all_scores, all_fmaps

class MultiSubBandDiscriminator(torch.nn.Module):
    def __init__(self, C, freq_init_ch, n=16, m=64,
                 tkernels=None, tchannels=None, tstrides=None, tdilations=None, tsubbands=None,
                 fkernel=5, fchannels=None, fstride=None, fdilations=None, activation=None):
        super().__init__()
        
        if activation is None:
            activation = lambda _: nn.LeakyReLU(negative_slope=0.1)
        
        if tkernels is None:
            tkernels = [7, 5, 3]
        if tchannels is None:
            tchannels = [4 * C, 8 * C, 16 * C, 16 * C, 16 * C]
        if tstrides is None:
            tstrides = repeat([1, 1, 3, 3, 1], 3)
        if tdilations is None:
            tdilations = [repeat([5, 7, 11], 6),
                          repeat([3, 5, 7], 5),
                          repeat([1, 2, 3], 5)]
        if tsubbands is None:
            tsubbands = [6, 11, 16]
        
        self.tsdbs = nn.ModuleList([
            SubBandDiscriminator(tsubband, tchannels, tkernel, tstride, tdilation, activation=activation)
            for tsubband, tkernel, tstride, tdilation in zip(tsubbands, tkernels, tstrides, tdilations)
        ])
        
        if fchannels is None:
            fchannels = [2 * C, 4 * C, 8 * C, 8 * C, 8 * C]
        if fstride is None:
            fstride = [1, 1, 3, 3, 1]
        if fdilations is None:
            fdilations = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]]
        
        self.fsbd = SubBandDiscriminator(init_channel=freq_init_ch, channels=fchannels, kernel=fkernel,
                                         strides=fstride, dilations=fdilations, activation=activation)
        
        # TODO: try with automatic cutoff
        self.pqmf_n = PQMFAnalysis(subbands=n, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMFAnalysis(subbands=m, taps=256, cutoff=0.1, beta=9.0)
        self.tsubbands = tsubbands
    
    def reset_parameters(self, init_fun=None):
        if init_fun is None:
            init_fun = lambda x, *args: init.kaiming_uniform_(x)
        self.fsbd.reset_parameters(init_fun)
        for tsdb in self.tsdbs:
            tsdb.reset_parameters(init_fun)
    
    def forward(self, x, *_):
        all_scores = []
        all_fmaps = []
        
        # Time analysis
        xn = self.pqmf_n(x)
        
        for tsdb, tsubband in zip(self.tsdbs, self.tsubbands):
            scores, fmaps = tsdb(xn[:, :tsubband, :])
            all_scores.append(scores)
            all_fmaps.extend(fmaps)
        
        # Frequency analysis
        xm = rearrange(self.pqmf_m(x), 'b s t -> b t s')
        scores, fmaps = self.fsbd(xm)
        all_scores.append(scores)
        all_fmaps.extend(fmaps)
        
        return all_scores, all_fmaps

if __name__ == '__main__':
    from snake import Snake, snake_kaiming_normal_, snake_kaiming_uniform_
    from util import count_parameters
    
    # torch.manual_seed(0)
    x = torch.randn(1, 15 * 16000, device='cuda')
    
    alpha_init = 0.5
    act_fun = lambda c: Snake(c, init=alpha_init, correction='std')
    C = 16
    C_dec = 32
    
    # model = MultiCoMBDiscriminator(C=C, fmap_channels=[4 * C_dec, 2 * C_dec], activation=act_fun).cuda()
    model = MultiSubBandDiscriminator(C=C, freq_init_ch=x.size(-1) // 64).cuda()
    
    print(count_parameters(model))
    
    with torch.cuda.amp.autocast():
        outs, *_ = model(x)
    for out in outs:
        print(out)
        print(out.std())
    
    init_fun = lambda x: snake_kaiming_normal_(x, alpha_init, correction='std')
    model.reset_parameters(init_fun)
    
    print('WITH INIT')
    
    # 10,667,140
    # 12,347,140
    
    with torch.cuda.amp.autocast():
        outs, *_ = model(x)
    for out in outs:
        print(out)
        print(out.std())
