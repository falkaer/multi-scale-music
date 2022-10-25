import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.utils.checkpoint import checkpoint

from pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D
from pqmf import PQMFAnalysis, PQMFAnalysisLowPass, PQMFSynthesis
from util import init_module

# TODO: make non-causal version (is there a consistent way to just 
# TODO: change CausalConv to Conv and have padding just work?)

# Generator

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)
    
    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')
        
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
                x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
                x, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)[..., :-self.causal_padding]

def ConvDownsample(channels, stride):
    return CausalConv1d(in_channels=channels // 2, out_channels=channels,
                        kernel_size=2 * stride, stride=stride)

def ConvUpsample(channels, stride):
    return CausalConvTranspose1d(in_channels=2 * channels, out_channels=channels,
                                 kernel_size=2 * stride, stride=stride)

def PQMFDownsample(channels, stride):
    return nn.Sequential(PQMFAnalysis(subbands=stride),
                         CausalConv1d(in_channels=channels // 2 * stride,
                                      out_channels=channels,
                                      kernel_size=1))

def PQMFUpsample(channels, stride):
    return nn.Sequential(CausalConv1d(in_channels=2 * channels,
                                      out_channels=channels * stride,
                                      kernel_size=1),
                         PQMFSynthesis(subbands=stride))

def PQMFLowPassDownsample(channels, stride):
    return nn.Sequential(PQMFAnalysisLowPass(subbands=stride),
                         CausalConv1d(in_channels=channels // 2, out_channels=channels,
                                      kernel_size=1))

def PixelShuffleDownsample(channels, stride):
    return nn.Sequential(PixelUnshuffle1D(downscale_factor=stride),
                         CausalConv1d(in_channels=channels // 2 * stride,
                                      out_channels=channels,
                                      kernel_size=1))

def PixelShuffleUpsample(channels, stride):
    return nn.Sequential(CausalConv1d(in_channels=2 * channels,
                                      out_channels=channels * stride,
                                      kernel_size=1),
                         PixelShuffle1D(upscale_factor=stride))

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, activation):
        super().__init__()
        self.layers = nn.Sequential(
                CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=7, dilation=dilation),
                activation(out_channels),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=1))
        self.post_act = activation(out_channels)
    
    def reset_parameters(self, init_fun):
        init_module(self.layers[0], init_fun)
        # fixup init: initialize the second layer to zero
        init.zeros_(self.layers[2].weight)
        init.zeros_(self.layers[2].bias)
    
    def forward(self, x):
        return self.post_act(x + self.layers(x))

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride, activation, downsample):
        super().__init__()
        self.layers = nn.Sequential(
                ResidualUnit(in_channels=out_channels // 2,
                             out_channels=out_channels // 2, dilation=1, activation=activation),
                ResidualUnit(in_channels=out_channels // 2,
                             out_channels=out_channels // 2, dilation=3, activation=activation),
                ResidualUnit(in_channels=out_channels // 2,
                             out_channels=out_channels // 2, dilation=9, activation=activation),
                downsample(out_channels, stride),
                activation(out_channels))
    
    def reset_parameters(self, init_fun):
        for l in self.layers[:-2]:
            l.reset_parameters(init_fun)
        init_module(self.layers[-2], init_fun)
    
    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride, activation, upsample):
        super().__init__()
        self.layers = nn.Sequential(
                upsample(out_channels, stride),
                activation(out_channels),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=1, activation=activation),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=3, activation=activation),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=9, activation=activation))
    
    def reset_parameters(self, init_fun):
        init_module(self.layers[0], init_fun)
        for l in self.layers[2:]:
            l.reset_parameters(init_fun)
    
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, C, D, activation, downsample, checkpoints=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                    CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
                    activation(C)
            ),
            EncoderBlock(out_channels=2 * C, stride=2, activation=activation, downsample=downsample),
            EncoderBlock(out_channels=4 * C, stride=4, activation=activation, downsample=downsample),
            EncoderBlock(out_channels=8 * C, stride=5, activation=activation, downsample=downsample),
            EncoderBlock(out_channels=16 * C, stride=8, activation=activation, downsample=downsample),
            CausalConv1d(in_channels=16 * C, out_channels=D, kernel_size=3)
        ])
        self.checkpoints = checkpoints
    
    def reset_parameters(self, init_fun=None):
        if init_fun is None:
            init_fun = lambda x, *args: init.kaiming_uniform_(x)
        init_module(self.blocks[0][0], init_fun)
        for block in self.blocks[1:-1]:
            block.reset_parameters(init_fun)
        init_module(self.blocks[-1], init_fun)
    
    def forward(self, x):
        fmaps = []
        for i, block in enumerate(self.blocks):
            if i < self.checkpoints:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
            fmaps.append(x)
        return x, fmaps[:-1]

class Decoder(nn.Module):
    def __init__(self, C, D, activation, upsample, checkpoints=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                    CausalConv1d(in_channels=D, out_channels=16 * C, kernel_size=7),
                    activation(16 * C)
            ),
            DecoderBlock(out_channels=8 * C, stride=8, activation=activation, upsample=upsample),
            DecoderBlock(out_channels=4 * C, stride=5, activation=activation, upsample=upsample),
            DecoderBlock(out_channels=2 * C, stride=4, activation=activation, upsample=upsample),
            DecoderBlock(out_channels=C, stride=2, activation=activation, upsample=upsample),
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        ])
        self.checkpoints = checkpoints
    
    def reset_parameters(self, init_fun=None):
        if init_fun is None:
            init_fun = lambda x, *args: init.kaiming_uniform_(x)
        init_module(self.blocks[0][0], init_fun)
        for block in self.blocks[1:-1]:
            block.reset_parameters(init_fun)
        init_module(self.blocks[-1], init_fun)
    
    def forward(self, x):
        fmaps = []
        for i, block in enumerate(self.blocks):
            if len(self.blocks) - i - 1 < self.checkpoints:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
            fmaps.append(x)
        return x, fmaps[:-1]

if __name__ == '__main__':
    from snake import Snake, snake_kaiming_normal_, snake_kaiming_uniform_
    
    # torch.manual_seed(0)
    x = torch.randn(1, 1, 16000, device='cuda')
    
    alpha_init = 0.5
    act_fun = lambda c: Snake(c, init=alpha_init, correction=None)
    enc = Encoder(C=16, D=128, activation=act_fun, downsample=ConvDownsample, checkpoints=2).cuda()
    dec = Decoder(C=32, D=128, activation=act_fun, upsample=ConvUpsample, checkpoints=2).cuda()
    
    with torch.cuda.amp.autocast():
        z, enc_fmaps = enc(x)
        x_hat, dec_fmaps = dec(z)
    
    print()
