import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F

from pqmf import PQMFAnalysisLowPass

# Wave-based Discriminator
class WaveDiscriminatorBlock(nn.Module):
    def __init__(self, C, activation):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                    nn.ReflectionPad1d(7),
                    nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15),
                    activation(16)
            ),
            nn.Sequential(
                    nn.Conv1d(in_channels=16, out_channels=2 * C, kernel_size=41,
                              stride=4, padding=20, groups=4),
                    activation(2 * C)
            ),
            nn.Sequential(
                    nn.Conv1d(in_channels=2 * C, out_channels=8 * C, kernel_size=41,
                              stride=4, padding=20, groups=16),
                    activation(8 * C)
            ),
            nn.Sequential(
                    nn.Conv1d(in_channels=8 * C, out_channels=32 * C, kernel_size=41,
                              stride=4, padding=20, groups=64),
                    activation(32 * C)
            ),
            nn.Sequential(
                    nn.Conv1d(in_channels=32 * C, out_channels=32 * C, kernel_size=41,
                              stride=4, padding=20, groups=256),
                    activation(32 * C)
            ),
            nn.Sequential(
                    nn.Conv1d(in_channels=32 * C, out_channels=32 * C, kernel_size=5,
                              stride=1, padding=2),
                    activation(32 * C)
            ),
            nn.Conv1d(in_channels=32 * C, out_channels=1, kernel_size=3, stride=1,
                      padding=1)
        ])
    
    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)
        return rearrange(x, 'b 1 N -> b N'), fmaps

class WaveDiscriminator(nn.Module):
    def __init__(self, C, num_D, downsampling_factor=2, downsample=None, activation=None):
        super().__init__()
        
        if activation is None:
            activation = lambda _: nn.LeakyReLU(negative_slope=0.2)
        if downsample is None:
            downsample = lambda f: nn.AvgPool1d(kernel_size=2 * f,
                                                stride=f, padding=1,
                                                count_include_pad=False)
        
        self.discs = nn.ModuleList([WaveDiscriminatorBlock(C, activation) for _ in range(num_D)])
        self.downsamplers = nn.ModuleList([nn.Identity()])
        for i in range(1, num_D):
            self.downsamplers.append(downsample(downsampling_factor ** i))
    
    def reset_parameters(self, init_fun=None):
        pass
    
    def forward(self, x, *_):
        x = rearrange(x, 'b n -> b 1 n')
        all_scores, all_fmaps = [], []
        for down, disc in zip(self.downsamplers, self.discs):
            scores, fmaps = disc(down(x))
            all_scores.append(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps

# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()
        
        self.s_t = s_t
        self.s_f = s_f
        
        self.layers = nn.Sequential(
                nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=N,
                        kernel_size=(3, 3),
                        padding='same'
                ),
                nn.ELU(),
                nn.Conv2d(
                        in_channels=N,
                        out_channels=m * N,
                        kernel_size=(s_f + 2, s_t + 2),
                        stride=(s_f, s_t)
                )
        )
        
        self.skip_connection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=m * N,
                kernel_size=(1, 1), stride=(s_f, s_t)
        )
    
    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t + 1, 0, self.s_f + 1, 0])) + self.skip_connection(x)

class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=32, N=C, m=2, s_t=1, s_f=2),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=2 * C, N=2 * C, m=2, s_t=2, s_f=2),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=4 * C, N=4 * C, m=1, s_t=1, s_f=2),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=4 * C, N=4 * C, m=2, s_t=2, s_f=2),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=8 * C, N=8 * C, m=1, s_t=1, s_f=2),
                    nn.ELU()
            ),
            nn.Sequential(
                    ResidualUnit2d(in_channels=8 * C, N=8 * C, m=2, s_t=2, s_f=2),
                    nn.ELU()
            ),
            nn.Conv2d(in_channels=16 * C, out_channels=1,
                      kernel_size=(F_bins // 2 ** 6, 1))
        ])
        
        self.register_buffer('window', torch.hann_window(window_length=1024))
    
    def reset_parameters(self, init_fun=None):
        pass
    
    def forward(self, x, *_):
        stft = torch.stft(x, n_fft=1024, hop_length=256, window=self.window, return_complex=True)
        x = rearrange(torch.view_as_real(stft), 'b f n c -> b c f n')
        all_fmaps = []
        for layer in self.layers:
            x = layer(x)
            all_fmaps.append(x)
        return [rearrange(x, 'b 1 1 N -> b N')], all_fmaps
