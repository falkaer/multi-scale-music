# MIT License
# 
# Copyright (c) 2022 Rishikesh (ऋषिकेश) and Tomoki Hayashi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from https://github.com/rishikksh20/Avocodo-pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.signal as sig
from einops import rearrange

# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN

class PQMF(nn.Module):
    def __init__(self, subbands, taps=62, beta=9.0, cutoff=None):
        super().__init__()
        
        if cutoff is None:
            cutoff = optimize_pqmf_cutoff(subbands, taps, beta)
        
        self.subbands = subbands
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta
        
        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((subbands, len(QMF)))
        G = np.zeros((subbands, len(QMF)))
        for k in range(subbands):
            constant_factor = ((2 * k + 1)
                               * (np.pi / (2 * subbands))
                               * (np.arange(taps + 1) - (taps / 2)))
            phase = (-1) ** k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)
            
            G[k] = 2 * QMF * np.cos(constant_factor - phase)
        
        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()
        
        self.register_buffer('H', H)
        self.register_buffer('G', G)
        
        up_filter = torch.zeros((subbands, subbands, subbands))
        for k in range(subbands):
            up_filter[k, k, 0] = self.subbands
        self.register_buffer('up_filter', up_filter)
        self.pad_fn = nn.ConstantPad1d(self.taps // 2, 0)
    
    def analysis(self, x):
        return F.conv1d(self.pad_fn(x), self.H, stride=self.subbands)
    
    def synthesis(self, x):
        x = F.conv_transpose1d(x, self.up_filter, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.G)
    
    def analysis_lowpass(self, x):
        return F.conv1d(self.pad_fn(x), self.H[:1], stride=self.subbands)

class PQMFAnalysis(PQMF):
    def forward(self, x):
        return self.analysis(x)

class PQMFSynthesis(PQMF):
    def forward(self, x):
        return self.synthesis(x)

class PQMFAnalysisLowPass(PQMF):
    def forward(self, x):
        return self.analysis_lowpass(x)

# https://ieeexplore.ieee.org/abstract/document/681427
def optimize_pqmf_cutoff(subbands, taps, beta):
    import scipy.optimize as optimize
    
    def _objective(cutoff):
        h_proto = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        conv_h_proto = np.convolve(h_proto, h_proto[::-1], mode='full')
        length_conv_h = conv_h_proto.shape[0]
        half_length = length_conv_h // 2
        
        check_steps = np.arange(half_length // (2 * subbands)) * 2 * subbands
        _phi_new = conv_h_proto[half_length:][check_steps]
        phi_new = np.abs(_phi_new[1:]).max()
        # Since phi_new is not convex, This value should also be considered. 
        diff_zero_coef = np.abs(_phi_new[0] - 1 / (2 * subbands))
        
        return phi_new + diff_zero_coef
    
    ret = optimize.minimize(_objective, np.array([0.01]),
                            bounds=optimize.Bounds(0.01, 0.99))
    return ret.x[0]

if __name__ == '__main__':
    torch.manual_seed(0)
    
    X = torch.randn(1, 1, 16000, device='cuda')
    pqmf = PQMF(2, taps=62, beta=10.0).cuda()
    
    def l2norm(X):
        return torch.mean(X ** 2)
    
    xdown = pqmf.analysis(X)
    xdown_low, xdown_high = xdown.unbind(dim=1)
    print(xdown_low.shape)
    print(xdown_low)