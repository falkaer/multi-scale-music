import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np

from einops import rearrange
from torchaudio.transforms import MelScale

from librosa.filters import mel as librosa_mel_fn

# https://github.com/google-research/google-research/blob/68c738421186ce85339bfee16bf3ca2ea3ec16e4/ged_tts/distance_function/spectral_ops.py
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def log1p(x):
    return torch.log1p(x) if torch.is_tensor(x) else np.log1p(x)

def expm1(x):
    return torch.expm1(x) if torch.is_tensor(x) else np.expm1(x)

def mel_to_hertz(frequencies_mel):
    return _MEL_BREAK_FREQUENCY_HERTZ * expm1(frequencies_mel / _MEL_HIGH_FREQUENCY_Q)

def hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * log1p(frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)

def get_spectral_matrix(window_length, num_spec_bins=256, use_mel_scale=True,
                        sample_rate=16000, dtype=torch.float32, device=None):
    upper_edge_hertz = sample_rate / 2
    lower_edge_hertz = sample_rate / window_length
    
    if use_mel_scale:
        upper_edge_mel = hertz_to_mel(upper_edge_hertz)
        lower_edge_mel = hertz_to_mel(lower_edge_hertz)
        mel_frequencies = torch.linspace(lower_edge_mel, upper_edge_mel, num_spec_bins,
                                         dtype=dtype, device=device)
        hertz_frequencies = mel_to_hertz(mel_frequencies)
    else:
        hertz_frequencies = torch.linspace(lower_edge_hertz, upper_edge_hertz, num_spec_bins,
                                           dtype=dtype, device=device)
    
    time_col_vec = 2 * np.pi / sample_rate * torch.arange(0, window_length, dtype=dtype, device=device)
    tmat = rearrange(hertz_frequencies, 'n -> 1 n') * rearrange(time_col_vec, 'm -> m 1')
    dct_mat = torch.cos(tmat)
    dst_mat = torch.sin(tmat)
    return torch.complex(dct_mat, -dst_mat)

def real_complex_matmul(real_mat, complex_mat):
    return torch.complex(real_mat @ complex_mat.real, real_mat @ complex_mat.imag)

# worse quality than the STFT version when window_length is large - prefer STFT version
class MelDFTSpectrogram(nn.Module):
    def __init__(self, window_length, hop_size, n_mels=64, sample_rate=16000, device=None):
        super().__init__()
        self.hop_size = hop_size
        mat = get_spectral_matrix(window_length, num_spec_bins=n_mels,
                                  use_mel_scale=True, sample_rate=sample_rate,
                                  dtype=torch.float64, device=device).to(torch.cfloat)
        self.register_buffer('mat', mat)
        
        window = torch.hann_window(window_length, device=device)
        kernel = rearrange(torch.diag(window), 'm n -> m 1 n')
        self.register_buffer('kernel', kernel)
    
    @autocast(enabled=False)
    def forward(self, x):
        windowed_frames = F.conv1d(rearrange(x, '... n -> ... 1 n'), self.kernel, stride=self.hop_size)
        mel_dft = real_complex_matmul(windowed_frames.transpose(-1, -2), self.mat)
        return torch.sqrt(mel_dft.real ** 2 + mel_dft.imag ** 2).transpose(-1, -2)

class MelSTFTSpectrogram(nn.Module):
    def __init__(self, window_length, hop_size, n_mels=64, sample_rate=16000):
        super().__init__()
        self.n_fft = max(4 * n_mels, window_length)
        self.hop_size = hop_size
        mel = librosa_mel_fn(n_fft=self.n_fft,
                             n_mels=n_mels,
                             fmin=sample_rate / window_length,
                             fmax=sample_rate / 2,
                             sr=sample_rate)
        self.register_buffer('mel_basis', torch.from_numpy(mel).float())
        self.register_buffer('window', torch.hann_window(window_length))
        self.pad = nn.ReflectionPad1d(int((self.n_fft - self.hop_size) / 2))
    
    @autocast(enabled=False)
    def forward(self, x):
        shape = x.shape
        x = torch.atleast_2d(x)
        stft = torch.stft(self.pad(x), self.n_fft, self.hop_size,
                          self.window.size(0), self.window,
                          pad_mode='reflect', normalized=False,
                          onesided=True, return_complex=True)
        spec = torch.sqrt(torch.clamp(stft.real ** 2 + stft.imag ** 2, min=1e-7))
        mel_spec = self.mel_basis @ spec
        return mel_spec.reshape(*shape[:-1], *mel_spec.shape[-2:])

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(5 * 16000)
    S = MelSTFTSpectrogram(1024, 256, 256)
    print(S(x))