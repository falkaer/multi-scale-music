import torch
import torch.nn as nn
import torch.nn.functional as F

from librosa.filters import mel as librosa_mel_fn

def log(x, eps=1e-5):
    return torch.log(torch.clamp(x, min=eps))

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft=1024, num_mels=80,
                    hop_size=256, win_size=1024,
                    fmin=0, fmax=8000,
                    sampling_rate=16000, center=False):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)
    
    stft = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-7)
    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    return log(spec)

def hinge_adv_D_loss(D_x, D_G_x):
    return F.relu(1 - D_x).mean() + F.relu(1 + D_G_x).mean()

def hinge_adv_G_loss(D_G_x):
    return F.relu(1 - D_G_x).mean()

def ls_adv_D_loss(D_x, D_G_x):
    return torch.mean((D_x - 1) ** 2) + torch.mean(D_G_x ** 2)

def ls_adv_G_loss(D_G_x):
    return torch.mean((D_G_x - 1) ** 2)

def feature_loss(x_fmaps, G_x_fmaps):
    assert len(x_fmaps) == len(G_x_fmaps)
    losses = []
    for x_fmap, G_x_fmap in zip(x_fmaps, G_x_fmaps):
        losses.append(torch.abs(x_fmap - G_x_fmap).mean())
    return torch.stack(losses, dim=-1).mean()

def l1_mel_loss(x, G_x):
    return F.l1_loss(mel_spectrogram(G_x), mel_spectrogram(x))

# https://github.com/google-research/google-research/blob/68c738421186ce85339bfee16bf3ca2ea3ec16e4/ged_tts/distance_function/spectral_ops.py
def spectral_reconstruction_loss(x, G_x, spec, s):
    alpha_s = (s / 2) ** 0.5
    S_x = spec(x)
    S_G_x = spec(G_x)
    
    l1_dist = (S_x - S_G_x).abs().sum(dim=-2).mean()
    log_residuals = (log(S_x) - log(S_G_x)) ** 2
    l2_log_dist = torch.mean(torch.sum(log_residuals, dim=-2) ** 0.5)
    return l1_dist + alpha_s * l2_log_dist

class SpectralReconstructionLoss(nn.Module):
    def __init__(self, win_lengths, specs):
        super().__init__()
        self.win_lengths = win_lengths
        self.specs = specs
    
    def forward(self, x, G_x):
        return torch.stack([spectral_reconstruction_loss(x, G_x, spec, s) for s, spec in
                            zip(self.win_lengths, self.specs)])

if __name__ == '__main__':
    x = torch.randn(1, 16000)
    print(mel_spectrogram(x))
