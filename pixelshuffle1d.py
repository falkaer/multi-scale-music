import torch
from einops.layers.torch import Rearrange

def PixelShuffle1D(upscale_factor):
    return Rearrange('... (u c) n -> ... c (n u)', u=upscale_factor)

def PixelUnshuffle1D(downscale_factor):
    return Rearrange('... c (n u) -> ... (u c) n', u=downscale_factor)

if __name__ == '__main__':
    torch.manual_seed(0)
    X = torch.randn(1, 5, 10)
    U = PixelShuffle1D(5)
    
    print(U(X).shape)