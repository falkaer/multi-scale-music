from functools import partial
import torch
from torch import nn
from vq.vector_quantize_pytorch import VectorQuantize

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    
    def __init__(
            self,
            *,
            num_quantizers,
            shared_codebook=False,
            **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])
        
        if not shared_codebook:
            return
        
        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook
        
        for vq in rest_vq:
            vq._codebook = codebook
    
    @property
    def codebooks(self):
        return map(lambda x: x._codebook, self.layers)
    
    def embed_codes(self, codes, dim=-1, return_partials=False):
        quantized_out = 0.
        partials = []
        for code, vq in zip(codes.unbind(dim), self.layers):
            embed = vq.embed_codes(code)
            quantized_out = quantized_out + embed
            if return_partials:
                partials.append(embed)
        
        if return_partials:
            return quantized_out, torch.stack(partials, dim=dim - 1)
        else:
            return quantized_out
    
    def forward(self, x, return_distances=False):
        quantized_out = 0.
        residual = x
        
        all_losses = []
        all_num_expired = []
        all_indices = []
        
        if return_distances:
            all_dist_sq = []
        
        for layer in self.layers:
            if return_distances:
                quantized, indices, num_expired, loss, dist_sq = layer(residual, return_distances=True)
            else:
                quantized, indices, num_expired, loss = layer(residual)
            
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_num_expired.append(num_expired)
            all_losses.append(loss)
            
            if return_distances:
                all_dist_sq.append(dist_sq)
        
        if return_distances:
            all_indices, all_num_expired, all_losses = map(partial(torch.stack, dim=-1),
                                                                        (all_indices,
                                                                         all_num_expired,
                                                                         all_losses))
            all_dist_sq = torch.stack(all_dist_sq, dim=-2)
            return quantized_out, all_indices, all_num_expired, all_losses, all_dist_sq
        else:
            all_indices, all_num_expired, all_losses = map(partial(torch.stack, dim=-1),
                                                           (all_indices,
                                                            all_num_expired,
                                                            all_losses))
            return quantized_out, all_indices, all_num_expired, all_losses

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(1, 1500, 256, device='cuda')
    vq = ResidualVQ(dim=256, codebook_size=1024, num_quantizers=8).cuda()
    quantized, codes, num_expired, losses, dist_sq = vq(x, return_distances=True)
    
    quantized, partials = vq.embed_codes(codes, return_partials=True)
    
    print()
