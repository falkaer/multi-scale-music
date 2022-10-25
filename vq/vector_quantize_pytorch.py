import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast

from einops import rearrange, repeat
from contextlib import contextmanager

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    if temperature == 0:
        return t.argmax(dim=dim)
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)

def ema_inplace_(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    
    return samples[indices]

def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()
    
    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)
    
    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    
    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []
    
    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    
    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    
    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)
    
    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()
    
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    return torch.cat(all_samples, dim=0)

def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False,
           sample_fn=sample_vectors, all_reduce_fn=noop):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_fn(samples, num_clusters)
    
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = -samples @ means.t()
        else:
            dists = (samples.pow(2).sum(1, keepdim=True)
                     - 2 * samples @ means.t()
                     + means.t().pow(2).sum(0, keepdim=True))
        
        buckets = torch.argmax(-dists, dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)
        
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        all_reduce_fn(new_means)
        
        if use_cosine_sim:
            new_means = l2norm(new_means)
        
        means = torch.where(zero_mask[..., None], means, new_means)
    
    return means, bins

# regularization losses

def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)
        
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        
        self.sample_fn = sample_vectors_distributed if use_ddp else sample_vectors
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
    
    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters,
                                     sample_fn=self.sample_fn, all_reduce_fn=self.all_reduce_fn)
        
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
    
    def replace_(self, samples, mask):
        self.embed.data[mask] = self.sample_fn(samples, mask.sum().item())
        self.cluster_size.data[mask] = self.threshold_ema_dead_code
    
    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return 0
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return 0
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace_(batch_samples, mask=expired_codes)
        return expired_codes.sum()
    
    @autocast(enabled=False)
    def embed_distances_squared(self, x):
        shape = x.shape
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.detach().t()
        dist_sq = (flatten.pow(2).sum(1, keepdim=True)
                   - 2 * flatten @ embed
                   + embed.pow(2).sum(0, keepdim=True))
        return dist_sq.reshape(*shape[:-1], dist_sq.shape[-1])

    @autocast(enabled=False)
    def embed_codes(self, codes):
        return F.embedding(codes, self.embed)
    
    @autocast(enabled=False)
    def forward(self, x):
        x = x.float()
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        
        self.init_embed_(flatten)
        dist_sq = self.embed_distances_squared(x)
        
        embed_ind = gumbel_sample(-dist_sq, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_onehot = rearrange(embed_onehot, '... D -> (...) D')
        quantize = self.embed_codes(embed_ind)
        num_expired = torch.tensor(0, device=x.device, dtype=torch.int64)
        
        if self.training:
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)
            
            ema_inplace_(self.cluster_size, cluster_size, self.decay)
            
            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)
            
            ema_inplace_(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            num_expired += self.expire_codes_(x)
        
        return quantize, embed_ind, num_expired, dist_sq

class CosineSimCodebook(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            kmeans_init=False,
            kmeans_iters=10,
            decay=0.8,
            eps=1e-5,
            threshold_ema_dead_code=2,
            use_ddp=False,
            learnable_codebook=False,
            sample_codebook_temp=0.
    ):
        super().__init__()
        self.decay = decay
        
        if not kmeans_init:
            embed = l2norm(torch.randn(codebook_size, dim))
        else:
            embed = torch.zeros(codebook_size, dim)
        
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp
        
        self.sample_fn = sample_vectors_distributed if use_ddp else sample_vectors
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
    
    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim=True,
                                     sample_fn=self.sample_fn, all_reduce_fn=self.all_reduce_fn)
        
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
    
    def replace_(self, samples, mask):
        samples = l2norm(samples)
        self.embed.data[mask] = self.sample_fn(samples, mask.sum().item())
        self.cluster_size.data[mask] = self.threshold_ema_dead_code
    
    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return 0
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return 0
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace_(batch_samples, mask=expired_codes)
        return expired_codes.sum()
    
    # squared distance will be in [0, 4]
    @autocast(enabled=False)
    def embed_distances_squared(self, x):
        shape = x.shape
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)
        embed = l2norm(self.embed.detach()).t()
        dist_sq = 2 - 2 * flatten @ embed
        return dist_sq.reshape(*shape[:-1], dist_sq.shape[-1])

    @autocast(enabled=False)
    def embed_codes(self, codes):
        return F.embedding(codes, self.embed)
    
    @autocast(enabled=False)
    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)
        self.init_embed_(flatten)
        
        dist_sq = self.embed_distances_squared(x)
        embed_ind = gumbel_sample(-dist_sq, dim=-1, temperature=self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_onehot = rearrange(embed_onehot, '... D -> (...) D')
        quantize = self.embed_codes(embed_ind)
        num_expired = torch.tensor(0, device=x.device, dtype=torch.int64)
        
        if self.training:
            bins = embed_onehot.sum(0)
            self.all_reduce_fn(bins)
            
            ema_inplace_(self.cluster_size, bins, self.decay)
            
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)
            
            embed_sum = flatten.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)
            
            embed = l2norm(self.embed.detach())
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], embed,
                                           embed_normalized)
            ema_inplace_(self.embed, embed_normalized, self.decay)
            num_expired += self.expire_codes_(x)
        
        return quantize, embed_ind, num_expired, dist_sq

# main class

class VectorQuantize(nn.Module):
    def __init__(
            self,
            dim,
            codebook_size,
            n_embed=None,
            codebook_dim=None,
            decay=0.8,
            eps=1e-5,
            kmeans_init=False,
            kmeans_iters=10,
            use_cosine_sim=False,
            threshold_ema_dead_code=0,
            channel_last=True,
            accept_image_fmap=False,
            commitment_weight=None,
            commitment=1.,  # deprecate in next version, turn off by default
            orthogonal_reg_weight=0.,
            orthogonal_reg_active_codes_only=False,
            orthogonal_reg_max_codes=None,
            sample_codebook_temp=0.,
            sync_codebook=False
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)
        
        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection \
            else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection \
            else nn.Identity()
        
        self.eps = eps
        self.commitment_weight = default(commitment_weight, commitment)
        
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        
        codebook_class = EuclideanCodebook if not use_cosine_sim \
            else CosineSimCodebook
        
        self._codebook = codebook_class(
                dim=codebook_dim,
                codebook_size=n_embed,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_ddp=sync_codebook,
                learnable_codebook=has_codebook_orthogonal_loss,
                sample_codebook_temp=sample_codebook_temp
        )
        
        self.codebook_size = codebook_size
        self.norm = l2norm if use_cosine_sim else nn.Identity()
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
    
    @property
    def codebook(self):
        return self._codebook.embed
    
    def embed_distances_squared(self, x):
        return self._codebook.embed_distances_squared(x)
    
    def embed_codes(self, codes):
        return self._codebook.embed_codes(codes)
    
    def forward(self, x, return_distances=False):
        shape, device, codebook_size = x.shape, x.device, self.codebook_size
        
        need_transpose = not self.channel_last and not self.accept_image_fmap
        
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')
        
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        
        x = self.project_in(x)
        
        x = self.norm(x)
        quantize, embed_ind, num_expired, dist_sq = self._codebook(x)
        quantize = self.norm(quantize)
        
        if self.training:
            quantize = x + (quantize - x).detach()
        
        loss = x.new_zeros(2)
        
        if self.commitment_weight > 0:
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss[0] = commit_loss * self.commitment_weight
        
        if self.orthogonal_reg_weight > 0:
            codebook = self.codebook
            
            if self.orthogonal_reg_active_codes_only:
                # only calculate orthogonal loss for the activated codes for this batch
                unique_code_ids = torch.unique(embed_ind)
                codebook = codebook[unique_code_ids]
            
            num_codes = codebook.shape[0]
            if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                codebook = codebook[rand_ids]
            
            orthogonal_reg_loss = orthgonal_loss_fn(codebook).unsqueeze(0)
            loss[1] = orthogonal_reg_loss * self.orthogonal_reg_weight
        
        quantize = self.project_out(quantize)
        
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        
        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) -> b h w', h=height, w=width)
        
        if return_distances:
            return quantize, embed_ind, num_expired, loss, dist_sq
        else:
            return quantize, embed_ind, num_expired, loss
