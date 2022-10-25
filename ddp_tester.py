import torch
import torch.nn.functional as F
import torch.distributed as dist

import os

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    
    return samples[indices]

def pad_tensor(x, new_size, dim):
    padding_shape = list(x.shape)
    padding_shape[dim] = new_size - x.shape[dim]
    padding = x.new_empty(padding_shape)
    return torch.cat((x, padding), dim=dim)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

# def all_gather_variably_sized(x, sizes, dim=0):
#     max_size = max(sizes)
#     max_shape = list(x.shape)
#     max_shape[dim] = max_size
#     all_x = [x.new_empty(max_shape) for _ in range(dist.get_world_size())]
#     if x.size(dim) != max_size:
#         x = pad_tensor(x, max_size, dim=dim)
#     dist.all_gather(all_x, x)
#     return [x[:size] for x, size in zip(all_x, sizes)]

def pad_shape(shape, size, dim=0):
    return (size if i == dim else s for i, s in enumerate(shape))

def all_gather_variably_sized(x, sizes, dim=0):
    rank = dist.get_rank()
    all_x = []
    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(*pad_shape(x.shape, size, dim))
        dist.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    dist.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    # collect the number of samples per replica
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    
    # agree on how many samples should be drawn from each replica
    if dist.get_rank() == 0:
        mult = torch.distributions.Multinomial(num, probs=all_num_samples)
        samples_per_rank = mult.sample().long()
    else:
        samples_per_rank = torch.empty_like(all_num_samples)
    dist.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()
    
    # draw samples and all_gather
    local_samples = sample_vectors(local_samples, samples_per_rank[dist.get_rank()])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    return torch.cat(all_samples, dim=0)

def check_all_same(x):
    all_x = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(all_x, x)
    for y in all_x:
        assert torch.allclose(x, y), 'Difference: {}\n'.format(x - y)

if __name__ == '__main__':
    use_ddp = 'LOCAL_RANK' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = int(os.environ['LOCAL_RANK']) if use_ddp else 0
    torch.cuda.set_device(local_rank)
    
    if use_ddp:
        dist.init_process_group('nccl')
    
    torch.manual_seed(local_rank)
    B = torch.randint(4000, 8000, ()).item()
    X = torch.randn(B, 256, device='cuda')
    
    if local_rank == 0:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('profile', 'all_gather')
        ) as prof:
            for i in range(5):
                sample = sample_vectors_distributed(X, 10000)
                prof.step()
    else:
        for i in range(5):
            sample = sample_vectors_distributed(X, 10000)
    
    print(sample)
    print('Done')
