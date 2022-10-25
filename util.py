import math
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist
from torch.nn.utils.parametrize import register_parametrization

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

import numpy as np
import os
import os.path as osp
from glob import glob

from collections import defaultdict
from itertools import chain

from weight_norm import WeightNorm

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def overridekwargs(kwargs, **defaults):
    kwargs = dict(kwargs)
    kwargs.update(defaults)
    return kwargs

def isscalar(x):
    return torch.is_tensor(x) and x.ndim == 0 or isinstance(x, Number)

def isfinite(x):
    return torch.isfinite(x).all() if torch.is_tensor(x) else math.isfinite(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def ddp_unwrap(model):
    return model.module if isinstance(model, DDP) else model

def to_state_dicts(m):
    if isinstance(m, dict):
        return {k: to_state_dicts(v) for k, v in m.items()}
    elif isinstance(m, list):
        return [to_state_dicts(v) for v in m]
    elif callable(getattr(m, 'state_dict', None)):
        return ddp_unwrap(m).state_dict()
    else:
        return m

def load_state_dicts(m, chkpt):
    if isinstance(m, dict):
        for k, v in m.items():
            if k in chkpt:
                load_state_dicts(v, chkpt[k])
            else:
                print('KEY {} NOT FOUND IN CHECKPOINT'.format(k))
    elif isinstance(m, list):
        for v, d in zip(m, chkpt):
            load_state_dicts(v, d)
    elif callable(getattr(m, 'load_state_dict', None)):
        ddp_unwrap(m).load_state_dict(chkpt)

def save_checkpoint(modules, checkpoint_dir, **kwargs):
    dirname = osp.join(checkpoint_dir, wandb.run.name)
    os.makedirs(dirname, exist_ok=True)
    path = osp.join(dirname, 'model_{}.pth'.format(wandb.run.step))
    modules = to_state_dicts(modules)
    modules.update(kwargs)
    torch.save(modules, path)

def get_latest_checkpoint(checkpoint_dir, run_name=None):
    run_name = default(run_name, wandb.run.name if wandb.run else None)
    dirname = osp.join(checkpoint_dir, run_name)
    paths = glob(osp.join(dirname, 'model_*.pth'))
    names = map(osp.basename, paths)
    max_steps = max(int(name[6:len(name) - 4]) for name in names)
    return get_checkpoint(checkpoint_dir, run_name, max_steps)

def get_checkpoint(checkpoint_dir, run_name, run_step):
    return osp.join(checkpoint_dir, run_name, 'model_{}.pth'.format(run_step))

def broadcast_object(obj, src=0):
    l = [obj]
    dist.broadcast_object_list(l, src=src)
    return l[0]

class MetricDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = defaultdict(lambda: 0)
    
    def __getitem__(self, key):
        return super(MetricDict, self).__getitem__(key) / self.counter[key]
    
    def __setitem__(self, key, value):
        if torch.is_tensor(value):
            value = value.detach()
        if key in self:
            value = value + super(MetricDict, self).__getitem__(key)
        super(MetricDict, self).__setitem__(key, value)
        self.counter[key] += 1
    
    def update(self, mapping, **kwargs):
        if hasattr(mapping, 'items'):
            mapping = mapping.items()
        for k, v in chain(mapping, kwargs.items()):
            self.__setitem__(k, v)

# helps with data loader prefetching
class OversamplingDataset(Dataset):
    def __init__(self, dataset, oversampling_factor):
        super().__init__()
        self.dataset = dataset
        self.oversampling_factor = oversampling_factor
    
    @property
    def split(self):
        return self.dataset.split
    
    @property
    def orig_dataset(self):
        return self.dataset.orig_dataset
    
    @property
    def sample_rate(self):
        return self.dataset.sample_rate
    
    def __getitem__(self, item):
        return self.dataset[item % len(self.dataset)]
    
    def __len__(self):
        return len(self.dataset) * self.oversampling_factor
    
    def get_metadata(self, item):
        return self.dataset.get_metadata(item % len(self.dataset))

def init_module(module, init_fun):
    if hasattr(module, 'weight'):
        init_fun(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        # fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
        # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # init.uniform_(module.bias, -bound, bound)
        init.zeros_(module.bias)
    for m in module.children():
        init_module(m, init_fun)
    return module

def recursive_weight_norm(module):
    # collect to avoid iterating over the parametrizations
    modules = list(module.modules())
    for m in modules:
        if hasattr(m, 'weight'):
            register_parametrization(m, 'weight', WeightNorm(eps=1e-8))
    return module

def fig_to_pil(fig):
    from PIL import Image
    fig.canvas.draw()
    hw = fig.canvas.get_width_height()
    rgb = fig.canvas.tostring_rgb()
    return Image.frombytes('RGB', hw, rgb)

def plot_spectrogram(x, sr=16000, ax=None, norm=None, colorbar=False):
    import matplotlib.pyplot as plt
    from stft import MelSTFTSpectrogram
    if ax is None:
        fig = plt.figure(figsize=(12, 4), dpi=100)
        ax = fig.gca()
    else:
        fig = None
    S = MelSTFTSpectrogram(256, 64, 128, sr)
    dbspec = 20 * torch.log10(S(x))
    im = ax.imshow(dbspec, origin='lower', aspect='auto', norm=norm)
    num_frames = dbspec.shape[-1]
    num_sec = x.shape[-1] // sr
    ax.set_xticks(np.linspace(0, num_frames, num_sec))
    ax.set_xticklabels(np.arange(0, num_sec))
    if colorbar:
        plt.colorbar(im, ax=ax)
    if fig is not None:
        return fig

def check_causal(f, X, generative=False):
    assert X.requires_grad
    y = f(X)
    t = 0 if generative else 1
    for i in range(X.size(-2)):
        G = torch.autograd.grad(y[..., i, 0], X, retain_graph=True)[0][0]
        assert torch.all(G[i + t:] == 0), \
            'failed causality check at position {}, expected grad(f, X) = 0 but got {}\n'.format(i, G[i + 1:])

def check_receptive_field(f, X):
    assert X.requires_grad
    y = f(X)
    G = torch.autograd.grad(y[-1, 0], X)[0]
    for i in range(X.size(-1)):
        if G[..., i] != 0:
            return X.size(-1) - i

def configure_optimizer(model, lr, weight_decay=0.01, betas=(0.9, 0.95)):
    """
    Following minGPT:
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.LSTM, torch.nn.GRU)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif ('weight' in pn or 'bias' in pn) and isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                no_decay.add(fpn)
    
    # special case the start token parameter as not decayed
    no_decay.add('time_start_token')
    
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer

if __name__ == '__main__':
    from seq.rq_transformer import RQTransformer
    from vq import ResidualVQ
    quantizer = ResidualVQ(dim=256, codebook_size=1024, num_quantizers=8).cuda()
    model = RQTransformer(quantizer=quantizer, inner_dim=512, time_layers=6, resolution_layers=4).cuda()
    
    opt = configure_optimizer(model, 1e-4)
