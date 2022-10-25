import argparse
from glob import glob
from util import default, get_checkpoint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
import os.path as osp

def checkpoint_to_embeddings(chkpt):
    _, idext = chkpt.split('_')
    return osp.join(osp.dirname(chkpt), 'embeddings_{}'.format(idext))

def get_latest_embeddings(checkpoint_dir, run_name=None):
    import wandb
    run_name = default(run_name, wandb.run.name if wandb.run else None)
    dirname = osp.join(checkpoint_dir, run_name)
    paths = glob(osp.join(dirname, 'embeddings_*.pth'))
    names = map(osp.basename, paths)
    max_steps = max(int(name[6:len(name) - 4]) for name in names)
    return get_embeddings(checkpoint_dir, run_name, max_steps)

def get_embeddings(checkpoint_dir, run_name, run_step):
    return osp.join(checkpoint_dir, run_name, 'embeddings_{}.pth'.format(run_step))

# use original dataset just for metadata
class EmbeddingDataset(Dataset):
    def __init__(self, orig_dataset, checkpoint_dir, run_name, run_step=None, num_embeddings=-1):
        if run_step is None:
            emb = get_latest_embeddings(checkpoint_dir, run_name)
        else:
            emb = get_embeddings(checkpoint_dir, run_name, run_step)
        self.embeddings = torch.load(emb, map_location='cpu')[orig_dataset.split]
        self.orig_dataset = orig_dataset
        self.num_embeddings = num_embeddings
    
    @property
    def split(self):
        return self.orig_dataset.split
    
    @property
    def sample_rate(self):
        return self.orig_dataset.sample_rate
    
    def __getitem__(self, item):
        emb = self.embeddings[item]
        if 0 < self.num_embeddings < emb.shape[0]:
            offset = torch.randint(emb.shape[0] - self.num_embeddings, ())
            emb = emb[offset:offset + self.num_embeddings]
        else:
            offset = 0
        return emb, item, offset, *self.orig_dataset.get_labels(item)
    
    def __len__(self):
        return len(self.orig_dataset)
    
    def get_metadata(self, item):
        return self.orig_dataset.get_metadata(item)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['maestro', 'fma'])
    parser.add_argument('run_name', type=str)
    parser.add_argument('run_step', type=int, default=None)
    parser.add_argument('--clip_length', type=int, default=60)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--use_amp', type=bool, default=True)
    
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('--dim', type=int, default=256, help='Dimensionality of embeddings')
    model_args.add_argument('--activation', type=str, default='snake',
                            choices=['elu', 'gelu', 'snake', 'snake_periodic',
                                     'snake_corrected', 'snake_periodic_corrected'])
    model_args.add_argument('--downsample', type=str, default='conv',
                            choices=['conv', 'pqmf', 'pqmf_lowpass', 'pixelshuffle'])
    model_args.add_argument('--C_enc', type=int, default=32, help='No. parameter scaling factor of encoder')
    model_args.add_argument('--weight_norm', type=bool, default=False)
    
    quantizer_args = parser.add_argument_group('quantizer_args')
    quantizer_args.add_argument('--num_quantizers', type=int, default=8)
    quantizer_args.add_argument('--codebook_size', type=int, default=1024)
    quantizer_args.add_argument('--codebook_dim', type=int, default=None)
    quantizer_args.add_argument('--shared_codebook', type=bool, default=False)
    quantizer_args.add_argument('--use_cosine_sim', type=bool, default=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    import torch.distributed as dist
    from einops import rearrange
    from tqdm import tqdm
    
    from util import get_latest_checkpoint, load_state_dicts, recursive_weight_norm
    from vq.model import Encoder, ConvDownsample, PQMFDownsample, PQMFLowPassDownsample, PixelShuffleDownsample
    from snake import Snake
    
    args = parse_args()
    
    use_ddp = 'LOCAL_RANK' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = int(os.environ['LOCAL_RANK']) if use_ddp else 0
    torch.cuda.set_device(local_rank)
    
    if use_ddp:
        dist.init_process_group('nccl')
    
    snake_alpha = 0.5
    activation = {'snake'                   : lambda c: Snake(c, init=snake_alpha),
                  'snake_periodic'          : lambda c: Snake(c, init='periodic'),
                  'snake_corrected'         : lambda c: Snake(c, init=snake_alpha, correction='std'),
                  'snake_periodic_corrected': lambda c: Snake(c, init='periodic', correction='std'),
                  'elu'                     : lambda _: nn.ELU(),
                  'gelu'                    : lambda _: nn.GELU()}[args.activation]
    
    downsample = {'pqmf'        : PQMFDownsample,
                  'pqmf_lowpass': PQMFLowPassDownsample,
                  'conv'        : ConvDownsample,
                  'pixelshuffle': PixelShuffleDownsample}[args.downsample]
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    encoder = Encoder(C=args.C_enc,
                      D=args.dim,
                      activation=activation,
                      downsample=downsample,
                      checkpoints=0).to(device)
    
    if args.weight_norm:
        recursive_weight_norm(encoder)
    
    if args.run_step is None:
        chkpt = get_latest_checkpoint(args.checkpoint_dir, args.run_name)
    else:
        chkpt = get_checkpoint(args.checkpoint_dir, args.run_name, args.run_step)
    d = torch.load(chkpt, map_location=torch.device(device))
    load_state_dicts(encoder, d['encoder'])
    encoder = encoder.eval()
    
    def get_dataset_split(split):
        if args.dataset == 'maestro':
            from maestro_dataset import MAESTRO
            
            dataset = MAESTRO(split=split)
        else:
            from fma_dataset import FMA
            dataset = FMA(fma_size='medium', split=split)
        return dataset
    
    def embed(x):
        x = rearrange(x, 'n -> 1 1 n')
        e, _ = encoder(x)
        return rearrange(e, '1 d n -> n d')
    
    embeddings_per_sec = 50
    
    def extract(dataset):
        all_embeddings = []
        sample_overlap = dataset.sample_rate  # must be wider than receptive field
        embed_overlap = (sample_overlap // dataset.sample_rate) * embeddings_per_sec
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, prefetch_factor=2,
                            pin_memory=torch.cuda.is_available())
        
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=args.use_amp):
            for x, *_ in tqdm(loader, total=len(loader)):
                x = x.to(device, non_blocking=True)[0]
                xs = torch.split(x, args.clip_length * dataset.sample_rate)
                xs_overlap = [xs[0]]
                for i in range(1, len(xs)):
                    x_overlap = torch.cat((xs[i - 1][-sample_overlap:], xs[i]))
                    xs_overlap.append(x_overlap)
                
                embeddings = []
                for x in xs_overlap:
                    embeddings.append(embed(x)[embed_overlap:])
                all_embeddings.append(torch.cat(embeddings).cpu())
        return all_embeddings
    
    all_embeddings = {s: extract(get_dataset_split(s)) for s in ['train', 'test', 'validation']}
    torch.save(all_embeddings, checkpoint_to_embeddings(chkpt))
