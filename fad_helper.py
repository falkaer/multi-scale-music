import os
import os.path as osp
import shutil
import fcntl
import subprocess

import torch

SETUP_COMMAND = 'source venv/fad/bin/activate'
GENERATE_COMMAND = '{}; python -m frechet_audio_distance.create_embeddings_main --input_files {} --stats {} --batch_size {}'
COMPUTE_COMMAND = '{}; python -m frechet_audio_distance.compute_fad --background_stats {} --test_stats {}'

import torchaudio
from torch.utils.data import DataLoader

class FADHelper:
    def __init__(self, dir):
        self.paths = []
        self.dir = dir
    
    def add_files(self, audio_batch, filenames, sr=16000, bits_per_sample=16, encoding='PCM_S'):
        audio_batch = audio_batch.detach().cpu()
        paths = []
        for audio, fname in zip(audio_batch.unbind(dim=0), filenames):
            path = osp.abspath(osp.join(self.dir, 'audio', fname))
            os.makedirs(osp.dirname(path), exist_ok=True)
            base_path, ext = osp.splitext(path)
            i = 0
            while osp.exists(path):  # append _0, _1, until it does not exist
                path = '{}_{}{}'.format(base_path, i, ext)
                i += 1
            
            torchaudio.save(path, audio.unsqueeze(0), sample_rate=sr,
                            bits_per_sample=bits_per_sample, encoding=encoding)
            
            paths.append(path)
        self.paths.extend(paths)
    
    def flush_file_list(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(osp.join(self.dir, 'files.csv'), 'a') as f:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX)
                f.write('\n'.join(self.paths) + '\n')
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)
    
    def generate_stats(self, batch_size):
        input_path = osp.abspath(osp.join(self.dir, 'files.csv'))
        stats_path = osp.abspath(osp.join(self.dir, 'stats'))
        command = GENERATE_COMMAND.format(SETUP_COMMAND, input_path, stats_path, batch_size)
        subprocess.run(command, stderr=subprocess.STDOUT, shell=True)
    
    def compute_fad(self, background_stats_path):
        background_stats_path = osp.abspath(background_stats_path)
        test_stats_path = osp.abspath(osp.join(self.dir, 'stats'))
        command = COMPUTE_COMMAND.format(SETUP_COMMAND, background_stats_path, test_stats_path)
        ret = subprocess.run(command, capture_output=True, shell=True)
        print(ret.stdout)
        try:
            return float(ret.stdout.decode('utf-8')[5:]) # FAD: ...
        except ValueError:
            return float('nan')
    
    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

# extract stats for background data
if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['maestro', 'fma'], help='Which dataset to extract stats for')
    parser.add_argument('split', type=str, help='Which split to extract stats for')
    parser.add_argument('--clip_length', type=int, default=5, help='Length of clips to extract (in seconds)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for audioset model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomly sampling audio')
    parser.add_argument('--min_samples', type=int, default=1000, help='Smallest number of samples to extract, '
                                                                      'will supersample if greater than dataset size')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.dataset == 'maestro':
        from maestro_dataset import MAESTRO
        
        dataset = MAESTRO(args.clip_length * MAESTRO.sample_rate, split=args.split)
    else:
        
        from fma_dataset import FMA
        dataset = FMA(args.clip_length * FMA.sample_rate, fma_size='medium', split=args.split)
    
    helper = FADHelper(osp.join('fad', 'background', args.dataset, dataset.split, str(args.seed)))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8, prefetch_factor=2)
    
    while len(helper.paths) < args.min_samples:
        for audio_batch, ids, offsets, fnames, titles in tqdm(loader, total=len(loader)):
            remainder = args.min_samples - (len(helper.paths) + len(fnames))
            if remainder < 0:
                fnames = fnames[:remainder]
                audio_batch = audio_batch[:remainder]
            helper.add_files(audio_batch, fnames, sr=dataset.sample_rate)
    
    helper.flush_file_list()
    helper.generate_stats(args.batch_size)
