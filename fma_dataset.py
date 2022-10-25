import torch
import torchaudio
from torch.utils.data import Dataset

import numpy as np
import glob

# TODO: if necessary, use webdataset later - interface will be the same
# import webdataset as wds

import os.path as osp
import pandas as pd

FMA_ROOT = 'fma'

def to_track_id(path):
    fname = osp.basename(path)
    trackid, _ = osp.splitext(fname)
    return int(trackid)

def get_audio_path(audio_dir, track_id, ext='mp3'):
    tid_str = '{:06d}'.format(track_id)
    return osp.join(audio_dir, tid_str[:3], tid_str + '.' + ext)

def to_filename(path):
    pardir = osp.basename(osp.dirname(path))
    fname = osp.basename(path)
    return osp.join(pardir, fname)

def to_title(meta):
    return '{} - {} ({})'.format(meta['artist', 'name'],
                                 meta['track', 'title'],
                                 meta['track', 'genre_top'])

def map_split(split):
    return {'train'     : 'training',
            'test'      : 'test',
            'validation': 'validation'}[split]

subsets = ['small', 'medium', 'large']

def get_subsets(size):
    return subsets[:subsets.index(size) + 1]

class FMA(Dataset):
    sample_rate = 16000
    
    def __init__(self, num_frames=-1, fma_size='medium', split='train'):
        tracks = pd.read_csv(osp.join(FMA_ROOT, 'tracks.csv'), sep=',', index_col=0, header=[0, 1])
        paths = glob.glob(osp.join(FMA_ROOT, 'fma_{}_clean'.format(fma_size), '**', '*.wav'))
        fnames = [to_filename(p) for p in paths]
        
        trackids = np.array([to_track_id(p) for p in fnames], dtype=np.int64)
        fnames = np.array(fnames, dtype=str)
        sorted_idx = np.argsort(trackids)
        
        mask = tracks['set', 'subset'].isin(get_subsets(fma_size))
        mask = mask & tracks.index.isin(trackids)
        
        tracks = tracks[mask]
        tracks['audio_filename'] = fnames[sorted_idx]
        tracks = tracks[tracks['set', 'split'] == map_split(split)]
        
        self.metadata = tracks
        self.num_frames = num_frames
        self.fma_size = fma_size
        self.split = split
    
    def __getitem__(self, item):
        meta = self.get_metadata(item)
        fname = meta.audio_filename.iloc[0]
        path = osp.join(FMA_ROOT, 'fma_{}_clean'.format(self.fma_size), fname)
        if self.num_frames <= 0:
            waveform, _ = torchaudio.load(path)
            offset = 0
        else:
            info = torchaudio.info(path)
            offset = torch.randint(info.num_frames - self.num_frames, ())
            waveform, _ = torchaudio.load(path, frame_offset=offset, num_frames=self.num_frames)
        return waveform[0], item, offset, *self.get_labels(item)
    
    def __len__(self):
        return len(self.metadata)
    
    def get_labels(self, item):
        meta = self.get_metadata(item)
        return meta.audio_filename.iloc[0], to_title(meta)
    
    def get_metadata(self, item):
        x = self.metadata.iloc[item]
        return x

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from tqdm import tqdm

    def collate_fn(batch):
        waveforms, ids, offsets, fnames, titles = zip(*batch)
        return nn.utils.rnn.pad_sequence(waveforms, batch_first=True), ids, offsets, fnames, titles

    for split in ['train', 'test', 'validation']:
        dataset = FMA(FMA.sample_rate * 15, split=split)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16, collate_fn=collate_fn)
        
        print()
