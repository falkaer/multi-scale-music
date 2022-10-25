import torch

import torchaudio
from torch.utils.data import Dataset

import glob

# TODO: if necessary, use webdataset later - interface will be the same
# import webdataset as wds

import os.path as osp
import pandas as pd

MAESTRO_ROOT = 'maestro-v3.0.0'
MAESTRO_CLEAN_ROOT = 'maestro-v3.0.0-clean'

def to_filename(path):
    pardir = osp.basename(osp.dirname(path))
    fname = osp.basename(path)
    return osp.join(pardir, fname)

def to_title(meta):
    return '{} - {} ({})'.format(meta['canonical_composer'],
                                 meta['canonical_title'],
                                 meta['year'])

class MAESTRO(Dataset):
    sample_rate = 16000
    
    def __init__(self, num_frames=-1, split='train'):
        tracks = pd.read_csv(osp.join(MAESTRO_ROOT, 'maestro-v3.0.0.csv'), sep=',')
        fnames = [to_filename(p) for p in glob.glob(osp.join(MAESTRO_CLEAN_ROOT, '**', '*.wav'))]
        self.metadata = tracks[tracks['audio_filename'].isin(fnames) & (tracks['split'] == split)]
        self.num_frames = num_frames
        self.split = split
    
    def __getitem__(self, item):
        meta = self.get_metadata(item)
        fname = meta.audio_filename
        path = osp.join(MAESTRO_CLEAN_ROOT, fname)
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
        return meta.audio_filename, to_title(meta)
    
    def get_metadata(self, item):
        x = self.metadata.iloc[item]
        return x

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    dataset = MAESTRO(MAESTRO.sample_rate * 5, split='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=8)
    
    for batch in tqdm(loader, total=len(loader)):
        print(batch)
