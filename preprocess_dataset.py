#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import glob

import torch

import torchaudio
import torchaudio.transforms as T

import pyloudnorm as pyln

import dask
from dask.diagnostics import ProgressBar

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_root')
    parser.add_argument('out_root')
    parser.add_argument('--in_type', type=str, default='wav')
    parser.add_argument('--out_type', type=str, default='wav')
    parser.add_argument('--in_rate', type=int, default=44100)
    parser.add_argument('--out_rate', type=int, default=16000)
    parser.add_argument('--out_encoding', type=str, default='PCM_S')
    parser.add_argument('--out_bits_per_sample', type=int, default=16)
    parser.add_argument('--loudnorm', default=False, action='store_true')
    parser.add_argument('--lufs', type=float, default=-20)
    parser.add_argument('--dry_run', default=False, action='store_true')
    return parser.parse_args()

class LoudnessNormalizer:
    def __init__(self, sample_rate, lufs):
        self.meter = pyln.Meter(sample_rate)
        self.lufs = lufs
    
    def __call__(self, x):
        x = x.squeeze(0).numpy()
        loudness = self.meter.integrated_loudness(x)
        x = torch.from_numpy(pyln.normalize.loudness(x, loudness, self.lufs))
        return x.unsqueeze(0)

def noop(x, *args, **kwargs):
    return x

def preprocess(inpath, outpath, bits_per_sample, encoding, resampler, normalizer):
    os.makedirs(osp.dirname(outpath), exist_ok=True)
    waveform, sample_rate = torchaudio.load(inpath, normalize=True, channels_first=True, format=args.in_type)
    if waveform.size(0) > 1:  # stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = resampler(waveform)
    waveform = normalizer(waveform)
    if not args.dry_run:
        torchaudio.save(outpath, waveform, resampler.new_freq,
                        bits_per_sample=bits_per_sample, encoding=encoding)

if __name__ == '__main__':
    args = parse_args()
    rootlen = len(args.audio_root) + 1
    
    torch.set_num_threads(1) # let dask handle parallelism
    resampler = T.Resample(orig_freq=args.in_rate,
                           new_freq=args.out_rate)
    
    normalizer = LoudnessNormalizer(args.out_rate, args.lufs) if args.loudnorm else noop
    
    tasks = []
    for inpath in glob.glob(osp.join(args.audio_root, '**/*.{}'.format(args.in_type))):
        outname, _ = osp.splitext(inpath[rootlen:])
        outpath = osp.join(args.out_root, outname + '.' + args.out_type)
        if osp.exists(outpath) and not args.dry_run:
            continue
        t = dask.delayed(preprocess)(inpath, outpath, args.out_bits_per_sample, args.out_encoding, resampler, normalizer)
        tasks.append(t)

    ProgressBar().register()
    dask.compute(*tasks)
