import argparse
from functools import partial

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as opt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler

from einops import rearrange

from tqdm import tqdm
import os
import os.path as osp

from contextlib import nullcontext

from embedding_dataset import EmbeddingDataset
from fad_helper import FADHelper
from fma_dataset import FMA
from seq.flash_attention import flash_attention
from seq.attention import simple_attention
from seq.losses import cross_entropy, weighted_cross_entropy
from seq.rq_transformer import Scale, ScaleNorm, RQTransformer, small_init_, tfixup_init_
from seq.sampling import sample_logits, scheduled_sample, softmax_
from seq.conformer import RQConformer, RQDilatedConformer
from snake import Snake

from vq import ResidualVQ

from maestro_dataset import MAESTRO
from util import MetricDict, OversamplingDataset, broadcast_object, configure_optimizer, count_parameters, \
    fig_to_pil, get_checkpoint, get_latest_checkpoint, isfinite, isscalar, load_state_dicts, plot_spectrogram, \
    recursive_weight_norm, save_checkpoint
from vq.model import ConvUpsample, Decoder, PQMFUpsample, PixelShuffleUpsample
from warmup import WarmupCosineAnnealingScheduler

EPS = 1e-10
EMBED_PER_SEC = 50
SAMPLES_PER_EMBED = 16000 // EMBED_PER_SEC
torch.backends.cudnn.benchmark = True

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('quant_run_name', type=str,
                        help='The run name of the quantization model to use for training data')
    parser.add_argument('quant_run_step', type=str,
                        help='The training step of the quantization model to fetch checkpoints and embeddings for')
    parser.add_argument('--checkpoint_dir', type=str, default='seq_checkpoints')
    parser.add_argument('--quant_checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_logs', default=False, action='store_true')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Accumulate training statistics and log every n batches')
    parser.add_argument('--resume', type=str, default=None, help='Resume run with given ID from latest checkpoint')
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='maestro', choices=['maestro', 'fma'])
    parser.add_argument('--dataset_oversampling', type=int, default=8, help='How much to oversample the dataset by')
    parser.add_argument('--seq_len', type=int, default=1024, help='Length of sequence (audio embeddings from encoder)')
    parser.add_argument('--grad_log_freq', type=int, default=200, help='Steps between logging gradient histograms')
    
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('--dim', type=int, default=256, help='Dimensionality of embeddings')
    model_args.add_argument('--inner_dim', type=int, default=256, help='Inner dimension in transformers')
    model_args.add_argument('--transformer', type=str, default='regular',
                            choices=['regular', 'conformer', 'dilated_conformer'])
    
    model_args.add_argument('--time_layers', type=int, default=12, help='Number of layers in time-wise transformer')
    model_args.add_argument('--resolution_layers', type=int, default=8,
                            help='Number of layers in resolution-wise transformer')
    model_args.add_argument('--heads', type=int, default=8, help='Number of heads per transformer')
    model_args.add_argument('--dim_head', type=int, default=64, help='Dimensions per transformer head')
    model_args.add_argument('--embed_dropout', type=float, default=0, help='Embedding dropout percent')
    
    model_args.add_argument('--ff_dropout', type=float, default=0, help='Feed-forward dropout percent')
    model_args.add_argument('--ff_mult', type=int, default=4,
                            help='Multiplier for internal dimension of feed-forward layers in transformer')
    model_args.add_argument('--ff_prenorm', type=str, default='layernorm',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'])
    model_args.add_argument('--ff_postnorm', type=str, default='none',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'])
    
    model_args.add_argument('--attention', type=str, default='flash',
                            choices=['flash', 'simple'], help='Which attention type to use')
    model_args.add_argument('--attn_prenorm', type=str, default='layernorm',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'], help='Attention prenorm')
    model_args.add_argument('--attn_postnorm', type=str, default='none',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'], help='Attention postnorm')
    model_args.add_argument('--attn_dropout', type=float, default=0, help='Attention dropout percent')
    model_args.add_argument('--postattn_dropout', type=float, default=0, help='Post attention dropout percent')
    
    model_args.add_argument('--transformer_prenorm', type=str, default='none',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'], help='Attention prenorm')
    model_args.add_argument('--transformer_midnorm', type=str, default='none',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'], help='Attention postnorm')
    model_args.add_argument('--transformer_postnorm', type=str, default='none',
                            choices=['none', 'scale', 'layernorm', 'scalenorm'], help='Attention postnorm')
    
    model_args.add_argument('--activation', type=str, default='gelu',
                            choices=['elu', 'gelu', 'snake', 'snake_periodic',
                                     'snake_corrected', 'snake_periodic_corrected'])
    
    model_args.add_argument('--use_rotary', action='store_true', help='Use rotary positional embeddings')
    model_args.add_argument('--use_alibi', action='store_true', help='Use ALiBi positional embeddings')
    
    model_args.add_argument('--init', type=str, default='small', choices=['small', 'tfixup'])
    
    quantizer_args = parser.add_argument_group('quantizer_args')
    quantizer_args.add_argument('--num_quantizers', type=int, default=8)
    quantizer_args.add_argument('--codebook_size', type=int, default=1024)
    quantizer_args.add_argument('--codebook_dim', type=int, default=None)
    quantizer_args.add_argument('--shared_codebook', action='store_true', default=False)
    quantizer_args.add_argument('--use_cosine_sim', action='store_true', default=False)
    quantizer_args.add_argument('--codebook_temp', type=float, default=0)
    
    decoder_args = parser.add_argument_group('decoder_args')
    decoder_args.add_argument('--dec_activation', type=str, default='snake',
                              choices=['elu', 'gelu', 'snake', 'snake_periodic',
                                       'snake_corrected', 'snake_periodic_corrected'])
    decoder_args.add_argument('--C_dec', type=int, default=32, help='No. parameter scaling factor of decoder')
    decoder_args.add_argument('--upsample', type=str, default='conv', choices=['conv', 'pqmf', 'pixelshuffle'])
    decoder_args.add_argument('--weight_norm', action='store_true')
    
    sampling_args = parser.add_argument_group('sampling_args')
    sampling_args.add_argument('--filter_top_k', type=float, default=None,
                               help='Discard bottom % of logits by magnitude when sampling')
    sampling_args.add_argument('--filter_top_p', type=float, default=0.92,
                               help='Retain only top % of logits by cumulative probability when sampling')
    sampling_args.add_argument('--temperature', type=float, default=1., help='Sampling temperature')
    sampling_args.add_argument('--scheduled_sampling', type=float, default=0.5, help='Scheduled sampling probability')
    
    optim_args = parser.add_argument_group('optim_args')
    optim_args.add_argument('--cross_entropy', type=str, default='hard', choices=['hard', 'soft'])
    optim_args.add_argument('--loss_decay', type=float, default=0)
    optim_args.add_argument('--batch_size', type=int, default=8)
    optim_args.add_argument('--model_lr', type=float, default=1e-4)
    optim_args.add_argument('--beta1', type=float, default=0.9)
    optim_args.add_argument('--beta2', type=float, default=0.95)
    optim_args.add_argument('--weight_decay', type=float, default=1e-4)
    optim_args.add_argument('--warmup_steps', type=int, default=1000)
    optim_args.add_argument('--max_grad_norm', type=float, default=1.)
    optim_args.add_argument('--use_amp', default=False, action='store_true')
    return parser

def code_to_wave(codes):
    quantized = quantizer.embed_codes(codes, dim=-1)
    quantized = rearrange(quantized, 'b n d -> b d n')
    out, _ = decoder(quantized)
    out = rearrange(out, 'b 1 n -> b n')
    return torch.tanh(out.float())

def validate(dataset):
    model.eval()
    decoder.cuda()
    
    val_metrics = MetricDict()
    
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                        collate_fn=collate_fn, num_workers=8, prefetch_factor=4, pin_memory=True)
    
    if logs:
        pred_fad_dir = osp.join('fad', wandb.run.name, 'preds', args.dataset, dataset.split)
        sample_fad_dir = osp.join('fad', wandb.run.name, 'samples', args.dataset, dataset.split)
    else:
        pred_fad_dir = None
        sample_fad_dir = None
    
    if use_ddp:
        sampler.set_epoch(epoch)
        pred_fad_dir = broadcast_object(pred_fad_dir, src=0)
        sample_fad_dir = broadcast_object(sample_fad_dir, src=0)
    
    pred_fad_helper = FADHelper(pred_fad_dir)
    sample_fad_helper = FADHelper(sample_fad_dir)
    pred_fad_helper.cleanup()
    sample_fad_helper.cleanup()
    
    # make sure no one deletes the fad workdir after this point
    if use_ddp:
        dist.barrier()
    
    for e, ids, offsets, fnames, titles in tqdm(loader, total=len(loader), disable=local_rank != 0):
        with amp_context():
            e = e.cuda(non_blocking=True)  # b n d
            if args.cross_entropy == 'hard' or args.codebook_temp == 0:
                _, codes, *_ = quantizer(e, return_distances=False)
                target = codes
            else:
                _, codes, *_, target = quantizer(e, return_distances=True)
                target = softmax_(target.mul_(-1 / args.codebook_temp))
            
            q_logits = model(codes)
            sampled_codes = model.generate(sample_len=5 * EMBED_PER_SEC, prime=codes, temperature=args.temperature,
                                           filter_top_k=args.filter_top_k, filter_top_p=args.filter_top_p)
            q_codes = sample_logits(q_logits, args.temperature, args.filter_top_k, args.filter_top_p)
            G_x, sampled_G_x = map(code_to_wave, (q_codes, sampled_codes))
            
            if args.loss_decay > 0:
                weights = 1 / torch.arange(1, args.num_quantizers + 1, device='cuda') ** args.loss_decay
                weights = weights + (1 - weights) * epoch / args.epochs
                loss = weighted_cross_entropy(q_logits, target, weights)
            else:
                loss = cross_entropy(q_logits, target)
            val_metrics.update({'{}/loss': loss})
        
        pred_fad_helper.add_files(G_x, fnames, sr=dataset.sample_rate)
        sample_fad_helper.add_files(sampled_G_x, fnames, sr=dataset.sample_rate)
    
    val_metrics = dict(val_metrics)
    if use_ddp:
        for x in val_metrics.values():
            dist.all_reduce(x)
            x /= dist.get_world_size()
    
    pred_fad_helper.flush_file_list()
    sample_fad_helper.flush_file_list()
    
    if logs:
        background_dir = osp.join('fad', 'background', args.dataset, dataset.split, '0', 'stats')
        pred_fad_helper.generate_stats(batch_size=64)
        sample_fad_helper.generate_stats(batch_size=64)
        val_metrics['{}/pred_fad_score'] = pred_fad_helper.compute_fad(background_dir)
        val_metrics['{}/sample_fad_score'] = sample_fad_helper.compute_fad(background_dir)
        
        log_dict = {k: v.cpu() if torch.is_tensor(v) else v for k, v in val_metrics.items()}
        log_dict = {k: v for k, v in log_dict.items() if isfinite(v)}
        log_dict = {k: v if isscalar(v) else wandb.Histogram(v)
                    for k, v in log_dict.items()}
        
        # log the last 3 samples in batch
        e, codes, G_x, sampled_G_x, ids, offsets, fnames, titles = map(lambda x: x[:3],
                                                                       (e, codes,
                                                                        G_x, sampled_G_x,
                                                                        ids, offsets,
                                                                        fnames, titles))
        x = code_to_wave(codes)
        x, G_x, sampled_G_x = map(torch.Tensor.cpu, (x, G_x, sampled_G_x))
        x, G_x, sampled_G_x = map(partial(torch.unbind, dim=0), (x, G_x, sampled_G_x))
        
        toaudio = lambda x, t=None: wandb.Audio(x, sample_rate=dataset.sample_rate, caption=t)
        log_dict['{}/real_audio'] = list(map(toaudio, x, titles))
        log_dict['{}/recon_audio'] = list(map(toaudio, G_x, titles))
        log_dict['{}/sampled_audio'] = list(map(toaudio, sampled_G_x))
        
        toimg = lambda x, t=None: wandb.Image(fig_to_pil(plot_spectrogram(x, sr=dataset.sample_rate)), caption=t)
        log_dict['{}/real_specs'] = list(map(toimg, x, titles))
        log_dict['{}/recon_specs'] = list(map(toimg, G_x, titles))
        log_dict['{}/sampled_specs'] = list(map(toimg, sampled_G_x))
        
        log_dict = {k.format(dataset.split): v for k, v in log_dict.items()}
        
        global global_step
        wandb.log(log_dict, step=global_step)
    
    decoder.cpu()

def step(emb):
    with amp_context():
        emb = emb.cuda(non_blocking=True)  # b n d
        if args.cross_entropy == 'hard' or args.codebook_temp == 0:
            _, codes, *_ = quantizer(emb, return_distances=False)
            target = codes
        else:
            _, codes, *_, target = quantizer(emb, return_distances=True)
            target = softmax_(target.mul_(-1 / args.codebook_temp))
        
        if args.scheduled_sampling > 0:
            sched_proba = args.scheduled_sampling / args.epochs * epoch
            if args.cross_entropy == 'hard' or args.codebook_temp == 0:
                target = target.clone()
            codes = scheduled_sample(model, codes,
                                     sample_proba=sched_proba,
                                     temperature=args.temperature,
                                     filter_top_k=args.filter_top_k,
                                     filter_top_p=args.filter_top_p,
                                     inplace=True)
        else:
            sched_proba = 0
        
        q_logits = ddp_model(codes)
        
        if args.loss_decay > 0:
            weights = 1 / torch.arange(1, args.num_quantizers + 1, device='cuda') ** args.loss_decay
            weights = weights + (1 - weights) * epoch / args.epochs
            loss = weighted_cross_entropy(q_logits, target, weights)
        else:
            weights = torch.ones(args.num_quantizers, device='cuda')
            loss = cross_entropy(q_logits, target)
    
    optim.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if args.max_grad_norm is not None:
        scaler.unscale_(optim)
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm,
                                        norm_type=2., error_if_nonfinite=False)
    else:
        norm = torch.tensor(0.)
    scaler.step(optim)
    scaler.update()
    
    assert scaler.get_scale() > 0
    
    if logs:
        global global_step, metrics
        
        metrics.update({
            'train/loss'       : loss,
            'train/lr'         : scheduler.get_last_lr()[0],
            'train/grad_scaler': scaler.get_scale(),
            'train/grad_norm'  : norm,
            'train/sched_proba': sched_proba,
            'train/weights'    : weights
        })
        
        if (global_step + 1) % args.log_every == 0:
            metrics = {k: v.cpu() if torch.is_tensor(v) else v for k, v in metrics.items()}
            metrics = {k: v for k, v in metrics.items() if isfinite(v)}
            metrics = {k: v if isscalar(v) else wandb.Histogram(v) for k, v in metrics.items()}
            wandb.log(metrics, step=global_step)
            metrics = MetricDict()
        
        global_step += effective_batch_size
    
    scheduler.step()

def train():
    ddp_model.train()
    if use_ddp:
        train_sampler.set_epoch(epoch)
    
    for emb, ids, offsets, fnames, titles in tqdm(train_loader, total=len(train_loader), disable=local_rank != 0):
        step(emb)
    
    if logs and (epoch + 1) % args.validate_every == 0:
        # log the last 3 samples in batch
        emb, ids, offsets, fnames, titles = map(lambda x: x[:3], (emb, ids, offsets, fnames, titles))
        with amp_context(), torch.inference_mode():
            codes = quantizer(emb.cuda(non_blocking=True))[1]
            q_logits = model(codes)
            q_codes = sample_logits(q_logits, args.temperature, args.filter_top_k, args.filter_top_p)
            decoder.cuda()
            G_x = code_to_wave(q_codes).cpu().unbind(dim=0)
            x = code_to_wave(codes).cpu().unbind(dim=0)
            decoder.cpu()
        
        toaudio = lambda x, t=None: wandb.Audio(x, sample_rate=train_dataset.sample_rate, caption=t)
        toimg = lambda x, t=None: wandb.Image(fig_to_pil(plot_spectrogram(x, sr=train_dataset.sample_rate)), caption=t)
        log_dict = {'train/real_audio' : list(map(toaudio, x, titles)),
                    'train/recon_audio': list(map(toaudio, G_x, titles)),
                    'train/real_specs' : list(map(toimg, x, titles)),
                    'train/recon_specs': list(map(toimg, G_x, titles))}
        wandb.log(log_dict, step=global_step)

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    use_ddp = 'LOCAL_RANK' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = int(os.environ['LOCAL_RANK']) if use_ddp else 0
    logs = local_rank == 0 and not args.no_logs
    print('LOCAL_RANK: {}'.format(local_rank))
    torch.cuda.set_device(local_rank)
    
    if use_ddp:
        dist.init_process_group('nccl')
    
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    
    if logs:
        settings = wandb.Settings(start_method='fork')
        if args.resume:
            wandb.init(project='multi_scale_seq_model', entity='falkaer',
                       resume='must', id=args.resume, settings=settings)
        else:
            wandb.init(project='multi_scale_seq_model', entity='falkaer',
                       settings=settings)
            wandb.config.update(args)
        global_step = wandb.run.step
    
    effective_batch_size = args.batch_size
    if use_ddp:
        effective_batch_size *= dist.get_world_size()
    
    def collate_fn(batch):
        embeds, ids, offsets, fnames, titles = zip(*batch)
        return nn.utils.rnn.pad_sequence(embeds, batch_first=True), ids, offsets, fnames, titles
    
    if args.dataset == 'maestro':
        train_dataset = MAESTRO(split='train')
        test_dataset = MAESTRO(split='test')
        valid_dataset = MAESTRO(split='validation')
    else:
        train_dataset = FMA(fma_size='medium', split='train')
        valid_dataset = FMA(fma_size='medium', split='validation')
        test_dataset = FMA(fma_size='medium', split='test')
    
    EmbeddingDataset = partial(EmbeddingDataset, checkpoint_dir=args.quant_checkpoint_dir,
                               run_name=args.quant_run_name, run_step=args.quant_run_step)
    
    train_dataset = EmbeddingDataset(train_dataset, num_embeddings=args.seq_len)
    test_dataset = EmbeddingDataset(test_dataset, num_embeddings=5 * EMBED_PER_SEC)
    valid_dataset = EmbeddingDataset(valid_dataset, num_embeddings=5 * EMBED_PER_SEC)
    
    train_dataset = OversamplingDataset(train_dataset, oversampling_factor=args.dataset_oversampling)
    
    if use_ddp:  # + world_size to use seeds greater than the manual_seed on each replica
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed + dist.get_world_size())
    else:
        train_sampler = RandomSampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=8, prefetch_factor=4, pin_memory=True)
    
    snake_alpha = 0.5
    dec_activation = {'snake'                   : lambda c: Snake(c, init=snake_alpha),
                      'snake_periodic'          : lambda c: Snake(c, init='periodic'),
                      'snake_corrected'         : lambda c: Snake(c, init=snake_alpha, correction='std'),
                      'snake_periodic_corrected': lambda c: Snake(c, init='periodic', correction='std'),
                      'elu'                     : lambda _: nn.ELU(),
                      'gelu'                    : lambda _: nn.GELU()}[args.dec_activation]
    
    upsample = {'pqmf'        : PQMFUpsample,
                'conv'        : ConvUpsample,
                'pixelshuffle': PixelShuffleUpsample}[args.upsample]
    
    decoder = Decoder(D=args.dim, C=args.C_dec, activation=dec_activation, upsample=upsample).eval()
    
    if args.weight_norm:
        recursive_weight_norm(decoder)
    
    quantizer = ResidualVQ(dim=args.dim,
                           num_quantizers=args.num_quantizers,
                           codebook_size=args.codebook_size,
                           codebook_dim=args.codebook_dim,
                           shared_codebook=args.shared_codebook,
                           use_cosine_sim=args.use_cosine_sim,
                           sample_codebook_temp=args.codebook_temp).cuda()
    
    checkpoint_path = get_checkpoint(args.quant_checkpoint_dir, args.quant_run_name, args.quant_run_step)
    if use_ddp:
        l = [checkpoint_path]
        dist.broadcast_object_list(l, src=0)
        checkpoint_path = l[0]
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    load_state_dicts({'quantizer': quantizer,
                      'decoder'  : decoder}, chkpt)
    
    quantizer.eval()
    
    attention = {'flash' : flash_attention,
                 'simple': simple_attention}[args.attention]
    norm_dict = {'none'     : nn.Identity,
                 'layernorm': nn.LayerNorm,
                 'scalenorm': ScaleNorm,
                 'scale'    : Scale}
    
    transformer_class = {'regular'          : RQTransformer,
                         'conformer'        : RQConformer,
                         'dilated_conformer': RQDilatedConformer}[args.transformer]
    
    activation = {'snake'                   : lambda c: Snake(c, init=snake_alpha),
                  'snake_periodic'          : lambda c: Snake(c, init='periodic'),
                  'snake_corrected'         : lambda c: Snake(c, init=snake_alpha, correction='std'),
                  'snake_periodic_corrected': lambda c: Snake(c, init='periodic', correction='std'),
                  'elu'                     : lambda _: nn.ELU(),
                  'gelu'                    : lambda _: nn.GELU()}[args.activation]
    
    model = transformer_class(quantizer=quantizer,
                              inner_dim=args.inner_dim,
                              time_layers=args.time_layers,
                              resolution_layers=args.resolution_layers,
                              heads=args.heads,
                              dim_head=args.dim_head,
                              embed_dropout=args.embed_dropout,
    
                              ff_dropout=args.ff_dropout,
                              ff_mult=args.ff_mult,
                              ff_prenorm=norm_dict[args.ff_prenorm],
                              ff_postnorm=norm_dict[args.ff_postnorm],
    
                              attention=attention,
                              attn_prenorm=norm_dict[args.attn_prenorm],
                              attn_postnorm=norm_dict[args.attn_postnorm],
                              attn_dropout=args.attn_dropout,
                              postattn_dropout=args.postattn_dropout,
    
                              prenorm=norm_dict[args.transformer_prenorm],
                              midnorm=norm_dict[args.transformer_midnorm],
                              postnorm=norm_dict[args.transformer_postnorm],
    
                              use_rotary=args.use_rotary,
                              use_alibi=args.use_alibi,
                              causal=True).cuda()
    
    if args.init != 'none' and not args.resume:
        init_fun = {'none'  : lambda x: x,
                    'tfixup': tfixup_init_,
                    'small' : small_init_}[args.init]
        init_fun(model)
    
    if local_rank == 0:
        print('Parameter counts:\n',
              'RQ-transformer: {}\n'.format(count_parameters(model)),
              'Time transformer: {}\n'.format(count_parameters(model.time_transformer)),
              'Resolution transformer: {}\n'.format(count_parameters(model.resolution_transformer)))
    
    if logs:
        wandb.watch(model, log='gradients', log_freq=args.grad_log_freq)
    
    # optim = opt.AdamW(ddp_model.parameters(), lr=args.model_lr,
    #                   betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    optim = configure_optimizer(model, lr=args.model_lr,
                                weight_decay=args.weight_decay,
                                betas=(args.beta1, args.beta2))
    
    if use_ddp:
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = model
    
    scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 16)
    # scheduler steps every training step, not epoch
    scheduler = WarmupCosineAnnealingScheduler(optim, num_warmup_steps=args.warmup_steps,
                                               num_training_steps=args.epochs * len(train_loader))
    
    # TODO: higher lr but more warmup steps?
    # TODO: shared token embeddings - they actually do embed things and use smallinit (0.02 at least)
    # TODO: they also use absolute embeddings (just additive embeddings - does it make sense?)
    # TODO: https://github.com/kakaobrain/rq-vae-transformer/blob/2bf6ece4b85608cfae4c0e2969b17f75495e1639/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml
    # TODO: warmup
    # TODO: cosine annealing?
    
    if args.resume:
        if logs:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        else:
            checkpoint_path = ''
        if use_ddp:
            l = [checkpoint_path]
            dist.broadcast_object_list(l, src=0)
            checkpoint_path = l[0]
        chkpkt = torch.load(checkpoint_path, map_location='cpu')
        load_state_dicts({
            'model'    : ddp_model,
            'optim'    : optim,
            'scheduler': scheduler,
            'scaler'   : scaler
        }, chkpkt)
        epoch = chkpkt['epoch']
    else:
        epoch = 0
    
    amp_context = torch.cuda.amp.autocast if args.use_amp else nullcontext
    
    if args.profile:
        context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('profile', 'epoch'),
                with_flops=True, with_stack=True,
                with_modules=True, profile_memory=True
        )
        context.__enter__()
    
    metrics = MetricDict()
    
    for epoch in range(epoch, epoch + args.epochs):
        train()
        if args.profile:
            context.step()
            if epoch == 4:
                context.__exit__()
        
        with torch.inference_mode():
            # if (epoch + 1) % args.validate_every == 0:
            if (epoch + 1) % args.validate_every == 0:
                validate(valid_dataset)
            if (epoch + 1) % args.test_every == 0:
                validate(test_dataset)
            if (epoch + 1) % args.checkpoint_every == 0 and logs:
                save_checkpoint({
                    'model'    : ddp_model,
                    'optim'    : optim,
                    'scheduler': scheduler,
                    'scaler'   : scaler,
                    'epoch'    : epoch
                }, args.checkpoint_dir)
