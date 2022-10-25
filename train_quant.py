import argparse
from functools import partial

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler

from einops import rearrange

from torchvision.utils import make_grid
import torchaudio

from tqdm import tqdm
import os
import os.path as osp

from contextlib import nullcontext

from fad_helper import FADHelper
from fma_dataset import FMA
from pqmf import PQMFAnalysisLowPass
from vq.losses import SpectralReconstructionLoss, l1_mel_loss, feature_loss, \
    hinge_adv_D_loss, hinge_adv_G_loss, ls_adv_D_loss, ls_adv_G_loss
from snake import Snake, snake_kaiming_normal_

from vq import ResidualVQ
from vq.model import Encoder, Decoder
from vq.model import ConvDownsample, ConvUpsample, PQMFDownsample, PQMFLowPassDownsample, PQMFUpsample, \
    PixelShuffleDownsample, \
    PixelShuffleUpsample

from vq.soundstream_discriminators import STFTDiscriminator, WaveDiscriminator
from vq.avocodo_discriminators import MultiCoMBDiscriminator, MultiSubBandDiscriminator

from maestro_dataset import MAESTRO
from stft import MelSTFTSpectrogram
from util import MetricDict, OversamplingDataset, broadcast_object, count_parameters, \
    get_latest_checkpoint, isfinite, isscalar, load_state_dicts, save_checkpoint, recursive_weight_norm

EPS = 1e-10
torch.backends.cudnn.benchmark = True

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_logs', default=False, action='store_true')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Accumulate training statistics and log every n batches')
    parser.add_argument('--resume', type=str, default=None, help='Resume run with given ID from latest checkpoint')
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--dataset', type=str, default='maestro', choices=['maestro', 'fma'])
    parser.add_argument('--dataset_oversampling', type=int, default=1, help='How much to oversample the dataset by')
    parser.add_argument('--clip_length', type=float, default=15, help='Length of audio clips (in seconds)')
    
    model_args = parser.add_argument_group('model_args')
    model_args.add_argument('--dim', type=int, default=256, help='Dimensionality of embeddings')
    model_args.add_argument('--activation', type=str, default='snake',
                            choices=['elu', 'gelu', 'snake', 'snake_periodic',
                                     'snake_corrected', 'snake_periodic_corrected'])
    model_args.add_argument('--upsample', type=str, default='conv', choices=['conv', 'pqmf', 'pixelshuffle'])
    model_args.add_argument('--downsample', type=str, default='conv',
                            choices=['conv', 'pqmf', 'pqmf_lowpass', 'pixelshuffle'])
    model_args.add_argument('--C_enc', type=int, default=32, help='No. parameter scaling factor of encoder')
    model_args.add_argument('--C_dec', type=int, default=32, help='No. parameter scaling factor of decoder')
    model_args.add_argument('--discriminators', type=str, default='avocodo', choices=['soundstream', 'avocodo'])
    model_args.add_argument('--disc_downsample', type=str, default='avgpool', choices=['avgpool', 'pqmf'])
    model_args.add_argument('--weight_norm', action='store_true')
    model_args.add_argument('--init', type=str, default='none', choices=['none', 'kaiming'])
    
    quantizer_args = parser.add_argument_group('quantizer_args')
    quantizer_args.add_argument('--num_quantizers', type=int, default=8)
    quantizer_args.add_argument('--codebook_size', type=int, default=1024)
    quantizer_args.add_argument('--codebook_dim', type=int, default=None)
    quantizer_args.add_argument('--shared_codebook', action='store_true')
    quantizer_args.add_argument('--ema_decay', type=float, default=0.99)
    
    # quantizer tricks
    quantizer_args.add_argument('--use_cosine_sim', action='store_true')
    quantizer_args.add_argument('--threshold_ema_dead_code', type=float, default=2)
    quantizer_args.add_argument('--orthogonal_reg_max_codes', type=float, default=None)
    quantizer_args.add_argument('--orthogonal_reg_active_codes_only', action='store_true')
    quantizer_args.add_argument('--codebook_temp', type=float, default=0)
    quantizer_args.add_argument('--codebook_temp_decay', type=float, default=1)
    quantizer_args.add_argument('--kmeans_init', action='store_true')
    quantizer_args.add_argument('--drop_code', type=float, default=0.1, help='Probability of not quantizing a level')
    
    discriminator_args = parser.add_argument_group('discriminator_args')
    discriminator_args.add_argument('--disc_C', type=int, default=16,
                                    help='No. parameter scaling factor of discriminators')
    discriminator_args.add_argument('--disc_D', type=int, default=3,
                                    help='No. downsampling blocks in wave discriminator')
    
    optim_args = parser.add_argument_group('optim_args')
    optim_args.add_argument('--lambda_adv', type=float, default=1, help='Adversarial loss factor')
    optim_args.add_argument('--lambda_feat', type=float, default=100, help='Feature loss factor')
    optim_args.add_argument('--lambda_rec', type=float, default=1, help='Reconstruction loss factor')
    optim_args.add_argument('--lambda_commit', type=float, default=1, help='Commitment loss factor')
    optim_args.add_argument('--lambda_ortho', type=float, default=10,
                            help='Orthogonal codebook regularization loss factor')
    optim_args.add_argument('--gan_loss', type=str, default='ls', choices=['ls', 'hinge'])
    optim_args.add_argument('--rec_loss', type=str, default='l1', choices=['l1', 'spectral'])
    optim_args.add_argument('--batch_size', type=int, default=8)
    optim_args.add_argument('--model_lr', type=float, default=1e-4)
    optim_args.add_argument('--disc_lr', type=float, default=1e-4)
    optim_args.add_argument('--lr_decay', type=float, default=0.975)
    optim_args.add_argument('--beta1', type=float, default=0.5)
    optim_args.add_argument('--beta2', type=float, default=0.9)
    optim_args.add_argument('--weight_decay', type=float, default=1e-2)
    optim_args.add_argument('--use_amp', default=False, action='store_true')
    optim_args.add_argument('--offload_activations', default=False, action='store_true')
    optim_args.add_argument('--checkpoints', type=int, default=0)
    return parser

def validate(dataset):
    model.eval()
    disc_model.eval()
    
    val_metrics = MetricDict()
    
    sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                        collate_fn=collate_fn, num_workers=8, prefetch_factor=4, pin_memory=True)
    
    code_counts = torch.zeros(args.codebook_size, args.num_quantizers, dtype=torch.int64, device='cuda')
    
    if logs:
        fad_dir = osp.join('fad', wandb.run.name, args.dataset, dataset.split)
    else:
        fad_dir = None
    
    if use_ddp:
        sampler.set_epoch(epoch)
        fad_dir = broadcast_object(fad_dir, src=0)
    
    fad_helper = FADHelper(fad_dir)
    fad_helper.cleanup()
    
    # make sure no one deletes the fad workdir after this point
    if use_ddp:
        dist.barrier()
    
    for x, ids, offsets, fnames, titles in tqdm(loader, total=len(loader), disable=local_rank != 0):
        with amp_context():
            x = x.cuda(non_blocking=True)
            G_x, gen_fmaps, codes, num_expired, quant_losses = model(x)
            commit_losses, ortho_losses = quant_losses.unbind(dim=0)
            commit_loss = commit_losses.mean()
            ortho_loss = ortho_losses.mean()
            
            codes = rearrange(codes, 'b n q -> (b n) q')
            code_counts.scatter_add_(0, codes, codes.new_ones(()).expand_as(codes))
            
            all_D_x, all_D_x_fmaps = disc_model(x)
            all_D_G_x, all_D_G_x_fmaps = disc_model(G_x, gen_fmaps)
            
            adv_g_losses = torch.stack([adv_G_loss_fun(D_G_x) for D_G_x in all_D_G_x])
            adv_g_loss = adv_g_losses.mean()
            
            feat_g_losses = torch.stack([feature_loss(D_x_fmaps, D_G_x_fmaps)
                                         for D_x_fmaps, D_G_x_fmaps in
                                         zip(all_D_x_fmaps, all_D_G_x_fmaps)])
            feat_g_loss = feat_g_losses.mean()
            
            spec_g_losses = rec_loss_fun(x, G_x)
            spec_g_loss = spec_g_losses.mean()
            
            adv_d_losses = torch.stack([adv_D_loss_fun(D_x, D_G_x) for D_x, D_G_x in zip(all_D_x, all_D_G_x)])
            adv_d_loss = adv_d_losses.mean()
            
            gen_loss = (args.lambda_adv * adv_g_loss
                        + args.lambda_feat * feat_g_loss
                        + args.lambda_rec * spec_g_loss
                        + args.lambda_commit * commit_loss
                        + args.lambda_ortho * ortho_loss)
            
            val_metrics.update({
                '{}/gen_loss'     : gen_loss,
                '{}/commit_loss'  : commit_loss,
                '{}/ortho_loss'   : ortho_loss,
                '{}/adv_g_loss'   : adv_g_loss,
                '{}/feat_g_loss'  : feat_g_loss,
                '{}/spec_g_loss'  : spec_g_loss,
                '{}/adv_d_loss'   : adv_d_loss,
                
                # sublosses
                '{}/commit_losses': commit_losses,
                '{}/ortho_losses' : ortho_losses,
                '{}/adv_g_losses' : adv_g_losses,
                '{}/feat_g_losses': feat_g_losses,
                '{}/spec_g_losses': spec_g_losses,
                '{}/adv_d_losses' : adv_d_losses
            })
        
        # only add 5 second clips
        fad_helper.add_files(G_x[:, :5 * dataset.sample_rate], fnames, sr=dataset.sample_rate)
    
    val_metrics = dict(val_metrics)
    
    if use_ddp:
        for x in val_metrics.values():
            dist.all_reduce(x)
            x /= dist.get_world_size()
        dist.all_reduce(code_counts)
    
    fad_helper.flush_file_list()
    
    if logs:
        
        fad_helper.generate_stats(batch_size=64)
        fad_score = fad_helper.compute_fad(osp.join('fad', 'background', args.dataset, dataset.split, '0', 'stats'))
        val_metrics['{}/fad_score'] = fad_score
        
        p_codes = code_counts / code_counts.sum(dim=0, keepdim=True)
        code_entropy = -torch.sum(p_codes * torch.log2(p_codes + EPS), dim=0)
        code_perplexity = 2 ** code_entropy
        
        # get a few random samples
        inds = np.random.choice(len(dataset), 3, replace=False)
        x, *_ = collate_fn([dataset[i] for i in inds])
        x = x.cuda()
        G_x, *_ = model(x)
        
        S_x = vis_spec(x)
        S_G_x = vis_spec(G_x)
        img = rearrange([S_x, S_G_x], 'b1 b2 f n -> (b2 b1) 1 f n')
        img = torch.flip(img, dims=(2,))
        img = make_grid(img, nrow=2, normalize=True, scale_each=True)
        x, G_x, img = map(torch.Tensor.cpu, (x, G_x, img))
        
        log_dict = {k.format(dataset.split): v for k, v in val_metrics.items()}
        log_dict = {k: v.cpu() if torch.is_tensor(v) else v for k, v in log_dict.items()}
        log_dict = {k: v for k, v in log_dict.items() if isfinite(v)}
        log_dict = {k: v if isscalar(v) else wandb.Histogram(v)
                    for k, v in log_dict.items()}
        
        log_dict.update({
            '{}/spec_samples'.format(dataset.split)     : wandb.Image(img),
            '{}/code_entropies'.format(dataset.split)   : wandb.Histogram(code_entropy.detach().cpu()),
            '{}/code_perplexities'.format(dataset.split): wandb.Histogram(code_perplexity.detach().cpu()),
            '{}/code_entropy'.format(dataset.split)     : code_entropy.mean(),
            '{}/code_perplexity'.format(dataset.split)  : code_perplexity.mean()
        })
        
        # sort so that we just get a measure of codebook "concentration"
        # log_dict.update({
        #     '{}/codebook_{}_count'.format(dataset.split, i + 1):
        #         wandb.Histogram(torch.sort(code_counts[:, i]).values.cpu(), num_bins=128)
        #     for i in range(args.num_quantizers)
        # })
        
        log_dict.update({
            '{}/codebook_{}_cluster_size'.format(dataset.split, i + 1):
                wandb.Histogram(torch.sort(vq._codebook.cluster_size).values.cpu(), num_bins=128)
            for i, vq in enumerate(quantizer.layers)
        })
        
        for i, ind in enumerate(inds):
            fname, title = dataset.get_labels(ind)
            log_dict.update({
                '{}/audio_orig_{}'.format(dataset.split, i) : wandb.Audio(x[i], caption=title,
                                                                          sample_rate=dataset.sample_rate),
                '{}/audio_recon_{}'.format(dataset.split, i): wandb.Audio(G_x[i], caption=title,
                                                                          sample_rate=dataset.sample_rate)
            })
        
        global global_step
        wandb.log(log_dict, step=global_step)

def step(x):
    with amp_context():
        x = x.cuda(non_blocking=True)
        G_x, gen_fmaps, codes, num_expired, quant_losses = model(x)
        commit_losses, ortho_losses = quant_losses.unbind(dim=0)
        commit_loss = commit_losses.mean()
        ortho_loss = ortho_losses.mean()
        
        _, all_D_x_fmaps = disc_model(x)
        all_D_G_x, all_D_G_x_fmaps = disc_model(G_x, gen_fmaps)
        
        adv_g_losses = torch.stack([adv_G_loss_fun(D_G_x) for D_G_x in all_D_G_x])
        adv_g_loss = adv_g_losses.mean()
        
        feat_g_losses = torch.stack([feature_loss(D_x_fmaps, D_G_x_fmaps)
                                     for D_x_fmaps, D_G_x_fmaps in
                                     zip(all_D_x_fmaps, all_D_G_x_fmaps)])
        feat_g_loss = feat_g_losses.mean()
        
        spec_g_losses = rec_loss_fun(x, G_x)
        spec_g_loss = spec_g_losses.mean()
        
        gen_loss = (args.lambda_adv * adv_g_loss
                    + args.lambda_feat * feat_g_loss
                    + args.lambda_rec * spec_g_loss
                    + args.lambda_commit * commit_loss
                    + args.lambda_ortho * ortho_loss)
    
    g_optim.zero_grad(set_to_none=True)
    scaler.scale(gen_loss).backward()
    scaler.step(g_optim)
    
    # 2nd forward pass
    with amp_context():
        all_D_x, _ = disc_model(x)
        all_D_G_x, _ = disc_model(G_x.detach(), [fmap.detach() for fmap in gen_fmaps])
        adv_d_losses = torch.stack([adv_D_loss_fun(D_x, D_G_x) for D_x, D_G_x in zip(all_D_x, all_D_G_x)])
        adv_d_loss = adv_d_losses.mean()
    
    disc_loss = args.lambda_adv * adv_d_loss
    
    d_optim.zero_grad(set_to_none=True)
    scaler.scale(disc_loss).backward()
    scaler.step(d_optim)
    
    scaler.update()
    assert scaler.get_scale() > 0
    
    if logs:
        
        global global_step, metrics
        
        p_codes = [vq.cluster_size / vq.cluster_size.sum() for vq in quantizer.codebooks]
        p_codes = torch.stack(p_codes, dim=0)
        
        code_entropy = - torch.sum(p_codes * torch.log2(p_codes + EPS), dim=-1)
        code_perplexity = 2 ** code_entropy
        
        metrics.update({
            'train/gen_loss'         : gen_loss,
            'train/commit_loss'      : commit_loss,
            'train/ortho_loss'       : ortho_loss,
            'train/adv_g_loss'       : adv_g_loss,
            'train/feat_g_loss'      : feat_g_loss,
            'train/spec_g_loss'      : spec_g_loss,
            'train/adv_d_loss'       : adv_d_loss,
            
            'train/num_expired'      : num_expired,
            
            # sublosses
            'train/commit_losses'    : commit_losses,
            'train/ortho_losses'     : ortho_losses,
            'train/adv_g_losses'     : adv_g_losses,
            'train/feat_g_losses'    : feat_g_losses,
            'train/spec_g_losses'    : spec_g_losses,
            'train/adv_d_losses'     : adv_d_losses,
            
            # codebook metrics
            'train/code_entropies'   : code_entropy,
            'train/code_perplexities': code_perplexity,
            
            'train/code_entropy'     : code_entropy.mean(),
            'train/code_perplexity'  : code_perplexity.mean(),
            
            'train/codebook_temp'    : args.codebook_temp * args.codebook_temp_decay ** epoch,
            'train/gen_lr'           : g_scheduler.get_last_lr()[0],
            'train/disc_lr'          : d_scheduler.get_last_lr()[0],
            'train/grad_scaler'      : scaler.get_scale()
        })
        
        if global_step > 0 and global_step % args.log_every == 0:
            metrics = {k: v.cpu() if torch.is_tensor(v) else v for k, v in metrics.items()}
            metrics = {k: v for k, v in metrics.items() if isfinite(v)}
            metrics = {k: v if isscalar(v) else wandb.Histogram(v) for k, v in metrics.items()}
            wandb.log(metrics, step=global_step)
            metrics = MetricDict()
        
        global_step += effective_batch_size

def train():
    model.train()
    disc_model.train()
    
    if use_ddp:
        train_sampler.set_epoch(epoch)
    
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
    
    for x, ids, offsets, fnames, titles in tqdm(train_loader, total=len(train_loader), disable=local_rank != 0):
        step(x)
        
        if args.profile:
            context.step()
            if epoch == 4:
                context.__exit__()
    
    g_scheduler.step()
    d_scheduler.step()
    
    # anneal the codebook sampling temperature
    for vq in quantizer.layers:
        vq._codebook.sample_codebook_temp = args.codebook_temp * args.codebook_temp_decay ** epoch

class ModelWrapper(nn.Module):
    def __init__(self, encoder, quantizer, decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def forward(self, x):
        x = rearrange(x, 'b n -> b 1 n')
        e, enc_fmaps = self.encoder(x)
        e = rearrange(e, 'b d n -> b n d')
        quantized, codes, num_expired, losses = self.quantizer(e.float())
        quantized = rearrange(quantized, 'b n d -> b d n')
        # with forward_context():
        o, dec_fmaps = self.decoder(quantized)
        o = rearrange(o, 'b 1 n -> b n')
        x_hat = torch.tanh(o.float())
        # last two downsampled fmaps from decoder
        fmaps = dec_fmaps[-3:-1]
        return x_hat, fmaps, codes, num_expired, losses

class DiscriminatorWrapper(nn.Module):
    def __init__(self, *discs):
        super().__init__()
        self.discs = nn.ModuleList(discs)
    
    def forward(self, x, x_fmaps=None):
        all_scores, all_fmaps = [], []
        for d in self.discs:
            scores, fmaps = d(x, x_fmaps)
            all_scores.extend(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps

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
            wandb.init(project='multi_scale_music', entity='falkaer',
                       resume='must', id=args.resume, settings=settings)
        else:
            wandb.init(project='multi_scale_music', entity='falkaer',
                       settings=settings)
            wandb.config.update(args)
        global_step = wandb.run.step
    
    effective_batch_size = args.batch_size
    if use_ddp:
        effective_batch_size *= dist.get_world_size()
    
    def collate_fn(batch):
        waveforms, ids, offsets, fnames, titles = zip(*batch)
        return nn.utils.rnn.pad_sequence(waveforms, batch_first=True), ids, offsets, fnames, titles
    
    if args.dataset == 'maestro':
        train_dataset = MAESTRO(int(args.clip_length * MAESTRO.sample_rate), split='train')
        valid_dataset = MAESTRO(int(args.clip_length * MAESTRO.sample_rate), split='validation')
        test_dataset = MAESTRO(int(args.clip_length * MAESTRO.sample_rate), split='test')
    else:
        train_dataset = FMA(int(args.clip_length * FMA.sample_rate), fma_size='medium', split='train')
        valid_dataset = FMA(int(args.clip_length * FMA.sample_rate), fma_size='medium', split='validation')
        test_dataset = FMA(int(args.clip_length * FMA.sample_rate), fma_size='medium', split='test')
    
    train_dataset = OversamplingDataset(train_dataset, oversampling_factor=args.dataset_oversampling)
    
    if use_ddp:  # + world_size to use seeds greater than the manual_seed on each replica
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed + dist.get_world_size())
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=8, prefetch_factor=4, pin_memory=True)
    
    snake_alpha = 0.5
    activation = {'snake'                   : lambda c: Snake(c, init=snake_alpha),
                  'snake_periodic'          : lambda c: Snake(c, init='periodic'),
                  'snake_corrected'         : lambda c: Snake(c, init=snake_alpha, correction='std'),
                  'snake_periodic_corrected': lambda c: Snake(c, init='periodic', correction='std'),
                  'elu'                     : lambda _: nn.ELU(),
                  'gelu'                    : lambda _: nn.GELU()}[args.activation]
    
    upsample = {'pqmf'        : PQMFUpsample,
                'conv'        : ConvUpsample,
                'pixelshuffle': PixelShuffleUpsample}[args.upsample]
    
    downsample = {'pqmf'        : PQMFDownsample,
                  'pqmf_lowpass': PQMFLowPassDownsample,
                  'conv'        : ConvDownsample,
                  'pixelshuffle': PixelShuffleDownsample}[args.downsample]
    
    encoder = Encoder(C=args.C_enc, D=args.dim, activation=activation, downsample=downsample,
                      checkpoints=args.checkpoints).cuda()
    decoder = Decoder(C=args.C_dec, D=args.dim, activation=activation, upsample=upsample,
                      checkpoints=args.checkpoints).cuda()
    
    # TODO: non-causal encoder/decoder
    # TODO: learning rate schedule - high lr first (~1e-3), then exponential decay to 1e-4 see one of the paper for their schedule
    # TODO: avocodo discriminators
    # TODO: bigger encoder / decoder
    # TODO: consistent and more principled way to calculate fad score
    # TODO: use the gumbel softmax sampling with a schedule like in one of the papers
    
    # TODO: is the high-alpha init unstable because we need to use a low-pass filter after snake?
    # TODO: torch.autograd.graph.save_on_cpu ?
    # TODO: channels last?
    
    quantizer = ResidualVQ(dim=args.dim,
                           num_quantizers=args.num_quantizers,
                           codebook_size=args.codebook_size,
                           codebook_dim=args.codebook_dim,
                           shared_codebook=args.shared_codebook,
                           decay=args.ema_decay,
                           use_cosine_sim=args.use_cosine_sim,
                           threshold_ema_dead_code=args.threshold_ema_dead_code,
                           orthogonal_reg_weight=1 if args.lambda_ortho > 0 else 0,
                           sample_codebook_temp=args.codebook_temp,
                           sync_codebook=use_ddp,
                           kmeans_init=args.kmeans_init, kmeans_iters=100).cuda()
    
    if args.discriminators == 'soundstream':
        disc_downsample = {'avgpool': lambda f: nn.AvgPool1d(kernel_size=2 * f,
                                                             stride=f, padding=1,
                                                             count_include_pad=False),
                           'pqmf'   : PQMFAnalysisLowPass}[args.disc_downsample]
        
        discs = [
            WaveDiscriminator(C=args.disc_C, num_D=args.disc_D, downsampling_factor=2,
                              downsample=disc_downsample).cuda(),
            STFTDiscriminator(C=args.disc_C, F_bins=1024 // 2).cuda()
        ]
    elif args.discriminators == 'avocodo':
        discs = [
            MultiSubBandDiscriminator(C=args.disc_C,
                                      freq_init_ch=args.clip_length * train_dataset.sample_rate // 64).cuda(),
            MultiCoMBDiscriminator(C=args.disc_C, fmap_channels=[4 * args.C_dec, 2 * args.C_dec]).cuda()
        ]
    
    model = ModelWrapper(encoder, quantizer, decoder)
    disc_model = DiscriminatorWrapper(*discs)
    
    if args.init != 'none' and not args.resume:
        if args.activation.startswith('snake'):
            correction = 'std' if args.activation.endswith('corrected') else None
            init_fun = lambda x: snake_kaiming_normal_(x, snake_alpha, correction=correction)
        else:
            init_fun = init.kaiming_normal_
        
        encoder.reset_parameters(init_fun)
        decoder.reset_parameters(init_fun)
        # for disc in discs:
        #     disc.reset_parameters(init_fun)
    
    if args.weight_norm:
        recursive_weight_norm(encoder)
        recursive_weight_norm(decoder)
        recursive_weight_norm(disc_model)
    
    if local_rank == 0:
        print('Parameter counts:\n',
              'Encoder: {}\n'.format(count_parameters(encoder)),
              'Decoder: {}\n'.format(count_parameters(decoder)),
              'Discriminators: {}\n'.format(list(map(count_parameters, discs))))
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
        disc_model = DDP(disc_model, device_ids=[local_rank])
    
    g_optim = optim.AdamW(model.parameters(), lr=args.model_lr,
                          betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    d_optim = optim.AdamW(disc_model.parameters(), lr=args.disc_lr,
                          betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler(init_scale=1, growth_interval=1_000_000_000)  # never grow
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optim, gamma=args.lr_decay)
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optim, gamma=args.lr_decay)
    
    if args.resume:
        if logs:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        else:
            checkpoint_path = ''
        if use_ddp:
            l = [checkpoint_path]
            dist.broadcast_object_list(l, src=0)
            checkpoint_path = l[0]
        chkpt = torch.load(checkpoint_path, map_location='cpu')
        load_state_dicts({
            'encoder'    : encoder,
            'quantizer'  : quantizer,
            'decoder'    : decoder,
            'discs'      : discs,
            'd_optim'    : d_optim,
            'g_optim'    : g_optim,
            'g_scheduler': g_scheduler,
            'd_scheduler': d_scheduler,
            'scaler'     : scaler
        }, chkpt)
        epoch = chkpt['epoch']
        for vq in quantizer.layers:
            vq._codebook.sample_codebook_temp = args.codebook_temp * args.codebook_temp_decay ** epoch
    else:
        epoch = 0
    
    if args.rec_loss == 'spectral':
        win_lengths = [2 ** i for i in range(6, 12)]
        melspecs = [MelSTFTSpectrogram(s, s // 4, n_mels=64,
                                       sample_rate=train_dataset.sample_rate).cuda() for s in win_lengths]
        rec_loss_fun = SpectralReconstructionLoss(win_lengths, melspecs)
    elif args.rec_loss == 'l1':
        rec_loss_fun = l1_mel_loss
    
    if args.gan_loss == 'ls':
        adv_D_loss_fun = ls_adv_D_loss
        adv_G_loss_fun = ls_adv_G_loss
    elif args.gan_loss == 'hinge':
        adv_D_loss_fun = hinge_adv_D_loss
        adv_G_loss_fun = hinge_adv_G_loss
    
    offload_context = partial(torch.autograd.graph.save_on_cpu, pin_memory=True) \
        if args.offload_activations else nullcontext
    amp_context = torch.cuda.amp.autocast if args.use_amp else nullcontext
    
    vis_spec = MelSTFTSpectrogram(1024, 1024 // 4, n_mels=128, sample_rate=train_dataset.sample_rate).cuda()
    
    metrics = MetricDict()
    
    for epoch in range(epoch, epoch + args.epochs):
        train()
        
        with torch.inference_mode():
            if (epoch + 1) % args.validate_every == 0:
                validate(valid_dataset)
            if (epoch + 1) % args.test_every == 0:
                validate(test_dataset)
            if (epoch + 1) % args.checkpoint_every == 0 and logs:
                save_checkpoint({
                    'encoder'    : encoder,
                    'quantizer'  : quantizer,
                    'decoder'    : decoder,
                    # 'wave_disc': wave_disc,
                    # 'stft_disc': stft_disc,
                    'discs'      : discs,
                    'd_optim'    : d_optim,
                    'g_optim'    : g_optim,
                    'g_scheduler': g_scheduler,
                    'd_scheduler': d_scheduler,
                    'scaler'     : scaler,
                    'epoch'      : epoch
                }, args.checkpoint_dir)
