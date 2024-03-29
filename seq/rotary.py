# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
# We split the input differently ((d 2) -> d 2 instead of (2 d) -> d 2), following the original
# paper's implementation. This should not matter.

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This implementation is inspired by GPT-NeoX https://github.com/EleutherAI/gpt-neox
# NOTE: Almost the same right now, moving parts to Triton is the next step

from typing import Tuple
import math

import torch
import torch.nn as nn

from einops import repeat

def rotate_half(x):
    # rearrange doesn't work with torch.jit
    # x = rearrange(x, '... (d r) -> ... d r', r=2)
    x = x.unflatten(dim=-1, sizes=(-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack((-x2, x1), dim=-1)
    # return rearrange(rotated_x, '... d r -> ... (d r)')
    return rotated_x.flatten(start_dim=-2)

@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int = -2):
    # NOTE: This could probably be moved to Triton
    
    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:x.shape[seq_dimension], :]
    sin = sin[:x.shape[seq_dimension], :]
    if seq_dimension == -3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """
    
    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('_cos_cached', None, persistent=False)
        self.register_buffer('_sin_cached', None, persistent=False)
        self._seq_len_cached = None
    
    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]
        
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seq_len != self._seq_len_cached or self._cos_cached.device != x.device
                or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = repeat(torch.cos(freqs).to(x.dtype), '... d -> ... (d 2)')
            self._sin_cached = repeat(torch.sin(freqs).to(x.dtype), '... d -> ... (d 2)')
        
        return self._cos_cached, self._sin_cached
    
    def forward(self, q, k, seq_dimension=-2):
        assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension)
        return apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension), \
               apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension)
