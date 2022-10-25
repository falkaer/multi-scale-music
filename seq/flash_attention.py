import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

@triton.jit
def make_bounds(offs_m, offs_n, M, N,
                EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M:
        mask = offs_n[None, :] < N
    elif EVEN_N:
        mask = offs_m[:, None] < M
    else:
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    return mask

@triton.jit
def bounds_mask(offs_m, offs_n, M, N,
                EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        val = 0
    else:
        mask = make_bounds(offs_m, offs_n, M, N, EVEN_M, EVEN_N)
        val = tl.where(mask, 0, float('-inf'))
    return val

@triton.jit
def causal_mask(offs_m, offs_n, M, N,
                EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    shift = N - M
    mask = shift + offs_m[:, None] >= offs_n[None, :]
    if not (EVEN_M & EVEN_N):
        mask = mask & make_bounds(offs_m, offs_n, M, N, EVEN_M, EVEN_N)
    return tl.where(mask, 0, float('-inf'))

@triton.jit
def causal_alibi_mask(slope, offs_m, offs_n, M, N,
                      EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    shift = N - M
    alibi = (offs_n[None, :] - offs_m[:, None] - shift) * slope
    mask = alibi <= 0
    if not (EVEN_M & EVEN_N):
        mask = mask & make_bounds(offs_m, offs_n, M, N, EVEN_M, EVEN_N)
    return tl.where(mask, alibi, float('-inf'))

@triton.jit
def symmetric_alibi_mask(slope, offs_m, offs_n, M, N,
                         EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    alibi = -tl.abs((M - N) + offs_n[None, :] - offs_m[:, None]) * slope
    if not (EVEN_M & EVEN_N):
        mask = make_bounds(offs_m, offs_n, M, N, EVEN_M, EVEN_N)
        mask, alibi = tl.broadcast(mask, alibi)
        alibi = tl.where(mask, alibi, float('-inf'))
    return alibi

@triton.jit
def apply_dropout(x, offsets, p, seed, mask_val=float('-inf')):
    rand = tl.rand(seed, offsets)
    scale = 1 / (1 - p)
    return tl.where(rand > p, x * scale, mask_val)

@triton.jit
def _fwd_kernel(
        Q, K, V, S, Out, sm_scale,
        TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_tz, stride_th, stride_tm,
        stride_lz, stride_lh, stride_lm,
        stride_mz, stride_mh, stride_mm,
        M_Q, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
        CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr
):
    start_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_z * stride_tz + off_h * stride_th + offs_m * stride_tm
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M_Q, other=0)
    # q = tl.load(q_ptrs)
    q = q.to(tl.float16)
    if USE_ALIBI:
        slope = tl.load(S + off_h)
    if CAUSAL & EVEN_M & EVEN_N:
        bound = start_m + BLOCK_M
    else:
        bound = N_CTX
    # loop over k, v and update accumulator
    for start_n in range(0, bound, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=start_n + offs_n[:, None] < N_CTX, other=0)
        # k = tl.load(k_ptrs)
        k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        
        if USE_ALIBI & CAUSAL:
            qk += causal_alibi_mask(slope, offs_m, start_n + offs_n, M_Q, N_CTX, EVEN_M, EVEN_N)
        elif USE_ALIBI:
            qk += symmetric_alibi_mask(slope, offs_m, start_n + offs_n, M_Q, N_CTX, EVEN_M, EVEN_N)
        elif CAUSAL:
            qk += causal_mask(offs_m, start_n + offs_n, M_Q, N_CTX, EVEN_M, EVEN_N)
        else:
            qk += bounds_mask(offs_m, start_n + offs_n, M_Q, N_CTX, EVEN_M, EVEN_N)
        
        # -- compute m_ij, p, l_ij
        m_ij = tl.maximum(tl.max(qk, axis=1), -10000)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)  # 0 < alpha <= 1
        beta = tl.exp(m_ij - m_i_new)  # 0 <= beta < 1
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)
        acc = acc * acc_scale[:, None]
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=start_n + offs_n[:, None] < N_CTX, other=0)
        # v = tl.load(v_ptrs)
        v = v.to(tl.float16)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        
        v_ptrs += BLOCK_N * stride_vn
        k_ptrs += BLOCK_N * stride_kn
        # TODO: inplace add BLOCK_N to offs_n in each iteration?
        # offs_n += BLOCK_N # causes segfault atm
    
    # rematerialize offsets to save registers
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_z * stride_lz + off_h * stride_lh + offs_m * stride_lm
    m_ptrs = M + off_z * stride_mz + off_h * stride_mh + offs_m * stride_mm
    if EVEN_M:
        tl.store(l_ptrs, l_i)
        tl.store(m_ptrs, m_i)
    else:
        tl.store(l_ptrs, l_i, mask=offs_m < M_Q)
        tl.store(m_ptrs, m_i, mask=offs_m < M_Q)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out_ptrs = Out + off_o
    if EVEN_M:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < M_Q)

@triton.jit
def _bwd_preprocess(
        Out, DO, NDO, L, Delta, M_Q,
        stride_oz, stride_oh, stride_om, stride_od,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_ndoz, stride_ndoh, stride_ndom, stride_ndod,
        stride_lz, stride_lh, stride_lm,
        stride_dz, stride_dh, stride_dm,
        BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr,
        EVEN_M: tl.constexpr
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    # initialize offsets
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers
    Out = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    DO = DO + off_z * stride_doz + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    NDO = NDO + off_z * stride_ndoz + off_h * stride_ndoh + offs_m[:, None] * stride_ndom + offs_d[None,
                                                                                            :] * stride_ndod
    L = L + off_z * stride_lz + off_h * stride_lh + offs_m * stride_lm
    Delta = Delta + off_z * stride_dz + off_h * stride_dh + offs_m * stride_dm
    # load
    if EVEN_M:
        o = tl.load(Out).to(tl.float32)
        do = tl.load(DO).to(tl.float32)
        denom = tl.load(L).to(tl.float32)
    else:
        o = tl.load(Out, mask=offs_m[:, None] < M_Q).to(tl.float32)
        do = tl.load(DO, mask=offs_m[:, None] < M_Q).to(tl.float32)
        denom = tl.load(L, mask=offs_m < M_Q).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    if EVEN_M:
        tl.store(NDO, do)
        tl.store(Delta, delta)
    else:
        tl.store(NDO, do, mask=offs_m[:, None] < M_Q)
        tl.store(Delta, delta, mask=offs_m < M_Q)

@triton.jit
def _bwd_kernel(
        Q, K, V, S, sm_scale,
        DO, DQ, DK, DV, M, D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_doz, stride_doh, stride_dom, stride_dok,
        stride_dqz, stride_dqh, stride_dqm, stride_dqk,
        stride_dkz, stride_dkh, stride_dkn, stride_dkk,
        stride_dvz, stride_dvh, stride_dvn, stride_dvk,
        stride_mz, stride_mh, stride_mm,
        stride_dz, stride_dh, stride_dm,
        M_Q, N_CTX, BLOCK_DMODEL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
        CAUSAL: tl.constexpr, USE_ALIBI: tl.constexpr
):
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    DQ += off_z * stride_dqz + off_h * stride_dqh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    if USE_ALIBI:
        slope = tl.load(S + off_h)
    for start_n in range(0, N_CTX, BLOCK_N):
        # start_n = tl.multiple_of(start_n, BLOCK_M)
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = tl.arange(0, BLOCK_M)
        # initialize pointers to value-like data
        k_ptrs = K + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk)
        do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok)
        # pointer to row-wise quantities in value-like data
        m_ptrs = M + off_z * stride_mz + off_h * stride_mh + offs_m * stride_mm
        D_ptrs = D + off_z * stride_dz + off_h * stride_dh + offs_m * stride_dm
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        if EVEN_N:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0)
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0)
        k = k.to(tl.float16)
        v = v.to(tl.float16)
        if CAUSAL:
            begin = start_n + M_Q - N_CTX
            dq_ptrs += begin * stride_dqm
            q_ptrs += begin * stride_qm
            do_ptrs += begin * stride_dom
            m_ptrs += begin * stride_mm
            D_ptrs += begin * stride_dm
        else:
            begin = 0
        # loop over rows
        for start_m in range(begin, M_Q, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            if EVEN_M:
                q = tl.load(q_ptrs)
            else:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < M_Q, other=0)
            q = q.to(tl.float16)
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, k, trans_b=True)
            qk *= sm_scale
            if USE_ALIBI & CAUSAL:
                qk += causal_alibi_mask(slope, offs_m_curr, offs_n_curr, M_Q, N_CTX, EVEN_M, EVEN_N)
            elif USE_ALIBI:
                qk += symmetric_alibi_mask(slope, offs_m_curr, offs_n_curr, M_Q, N_CTX, EVEN_M, EVEN_N)
            elif CAUSAL:
                qk += causal_mask(offs_m_curr, offs_n_curr, M_Q, N_CTX, EVEN_M, EVEN_N)
            if EVEN_M:
                m = tl.load(m_ptrs)
                Di = tl.load(D_ptrs)
                do = tl.load(do_ptrs)
            else:
                m = tl.load(m_ptrs, mask=offs_m_curr < M_Q, other=0)
                Di = tl.load(D_ptrs, mask=offs_m_curr < M_Q, other=0)
                do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < M_Q, other=0)
            do = do.to(tl.float16)
            # compute dv
            p = tl.exp(qk - m[:, None])
            dv += tl.dot(p.to(tl.float16), do, trans_a=True)
            # compute dp = dot(v, do)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v, trans_b=True)
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(ds.to(tl.float16), q, trans_a=True)
            # # compute dq
            if EVEN_M:
                dq = tl.load(dq_ptrs, eviction_policy='evict_last')
            else:
                dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < M_Q, other=0,
                             eviction_policy='evict_last')
            dq += tl.dot(ds.to(tl.float16), k)
            # TODO: could atomic add be faster?
            if EVEN_M:
                tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            else:
                tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < M_Q,
                         eviction_policy='evict_last')
            # # increment pointers
            q_ptrs += BLOCK_M * stride_qm
            dq_ptrs += BLOCK_M * stride_dqm
            do_ptrs += BLOCK_M * stride_dom
            m_ptrs += BLOCK_M * stride_mm
            D_ptrs += BLOCK_M * stride_dm
        
        # write-back
        offs_d = tl.arange(0, BLOCK_DMODEL)
        dv_ptrs = DV + (offs_n_curr[:, None] * stride_dvn + offs_d[None, :] * stride_dvk)
        dk_ptrs = DK + (offs_n_curr[:, None] * stride_dkn + offs_d[None, :] * stride_dkk)
        if EVEN_N:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_n_curr[:, None] < N_CTX)
            tl.store(dk_ptrs, dk, mask=offs_n_curr[:, None] < N_CTX)

cached_slopes = {}

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=True, use_alibi=True, dropout=0.):
        BLOCK = 128
        # shape constraints
        qz, qh, qm, qd = q.shape
        kz, kh, kn, kd = k.shape
        vz, vh, vn, vd = v.shape
        assert qz == kz == vz
        assert qh == kh == vh
        assert kn == vn
        
        assert qd == kd == vd
        assert qd in {16, 32, 64, 128}
        
        # dims must be power of 2
        assert qd % 2 == 0
        
        if qh in cached_slopes:
            slopes = cached_slopes[qh]
        else:
            ratio = 2 ** (-2 ** -(math.log2(qh) - 3))
            slopes = ratio ** torch.arange(1, qh + 1, device=q.device)
            cached_slopes[qh] = slopes.half()
        
        o = torch.empty_like(q)
        tmp = torch.empty((qz, qh, max(triton.next_power_of_2(qm), BLOCK)), device=q.device, dtype=torch.float32)
        L = torch.empty((qz, qh, qm), device=q.device, dtype=torch.float32)
        m = torch.empty((qz, qh, qm), device=q.device, dtype=torch.float32)
        num_warps = 4 if kd <= 64 else 8
        grid = (triton.cdiv(qm, BLOCK), qh, qz)
        _fwd_kernel[grid](
                q, k, v, slopes, o, sm_scale,
                tmp, L, m,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                tmp.stride(0), tmp.stride(1), tmp.stride(2),
                L.stride(0), L.stride(1), L.stride(2),
                m.stride(0), m.stride(1), m.stride(2),
                qm, kn,
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=kd,
                EVEN_M=qm % BLOCK == 0, EVEN_N=kn % BLOCK == 0,
                CAUSAL=causal, USE_ALIBI=use_alibi,
                num_warps=num_warps, num_stages=1,
        )
        
        ctx.save_for_backward(q, k, v, slopes, o, L, m)
        ctx.BLOCK = BLOCK
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = kd
        ctx.causal = causal
        ctx.use_alibi = use_alibi
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, slopes, o, l, m = ctx.saved_tensors
        qz, qh, qm, qd = q.shape
        kz, kh, kn, kd = k.shape
        vz, vh, vn, vd = v.shape
        dq = torch.zeros_like(q, dtype=do.dtype)
        dk = torch.empty_like(k, dtype=do.dtype)
        dv = torch.empty_like(v, dtype=do.dtype)
        dos = torch.empty_like(do)
        delta = torch.empty_like(l)
        
        q_, k_, v_ = q, k, v
        if qm % ctx.BLOCK != 0:
            q_ = F.pad(q, (0, 0, 0, max(triton.next_power_of_2(qm), ctx.BLOCK)))
        if kn % ctx.BLOCK != 0:
            k_ = F.pad(k, (0, 0, 0, max(triton.next_power_of_2(kn), ctx.BLOCK)))
            v_ = F.pad(v, (0, 0, 0, max(triton.next_power_of_2(vn), ctx.BLOCK)))
        
        _bwd_preprocess[(triton.cdiv(qm, ctx.BLOCK), qh, qz)](
                o, do, dos, l, delta, qm,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dos.stride(0), dos.stride(1), dos.stride(2), dos.stride(3),
                l.stride(0), l.stride(1), l.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, BLOCK_M=ctx.BLOCK,
                EVEN_M=qm % ctx.BLOCK == 0
        )
        _bwd_kernel[(qh, qz)](
                q_, k_, v_, slopes, ctx.sm_scale,
                dos, dq, dk, dv, m, delta,
                q_.stride(0), q_.stride(1), q_.stride(2), q_.stride(3),
                k_.stride(0), k_.stride(1), k_.stride(2), k_.stride(3),
                v_.stride(0), v_.stride(1), v_.stride(2), v_.stride(3),
                dos.stride(0), dos.stride(1), dos.stride(2), dos.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                m.stride(0), m.stride(1), m.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                qm, kn, BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                BLOCK_M=ctx.BLOCK, BLOCK_N=ctx.BLOCK,
                EVEN_M=qm % ctx.BLOCK == 0, EVEN_N=kn % ctx.BLOCK == 0,
                CAUSAL=ctx.causal, USE_ALIBI=ctx.use_alibi,
                num_warps=8, num_stages=1,
        )
        return dq, dk, dv, None, None, None, None

flash_attention = _attention.apply

def ref_alibi(i, j, H):
    from einops import rearrange
    m = torch.arange(i, device='cuda', dtype=torch.int32)
    n = torch.arange(j, device='cuda', dtype=torch.int32)
    bias = rearrange(n, 'j -> 1 1 j') - rearrange(m, 'i -> 1 i 1')
    bias = -torch.abs(bias + (i - j))  # use symmetric alibi
    ratio = (2 ** (-2 ** -(math.log2(H) - 3)))
    slopes = ratio ** torch.arange(1, H + 1, device='cuda')
    slopes = rearrange(slopes, 'h -> h 1 1')
    return slopes * bias

def ref_attention(q, k, v, sm_scale, causal=True, use_alibi=True):
    H = q.shape[1]
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    i, j = p.shape[-2:]
    if use_alibi:
        p = p + ref_alibi(i, j, H)
    if causal:
        mask = torch.ones((i, j), dtype=torch.bool, device=q.device).triu(j - i + 1)
        p = p.masked_fill(mask, float('-inf'))
    
    p = torch.softmax(p.float(), dim=-1).to(v.dtype)
    return torch.matmul(p, v)

def test_op(Z, H, M_Q, N_CTX, D_HEAD, causal=True, use_alibi=True, dtype=torch.float16, grad_dtype=None,
            forward_only=False):
    grad_dtype = dtype if grad_dtype is None else grad_dtype
    torch.manual_seed(20)
    q = torch.empty((Z, H, M_Q, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5).requires_grad_()
    sm_scale = 0.3
    # reference implementation
    ref_out = ref_attention(q, k, v, sm_scale, causal, use_alibi)
    # triton implementation
    tri_out = flash_attention(q, k, v, sm_scale, causal, use_alibi)
    triton.testing.assert_almost_equal(ref_out, tri_out)
    
    if not forward_only:
        do = torch.randn_like(ref_out, dtype=grad_dtype)
        ref_out.backward(do)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
        
        tri_out.backward(do)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None
        # compare
        triton.testing.assert_almost_equal(ref_dv, tri_dv)
        triton.testing.assert_almost_equal(ref_dk, tri_dk)
        triton.testing.assert_almost_equal(ref_dq, tri_dq)

def test_fwd(q, k, v, causal=True, use_alibi=True):
    sm_scale = 0.3
    flash_attention(q, k, v, sm_scale, causal, use_alibi)
    torch.cuda.synchronize()

def test_bwd(q, k, v, do, causal=True, use_alibi=True):
    o = flash_attention(q, k, v)
    o.backward()

if __name__ == '__main__':
    # backward requires n = m to be power of 2
    test_op(1, 8, 1024, 1024, 64, True, False)
    test_op(1, 8, 1024, 1024, 64, True, True)
    
    test_op(1, 8, 8, 8, 64, True, False)
    test_op(1, 8, 8, 8, 64, True, True)
    
    # all of these cases only work for the forward pass atm
    # test_op(1, 8, 250, 250, 64, True, False, forward_only=True)
    # test_op(1, 8, 250, 250, 64, True, True, forward_only=True)
    # 
    # test_op(1, 8, 250, 250, 64, False, False, forward_only=True)
    # test_op(1, 8, 250, 250, 64, False, True, forward_only=True)
    # 
    # test_op(1, 8, 1, 256, 64, False, False, forward_only=True)
    # test_op(1, 8, 1, 256, 64, False, True, forward_only=True)
    # 
    test_op(1, 8, 250, 250, 64, True, False)
    test_op(1, 8, 250, 250, 64, True, False)
    test_op(1, 8, 250, 250, 64, False, True)
    
    # test_op(1, 8, 256, 256, 64, False, True)
    # test_op(1, 8, 10, 250, 64, True, False)
    # test_op(1, 8, 10, 250, 64, True, True)
    # test_op(1, 8, 10, 250, 64, False, True)
    # test_op(1, 8, 250, 64, False, True)
    # test_op(1, 8, 250, 64, False, False)
    
    # test_vector_query(1, 8, 256, 128, True, True)
    # test_vector_query(1, 8, 256, 128, True, False)
