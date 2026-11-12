# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MXFP8 cast-only quantizes fused into their producers (MLPERF_CASTFUSE=1; needs MLPERF_MLA_ROPE2=1).

MLA up-projection gradients. With MLPERF_MLA_ROPE2=1 the query / key-value RoPE backward produce BF16 gradients
that TE then quantizes to MXFP8 (row-wise + column-wise, then a scale swizzle) as the grad_output of
linear_q_up_proj / linear_kv_up_proj. Here the RoPE / split backward writes that MXFP8 tensor directly
(libcastfuse.so: data + GEMM-swizzled scales), and TE's LayerNormLinear backward uses it as is
(grad_output_preprocess skips quantizing a QuantizedTensorStorage). Same bytes as the unfused path.

The autograd functions take the up-projection output before the head view ([s, b, h * d]) so that the MXFP8
gradient reaches the linear without a view backward. Forward = overlay_mlarope2's QRoPEQuant / KVRoPEQuant.
d_emb (the key RoPE gradient summed over heads) stays a Triton kernel: Megatron's, minus the d_kv stores, with
the BLOCK_H / num_warps Megatron's autotuner picked for its own kernel (same reduction order, same bits).
"""

import ctypes
import os

import torch

ENABLED = os.getenv("MLPERF_CASTFUSE", "0") == "1"

_lib = None
_rope2 = None
if ENABLED:
    _path = os.getenv("MLPERF_CASTFUSE_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "libcastfuse.so"))
    try:
        _lib = ctypes.CDLL(_path)
        _lib.castfuse_q_rope_bwd_mxfp8.restype = ctypes.c_int
        _lib.castfuse_kv_split_mxfp8.restype = ctypes.c_int
    except OSError as e:  # pragma: no cover
        if int(os.getenv("RANK", "0")) == 0:
            print(f"[castfuse] cannot load {_path}: {e}; using the unfused path")
        _lib = None

try:
    import triton
    import triton.language as tl
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
    from megatron.core.fusions import fused_mla_rope2 as _rope2
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        _FusedMLARoPEKVSplit,
        _mla_rope_bwd_kv_split_kernel,
    )
except ImportError:  # pragma: no cover
    _lib = None

_reported = set()


def _report(msg):
    if msg not in _reported and int(os.getenv("RANK", "0")) == 0:
        _reported.add(msg)
        print(f"[castfuse] {msg}", flush=True)


def available() -> bool:
    """Whether the fused kernels are enabled and loaded (and overlay_mlarope2's forward kernels too)."""
    return _lib is not None and _rope2 is not None and _rope2.available()


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _mxfp8_buffers(T, N, device):
    """Data (row-wise, column-wise) [T, N] and swizzled scales [T, N / 32], [T / 32, N] (T % 128 == N % 128 == 0)."""
    return (torch.empty(T, N, dtype=torch.uint8, device=device),
            torch.empty(T, N, dtype=torch.uint8, device=device),
            torch.empty(T, N // 32, dtype=torch.uint8, device=device),
            torch.empty(T // 32, N, dtype=torch.uint8, device=device))


def _wrap(bufs, shape, dtype):
    rd, cd, rs, cs = bufs
    return MXFP8Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=rd.view(shape),
        rowwise_scale_inv=rs,
        columnwise_data=cd.view(shape),
        columnwise_scale_inv=cs,
        quantizer=MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True),
        requires_grad=False,
        fp8_dtype=tex.DType.kFloat8E4M3,
        with_gemm_swizzled_scales=True,
    )


def supported(q2d, kv2d, k_pos_emb, cos, sin, heads) -> bool:
    """q2d / kv2d: up-projection outputs [s, b, h * 192] / [s, b, h * 256] (BF16, contiguous)."""
    if not available():
        return False
    ok = (q2d.dim() == 3 and kv2d.dim() == 3 and q2d.is_contiguous() and kv2d.is_contiguous()
          and q2d.shape[-1] == heads * 192 and kv2d.shape[-1] == heads * 256
          and tuple(q2d.shape[:2]) == tuple(kv2d.shape[:2])
          and (q2d.shape[0] * q2d.shape[1]) % 128 == 0)
    if ok:
        S, B = q2d.shape[:2]
        ok = _rope2.supported(q2d.view(S, B, heads, 192), kv2d.view(S, B, heads, 256), k_pos_emb, cos, sin)
    _report("fused MLA up-proj grad MXFP8 kernels active" if ok else "unsupported inputs, unfused path")
    return ok


@triton.jit
def _cf_emb_kernel(
    dK,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    stride_dk_seq,
    stride_dk_nheads,
    stride_demb_seq,
    BLOCK_H: tl.constexpr,
):
    """Megatron's _mla_rope_bwd_kv_split_kernel, pid_head == 0 part only (d_emb), sbhd, no interleaving removal."""
    pid_m = tl.program_id(axis=0)
    token_idx = pid_m // batch_size
    x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
    x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
    for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
        dK_ptr = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
        mask = x_off < head_num * stride_dk_nheads
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(dK_ptr + x_left_off, mask=mask)
        x_right = tl.load(dK_ptr + x_right_off, mask=mask)
        x_left_accum += x_left
        x_right_accum += x_right
    x_left_accum = tl.sum(x_left_accum, axis=0)
    x_right_accum = tl.sum(x_right_accum, axis=0)
    x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
    x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    x_1 = x_left_accum * cos_left + x_right_accum * sin_right
    x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
    dEMB_ptr = dEMB + pid_m * stride_demb_seq
    tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_1)
    tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_2)


_emb_cfg = {}


def _megatron_emb_config(ctx, dk, dv):
    """BLOCK_H / num_warps / num_stages that Megatron's autotuner uses for its kv-split backward at these dims.
    Runs Megatron's backward once per shape (outside graph capture, as the unfused path would) to tune it."""
    key = (dk.shape[2], ctx.emb_dim, ctx.k_dim, ctx.v_dim)
    cfg = _emb_cfg.get(key)
    if cfg is None:
        _FusedMLARoPEKVSplit.backward(ctx, dk, dv)
        best = _mla_rope_bwd_kv_split_kernel.best_config
        cfg = (best.kwargs["BLOCK_H"], best.num_warps, best.num_stages)
        _emb_cfg[key] = cfg
        _report(f"d_emb kernel uses Megatron's autotuned config BLOCK_H={cfg[0]} num_warps={cfg[1]} "
                f"num_stages={cfg[2]} (heads {dk.shape[2]})")
    return cfg


def kv_emb_grad(ctx, dk, dv):
    """d_emb [s, b, 1, emb] exactly as _FusedMLARoPEKVSplit.backward (sbhd)."""
    cos, sin = ctx.saved_tensors
    S, B, H, _ = dk.shape
    block_h, num_warps, num_stages = _megatron_emb_config(ctx, dk, dv)
    dk3 = dk.view(-1, H, ctx.emb_dim + ctx.k_dim)
    d_emb = dk.new_empty(dk3.shape[0], 1, ctx.emb_dim)
    _cf_emb_kernel[(dk3.shape[0],)](
        dk3, d_emb, cos, sin, ctx.emb_dim, ctx.k_dim, H, B, dk3.stride(0), dk3.stride(1), d_emb.stride(0),
        BLOCK_H=block_h, num_warps=num_warps, num_stages=num_stages,
    )
    return d_emb.view(S, B, 1, ctx.emb_dim)


class QUpRoPEQuant(torch.autograd.Function):
    """overlay_mlarope2's QRoPEQuant on the q_up_proj output [s, b, h * 192]; backward returns the MXFP8
    (GEMM-swizzled) q_up_proj grad_output: Megatron's query RoPE backward fused with TE's MXFP8 cast."""

    @staticmethod
    def forward(ctx, q2d, cos, sin, heads, nope_dim, emb_dim):
        S, B, N = q2d.shape
        q = q2d.view(S, B, heads, N // heads)
        bufs = _rope2._empty(S, B, heads, 192, q.device)
        rc = _rope2._lib.mla2_q_rope_mxfp8(_ptr(q), _ptr(cos), _ptr(sin), S, B, heads, *[_ptr(b) for b in bufs],
                                           _stream())
        if rc != 0:
            raise RuntimeError(f"mla2_q_rope_mxfp8 failed ({rc})")
        ctx.save_for_backward(cos, sin)
        ctx.heads, ctx.nope_dim, ctx.emb_dim = heads, nope_dim, emb_dim
        return _rope2._wrap(bufs, q.shape, q.dtype)

    @staticmethod
    def backward(ctx, dq):
        cos, sin = ctx.saved_tensors
        S, B, H, D = dq.shape
        dq = dq.contiguous()
        if dq.dtype == torch.bfloat16 and D == 192 and ctx.emb_dim == 64 and dq.data_ptr() % 16 == 0:
            bufs = _mxfp8_buffers(S * B, H * D, dq.device)
            rc = _lib.castfuse_q_rope_bwd_mxfp8(_ptr(dq), _ptr(cos), _ptr(sin), S, B, H,
                                                *[_ptr(b) for b in bufs], _stream())
            if rc != 0:
                raise RuntimeError(f"castfuse_q_rope_bwd_mxfp8 failed ({rc})")
            return _wrap(bufs, (S, B, H * D), dq.dtype), None, None, None, None, None
        dq = _rope2.QRoPEQuant.backward(ctx, dq)[0]
        return dq.view(S, B, H * D), None, None, None, None, None


class KVUpRoPEQuant(torch.autograd.Function):
    """overlay_mlarope2's KVRoPEQuant on the kv_up_proj output [s, b, h * 256]; backward returns the MXFP8
    (GEMM-swizzled) kv_up_proj grad_output (Megatron's kv split backward fused with TE's MXFP8 cast) and d_emb."""

    @staticmethod
    def forward(ctx, kv2d, k_pos_emb, cos, sin, heads, emb_dim, k_dim, v_dim):
        S, B, N = kv2d.shape
        kv = kv2d.view(S, B, heads, N // heads)
        ctx.heads = heads
        key, value = _rope2.KVRoPEQuant.forward(ctx, kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim)
        return key, value

    @staticmethod
    def backward(ctx, dk, dv):
        S, B, H, _ = dk.shape
        dk = dk.contiguous()
        dv = dv.contiguous()
        if (dk.dtype == torch.bfloat16 and dv.dtype == torch.bfloat16 and ctx.k_dim == 128 and ctx.v_dim == 128
                and ctx.emb_dim == 64 and dk.data_ptr() % 16 == 0 and dv.data_ptr() % 16 == 0):
            d_emb = kv_emb_grad(ctx, dk, dv)
            bufs = _mxfp8_buffers(S * B, H * 256, dk.device)
            rc = _lib.castfuse_kv_split_mxfp8(_ptr(dk), _ptr(dv), S, B, H, *[_ptr(b) for b in bufs], _stream())
            if rc != 0:
                raise RuntimeError(f"castfuse_kv_split_mxfp8 failed ({rc})")
            return _wrap(bufs, (S, B, H * 256), dk.dtype), d_emb, None, None, None, None, None, None
        d_kv, d_emb = _FusedMLARoPEKVSplit.backward(ctx, dk, dv)[:2]
        return d_kv.view(S, B, -1), d_emb, None, None, None, None, None, None
