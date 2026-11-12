# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MLA query / key / value RoPE + MXFP8 quantization for MXFP8 attention, v2 kernels (MLPERF_MLA_ROPE2=1).

Drop-in for the fused path of multi_latent_attention.py
    query = _QuantizeQueryForFusedAttn(fused_apply_mla_rope_for_q(q, ...))
    key, value = _FusedKVRoPEQuantForFusedAttn(kv, k_pos_emb, ...)
    mxfp8_transpose_swizzle([query, key, value], "sbhd")
with two CUDA kernels (libmla_rope2.so next to this file, or $MLPERF_MLA_ROPE2_LIB) that read q and kv once and
write the MXFP8 data and the BHSD-permuted, GEMM-swizzled scales directly. Outputs are byte-identical. Backward:
the query RoPE backward is a v2 kernel with the same BF16 ops as Megatron's (byte-identical, ~15 vs 18 us); the
key/value backward is Megatron's KV-split RoPE backward, as before.
"""

import ctypes
import os
from types import SimpleNamespace

import torch

ENABLED = os.getenv("MLPERF_MLA_ROPE2", "0") == "1"

_lib = None
if ENABLED:
    _path = os.getenv("MLPERF_MLA_ROPE2_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "libmla_rope2.so"))
    try:
        _lib = ctypes.CDLL(_path)
        _lib.mla2_q_rope_mxfp8.restype = ctypes.c_int
        _lib.mla2_kv_rope_mxfp8.restype = ctypes.c_int
        _lib.mla2_q_rope_bwd.restype = ctypes.c_int
    except OSError as e:  # pragma: no cover
        if int(os.getenv("RANK", "0")) == 0:
            print(f"[mla_rope2] cannot load {_path}: {e}; using the TE / Megatron path")
        _lib = None

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
    from megatron.core.fusions.fused_mla_yarn_rope_apply import _FusedMLARoPEInplace, _FusedMLARoPEKVSplit
except ImportError:  # pragma: no cover
    _lib = None


def available() -> bool:
    """Whether the v2 kernels are enabled and loaded."""
    return _lib is not None


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _aligned(*ts):
    return all(t.data_ptr() % 16 == 0 for t in ts)


def _row_stride(k_pos_emb, B):
    """Elements between k_pos_emb rows (t * B + b); [s, b, 1, d] or [s, b, d]."""
    return k_pos_emb.stride(1) if B > 1 else k_pos_emb.stride(0)


_reported = set()


def _report(msg):
    """Print once per message on global rank 0 (whether the v2 path is taken)."""
    if msg not in _reported and int(os.getenv("RANK", "0")) == 0:
        _reported.add(msg)
        print(f"[mla_rope2] {msg}", flush=True)


def supported(q, kv, k_pos_emb, cos, sin) -> bool:
    """Shapes / layouts the kernels handle (DeepSeek-V3 MLA heads, sbhd, BF16, S % 128 == 0, H % 4 == 0)."""
    ok = _supported(q, kv, k_pos_emb, cos, sin)
    if _lib is not None:
        _report("v2 q/kv RoPE + MXFP8 kernels active" if ok else
                f"unsupported inputs, using the TE path: q {None if q is None else (tuple(q.shape), q.dtype)} "
                f"kv {tuple(kv.shape), kv.dtype} k_pos_emb {tuple(k_pos_emb.shape), k_pos_emb.stride()} "
                f"cos {tuple(cos.shape), cos.dtype}")
    return ok


def _supported(q, kv, k_pos_emb, cos, sin) -> bool:
    if _lib is None:
        return False
    if q is not None and (q.dim() != 4 or q.shape[-1] != 192 or q.dtype != torch.bfloat16 or not q.is_contiguous()):
        return False
    if kv.dim() != 4 or kv.shape[-1] != 256 or kv.dtype != torch.bfloat16 or not kv.is_contiguous():
        return False
    S, B, H, _ = kv.shape
    if q is not None and tuple(q.shape[:3]) != (S, B, H):
        return False
    if S % 128 != 0 or H % 4 != 0:
        return False
    if k_pos_emb.dtype != torch.bfloat16 or k_pos_emb.shape[-1] != 64 or k_pos_emb.stride(-1) != 1:
        return False
    if k_pos_emb.numel() != S * B * 64:
        return False
    if k_pos_emb.dim() not in (3, 4):
        return False
    if _row_stride(k_pos_emb, B) % 8 != 0 or (B > 1 and k_pos_emb.stride(0) != B * k_pos_emb.stride(1)):
        return False
    for t in (cos, sin):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.shape[-1] != 64 or t.numel() < S * 64:
            return False
    return _aligned(kv, k_pos_emb, cos, sin) and (q is None or _aligned(q))


def _quantizer():
    return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)


def _empty(S, B, H, D, device):
    C = B * H
    kp = 8 if D == 192 else 4
    np_ = 256 if D == 192 else 128
    return (torch.empty(S, C * D, dtype=torch.uint8, device=device),
            torch.empty(S, C * D, dtype=torch.uint8, device=device),
            torch.empty(C * S, kp, dtype=torch.uint8, device=device),
            torch.empty(C * S // 32, np_, dtype=torch.uint8, device=device))


def _wrap(bufs, shape, dtype):
    rd, cd, rs, cs = bufs
    quantizer = _quantizer()
    return MXFP8Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=rd.view(shape),
        rowwise_scale_inv=rs,
        columnwise_data=cd.view(shape),
        columnwise_scale_inv=cs,
        quantizer=quantizer,
        requires_grad=False,
        fp8_dtype=tex.DType.kFloat8E4M3,
        with_gemm_swizzled_scales=True,
    )


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class QRoPEQuant(torch.autograd.Function):
    """MXFP8 (swizzled scales) of Megatron's in-place MLA query RoPE; backward = Megatron's RoPE backward."""

    @staticmethod
    def forward(ctx, q, cos, sin, nope_dim, emb_dim):
        """q: [s, b, h, 192] BF16."""
        S, B, H, D = q.shape
        bufs = _empty(S, B, H, D, q.device)
        rc = _lib.mla2_q_rope_mxfp8(_ptr(q), _ptr(cos), _ptr(sin), S, B, H, *[_ptr(b) for b in bufs], _stream())
        if rc != 0:
            raise RuntimeError(f"mla2_q_rope_mxfp8 failed ({rc})")
        ctx.save_for_backward(cos, sin)
        ctx.nope_dim, ctx.emb_dim = nope_dim, emb_dim
        return _wrap(bufs, q.shape, q.dtype)

    @staticmethod
    def backward(ctx, dq):
        """Attention returns a BF16 query gradient; same as _FusedMLARoPEInplace.backward (in place, same BF16 ops:
        mul, neg, fma.rn), with the v2 kernel when the layout allows."""
        cos, sin = ctx.saved_tensors
        if dq.dtype == torch.bfloat16 and dq.dim() == 4 and dq.shape[-1] == 192 and ctx.emb_dim == 64:
            dq = dq.contiguous()
            if dq.data_ptr() % 16 == 0:
                S, B, H, _ = dq.shape
                rc = _lib.mla2_q_rope_bwd(_ptr(dq), _ptr(cos), _ptr(sin), S, B, H, _stream())
                if rc != 0:
                    raise RuntimeError(f"mla2_q_rope_bwd failed ({rc})")
                return dq, None, None, None, None
        rctx = SimpleNamespace(saved_tensors=(cos, sin), has_position_ids=False, nope_dim=ctx.nope_dim,
                               emb_dim=ctx.emb_dim, cu_seqlens_q=None, rotary_interleaved=False, inverse=False,
                               remove_interleaving=False, cp_rank=0, cp_size=1)
        return _FusedMLARoPEInplace.backward(rctx, dq)[0], None, None, None, None


class KVRoPEQuant(torch.autograd.Function):
    """MXFP8 (swizzled scales) key / value of TE's fused_mla_kv_rope_mxfp8; backward = Megatron's KV-split RoPE
    backward (as _FusedKVRoPEQuantForFusedAttn)."""

    @staticmethod
    def forward(ctx, kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim):
        """kv: [s, b, h, 256] BF16, k_pos_emb: [s, b, 1, 64] (may be a strided view)."""
        S, B, H, _ = kv.shape
        kbufs = _empty(S, B, H, 192, kv.device)
        vbufs = _empty(S, B, H, 128, kv.device)
        row_stride = _row_stride(k_pos_emb, B)
        rc = _lib.mla2_kv_rope_mxfp8(_ptr(kv), _ptr(k_pos_emb), ctypes.c_int64(row_stride), _ptr(cos), _ptr(sin),
                                     S, B, H, *[_ptr(b) for b in kbufs], *[_ptr(b) for b in vbufs], _stream())
        if rc != 0:
            raise RuntimeError(f"mla2_kv_rope_mxfp8 failed ({rc})")
        ctx.save_for_backward(cos, sin)
        ctx.emb_dim, ctx.k_dim, ctx.v_dim = emb_dim, k_dim, v_dim
        ctx.cu_seqlens_kv, ctx.cp_rank, ctx.cp_size = None, 0, 1
        ctx.remove_interleaving = False
        ctx.rotary_interleaved = False
        key = _wrap(kbufs, (S, B, H, k_dim + emb_dim), kv.dtype)
        value = _wrap(vbufs, (S, B, H, v_dim), kv.dtype)
        return key, value

    @staticmethod
    def backward(ctx, dk, dv):
        """Attention returns BF16 key and value gradients."""
        d_kv, d_emb = _FusedMLARoPEKVSplit.backward(ctx, dk, dv)[:2]
        return d_kv, d_emb, None, None, None, None, None
