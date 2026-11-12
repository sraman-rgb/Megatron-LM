# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fused DeepSeek-V3 router for 256 experts (MLPERF_FUSED_TOPK256=1).

One CUDA kernel (libtopk256.so next to this file, or $MLPERF_FUSED_TOPK256_LIB) replaces, bit for bit,

  forward:  TE fused_topk_with_score_function (sigmoid, expert bias, 8 groups / top-4 groups, top-8, scaling)
            + routing_map.sum(0) for the expert-bias token counts
            + torch.topk(probs, 8) of the flex / NCCL-EP dispatcher (token_probs, token_indices, same order)
            + TE fused_compute_score_for_moe_aux_loss + routing_map.sum(0) for the sequence aux loss
            (the aux loss itself still runs TE's fused_moe_aux_loss forward kernel on the same scores);
  backward: TE fused_moe_aux_loss bwd + score-for-aux-loss bwd + top-k bwd + the bf16 add of the two logit
            gradients (and the dispatcher's top-k zero/scatter backward), in one kernel.

The router returns the dispatcher's (token_probs [T, 8] bf16, token_indices [T, 8] int64) directly; the dense
probs / routing_map are never materialised.
"""

import ctypes
import os

import torch

ENABLED = os.getenv("MLPERF_FUSED_TOPK256", "0") == "1"
SUPPORTED_EXPERTS = (256, 16)  # 16: the 4-node proxy model
TOPK = 8
_TOK_PER_BLOCK = int(os.getenv("MLPERF_FUSED_TOPK256_TPB", "801"))  # 100 * warps + tokens per warp (v2); -n: v1 kernel, n tokens per block
_ALLREDUCE_UP = int(os.getenv("MLPERF_FUSED_TOPK256_ALLREDUCE_UP", "0"))

_lib = None
# overlay_casts (MLPERF_CASTS_TOPK_F32=1): token_probs are produced in fp32 (bf16-rounded values, i.e. exactly what
# the NCCL-EP dispatcher's token_probs.float() made of the bf16 probs) by libtopk256_casts.so, and the backward
# rounds the fp32 gradient to bf16 inside the kernel (exactly what ToCopyBackward did). Removes the per-layer
# [T, 8] bf16->fp32 cast before dispatch and the fp32->bf16 cast of its gradient; bit-identical.
_PROBS_F32 = os.getenv("MLPERF_CASTS_TOPK_F32", "0") == "1"
_DEBUG = os.getenv("MLPERF_FUSED_TOPK256_DEBUG", "0") == "1"
_debug_last = []


def _load():
    global _lib
    if _lib is not None:
        return _lib
    path = os.getenv(
        "MLPERF_FUSED_TOPK256_LIB",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "libtopk256_casts.so" if _PROBS_F32 else "libtopk256.so",
        ),
    )
    try:
        lib = ctypes.CDLL(path)
        lib.topk256_fwd.restype = ctypes.c_int
        lib.topk256_bwd.restype = ctypes.c_int
        if _PROBS_F32:
            lib.topk256_fwd_f32.restype = ctypes.c_int
            lib.topk256_bwd_f32.restype = ctypes.c_int
            if int(os.getenv("RANK", "0")) == 0:
                print(f"[fused_topk256] MLPERF_CASTS_TOPK_F32: fp32 token_probs from {path}")
        _lib = lib
    except OSError as e:  # pragma: no cover
        if int(os.getenv("RANK", "0")) == 0:
            print(f"[fused_topk256] cannot load {path}: {e}; using the TE path")
        _lib = False
    return _lib


def available() -> bool:
    """Whether the fused kernels are enabled and loaded."""
    return ENABLED and bool(_load())


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p(0)


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def supported(logits, expert_bias) -> bool:
    """Shape / dtype / alignment check for the fast path."""
    return (
        logits.dim() == 2
        and logits.shape[1] in SUPPORTED_EXPERTS
        and logits.dtype == torch.bfloat16
        and logits.is_cuda
        and logits.is_contiguous()
        and logits.data_ptr() % 16 == 0
        and (expert_bias is None or (expert_bias.dtype == torch.float32 and expert_bias.is_contiguous()))
    )


class FusedTopk256(torch.autograd.Function):
    """logits -> (token_probs, token_indices, seq aux loss)."""

    @staticmethod
    def forward(ctx, logits, expert_bias, bias_counts, scaling_factor, aux_coeff, reduce_fn, reduce_size):
        import transformer_engine_torch as tex

        lib = _load()
        n, num_experts = logits.shape
        dev = logits.device
        token_probs = torch.empty(n, TOPK, dtype=torch.float32 if _PROBS_F32 else logits.dtype, device=dev)
        token_indices = torch.empty(n, TOPK, dtype=torch.int64, device=dev)
        routed = torch.empty(n, TOPK, dtype=torch.uint8, device=dev)
        aux_scores = torch.empty(n, num_experts, dtype=torch.float32, device=dev)
        aux_counts = torch.zeros(num_experts, dtype=torch.int64, device=dev)
        rc = (lib.topk256_fwd_f32 if _PROBS_F32 else lib.topk256_fwd)(
            _ptr(logits), _ptr(expert_bias), n, num_experts, ctypes.c_float(scaling_factor), _ptr(token_probs),
            _ptr(token_indices), _ptr(routed), _ptr(aux_scores), _ptr(aux_counts), _ptr(bias_counts),
            _TOK_PER_BLOCK, _stream(),
        )
        assert rc == 0, f"topk256_fwd failed ({rc})"
        global_counts = reduce_fn(aux_counts) if reduce_fn is not None else aux_counts
        if _DEBUG:
            _debug_last[:] = [aux_scores, global_counts]
        total_num_tokens = n * reduce_size
        # TE FusedAuxLoss.forward (int total_num_tokens path), unchanged.
        aux_loss, const_buf = tex.fused_moe_aux_loss_fwd(
            probs=aux_scores,
            tokens_per_expert=global_counts,
            total_num_tokens=int(total_num_tokens),
            num_experts=num_experts,
            num_rows=n,
            num_cols=num_experts,
            topk=TOPK,
            coeff=aux_coeff,
        )
        ctx.save_for_backward(logits, token_indices, routed, const_buf, global_counts)
        ctx.scaling_factor = scaling_factor
        ctx.mark_non_differentiable(token_indices)
        return token_probs, token_indices, aux_loss

    @staticmethod
    def backward(ctx, grad_probs, _grad_idx, grad_aux):
        logits, token_indices, routed, const_buf, global_counts = ctx.saved_tensors
        if grad_probs is None:
            grad_probs = torch.zeros(
                token_indices.shape,
                dtype=torch.float32 if _PROBS_F32 else logits.dtype,
                device=logits.device,
            )
        grad_probs = grad_probs.contiguous()
        if _PROBS_F32 and grad_probs.dtype != torch.float32:
            grad_probs = grad_probs.float()
        if grad_aux is None:
            grad_aux = torch.zeros((), dtype=torch.float32, device=logits.device)
        grad_aux = grad_aux.float().contiguous()
        grad_logits = torch.empty_like(logits)
        rc = (_load().topk256_bwd_f32 if _PROBS_F32 else _load().topk256_bwd)(
            _ptr(logits), _ptr(token_indices), _ptr(routed), _ptr(grad_probs), _ptr(const_buf),
            _ptr(global_counts), _ptr(grad_aux), logits.shape[0], logits.shape[1], ctypes.c_float(ctx.scaling_factor),
            _ptr(grad_logits), _ALLREDUCE_UP, _stream(),
        )
        assert rc == 0, f"topk256_bwd failed ({rc})"
        return grad_logits, None, None, None, None, None, None


def fused_topk256(logits, expert_bias, bias_counts, scaling_factor, aux_coeff, reduce_fn=None, reduce_size=1):
    """Returns (token_probs [T, 8], token_indices [T, 8] int64, aux_loss) -- see the module docstring.

    bias_counts (int64 [256] or None) is incremented in place by the routing-map column counts."""
    return FusedTopk256.apply(logits, expert_bias, bias_counts, scaling_factor, aux_coeff, reduce_fn, reduce_size)
