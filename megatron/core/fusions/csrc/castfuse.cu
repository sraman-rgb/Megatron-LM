// libcastfuse.so: ctypes ABI of overlay_castfuse/mcore/fused_castfuse.py (MLPERF_CASTFUSE=1).
// Reconstructed on rhea 2026-11-11; kernels from mla_rope_mxfp8_kernels.cuh (TE port of the original kernels).
//   castfuse_q_rope_bwd_mxfp8(dq [S,B,H,192], cos, sin, S, B, H, rd, cd, rs, cs, stream)
//       Megatron's MLA query RoPE backward + MXFP8 of the result as [S*B, H*192]
//   castfuse_kv_split_mxfp8(dk [S,B,H,192], dv [S,B,H,128], S, B, H, rd, cd, rs, cs, stream)
//       d_kv = [dk[..., :128] | dv] per head, MXFP8 as [S*B, H*256]
// Outputs: row-/column-wise E4M3 data [T, N], GEMM-swizzled scales [T, N/32] and [T/32, N].
// Returns 0 on success, a negative value on a bad argument, or a cudaError_t.
#include "mla_rope_mxfp8_kernels.cuh"

using namespace mla_mxfp8;

namespace {

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

bool grad_shape_ok(int S, int B, int H) {
  const size_t T = static_cast<size_t>(S) * B;
  return S > 0 && B > 0 && H > 0 && T % kTileTokens == 0 && H % 2 == 0 &&
         T * H * kKVDim < (static_cast<size_t>(1) << 31) && T / kTileTokens <= 65535;
}

template <bool kKV>
int launch(const GradParams &p, cudaStream_t stream) {
  if (p.tokens == 0) return 0;
  const dim3 grid(p.columns / 128, p.tokens / kTileTokens);
  fused_mla_rope_bwd_mxfp8_kernel<kKV><<<grid, kThreads, 0, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace

extern "C" {

int castfuse_q_rope_bwd_mxfp8(const void *dq, const void *cos, const void *sin, int S, int B, int H, void *rd,
                              void *cd, void *rs, void *cs, cudaStream_t stream) {
  if (!grad_shape_ok(S, B, H)) return -1;
  const void *ptrs[] = {dq, cos, sin, rd, cd, rs, cs};
  for (const void *x : ptrs)
    if (!aligned16(x)) return -2;
  GradParams p{};
  p.a = static_cast<const uint8_t *>(dq);
  p.cos = static_cast<const uint16_t *>(cos);
  p.sin = static_cast<const uint16_t *>(sin);
  p.out = GradOutput{static_cast<uint8_t *>(rd), static_cast<uint8_t *>(cd), static_cast<uint8_t *>(rs),
                     static_cast<uint8_t *>(cs)};
  p.tokens = static_cast<uint32_t>(S * B);
  p.columns = static_cast<uint32_t>(H * kQKDim);
  p.heads = static_cast<uint32_t>(H);
  p.batch = static_cast<uint32_t>(B);
  return launch<false>(p, stream);
}

int castfuse_kv_split_mxfp8(const void *dk, const void *dv, int S, int B, int H, void *rd, void *cd, void *rs,
                            void *cs, cudaStream_t stream) {
  if (!grad_shape_ok(S, B, H)) return -1;
  const void *ptrs[] = {dk, dv, rd, cd, rs, cs};
  for (const void *x : ptrs)
    if (!aligned16(x)) return -2;
  GradParams p{};
  p.a = static_cast<const uint8_t *>(dk);
  p.b = static_cast<const uint8_t *>(dv);
  p.out = GradOutput{static_cast<uint8_t *>(rd), static_cast<uint8_t *>(cd), static_cast<uint8_t *>(rs),
                     static_cast<uint8_t *>(cs)};
  p.tokens = static_cast<uint32_t>(S * B);
  p.columns = static_cast<uint32_t>(H * kKVDim);
  p.heads = static_cast<uint32_t>(H);
  p.batch = static_cast<uint32_t>(B);
  return launch<true>(p, stream);
}

}  // extern "C"
