// libmla_rope2.so: ctypes ABI of overlay_mlarope2/mcore/fused_mla_rope2.py (MLPERF_MLA_ROPE2=1).
// Reconstructed on rhea 2026-11-11; kernels from mla_rope_mxfp8_kernels.cuh (TE port of the original kernels).
//   mla2_q_rope_mxfp8(q, cos, sin, S, B, H, q_rd, q_cd, q_rs, q_cs, stream)
//   mla2_kv_rope_mxfp8(kv, k_pos_emb, row_stride(int64, elements), cos, sin, S, B, H,
//                      k_rd, k_cd, k_rs, k_cs, v_rd, v_cd, v_rs, v_cs, stream)
//   mla2_q_rope_bwd(dq, cos, sin, S, B, H, stream)                     (in place)
// Data SBHD; scales BHSD-permuted + GEMM-swizzled (row-wise [B*H*S, 8|4], column-wise [B*H*S/32, 256|128]).
// Returns 0 on success, a negative value on a bad argument, or a cudaError_t.
#include <algorithm>
#include "mla_rope_mxfp8_kernels.cuh"

using namespace mla_mxfp8;

namespace {

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

template <bool kKV>
int launch_rope(const RopeParams &p, cudaStream_t stream) {
  using L = RopeLayout<kKV>;
  const size_t num_tiles =
      static_cast<size_t>(p.num_columns / kHeadsPerTile) * (p.seq_len / kTileTokens);
  if (num_tiles == 0) return 0;
  auto kernel = fused_mla_rope_mxfp8_attn_kernel<kKV>;
  cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kBytes);
  if (err != cudaSuccess) return err;
  int ctas_per_sm = 0, dev = 0, sms = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, kThreads, L::kBytes);
  if (err != cudaSuccess) return err;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const size_t grid = std::min(static_cast<size_t>(sms) * std::max(ctas_per_sm, 1), num_tiles);
  kernel<<<grid, kThreads, L::kBytes, stream>>>(p);
  return cudaGetLastError();
}

bool shape_ok(int S, int B, int H) {
  return S > 0 && B > 0 && H > 0 && S % kTileTokens == 0 && H % kHeadsPerTile == 0 &&
         static_cast<size_t>(S) * B * H * kKVDim * 2 < (static_cast<size_t>(1) << 31);
}

}  // namespace

extern "C" {

int mla2_q_rope_mxfp8(const void *q, const void *cos, const void *sin, int S, int B, int H, void *rd,
                      void *cd, void *rs, void *cs, cudaStream_t stream) {
  if (!shape_ok(S, B, H)) return -1;
  if (!aligned16(q) || !aligned16(cos) || !aligned16(sin) || !aligned16(rd) || !aligned16(cd) ||
      !aligned16(rs) || !aligned16(cs))
    return -2;
  RopeParams p{};
  p.src = static_cast<const uint8_t *>(q);
  p.cos = static_cast<const uint16_t *>(cos);
  p.sin = static_cast<const uint16_t *>(sin);
  p.out0 = AttnOutput{static_cast<uint8_t *>(rd), static_cast<uint8_t *>(cd), static_cast<uint8_t *>(rs),
                      static_cast<uint8_t *>(cs)};
  p.out1 = p.out0;
  p.num_columns = static_cast<uint32_t>(B * H);
  p.heads = static_cast<uint32_t>(H);
  p.seq_len = static_cast<uint32_t>(S);
  return launch_rope<false>(p, stream);
}

int mla2_kv_rope_mxfp8(const void *kv, const void *k_pos_emb, int64_t row_stride, const void *cos,
                       const void *sin, int S, int B, int H, void *k_rd, void *k_cd, void *k_rs, void *k_cs,
                       void *v_rd, void *v_cd, void *v_rs, void *v_cs, cudaStream_t stream) {
  if (!shape_ok(S, B, H)) return -1;
  if (row_stride < kRopeDim || row_stride % 8 != 0 ||
      static_cast<size_t>(row_stride) * 2 * S * B >= (static_cast<size_t>(1) << 31))
    return -3;
  const void *ptrs[] = {kv, k_pos_emb, cos, sin, k_rd, k_cd, k_rs, k_cs, v_rd, v_cd, v_rs, v_cs};
  for (const void *x : ptrs)
    if (!aligned16(x)) return -2;
  RopeParams p{};
  p.src = static_cast<const uint8_t *>(kv);
  p.k_rope = static_cast<const uint8_t *>(k_pos_emb);
  p.k_rope_row_bytes = static_cast<uint32_t>(row_stride) * 2;
  p.cos = static_cast<const uint16_t *>(cos);
  p.sin = static_cast<const uint16_t *>(sin);
  p.out0 = AttnOutput{static_cast<uint8_t *>(k_rd), static_cast<uint8_t *>(k_cd), static_cast<uint8_t *>(k_rs),
                      static_cast<uint8_t *>(k_cs)};
  p.out1 = AttnOutput{static_cast<uint8_t *>(v_rd), static_cast<uint8_t *>(v_cd), static_cast<uint8_t *>(v_rs),
                      static_cast<uint8_t *>(v_cs)};
  p.num_columns = static_cast<uint32_t>(B * H);
  p.heads = static_cast<uint32_t>(H);
  p.seq_len = static_cast<uint32_t>(S);
  return launch_rope<true>(p, stream);
}

int mla2_q_rope_bwd(void *dq, const void *cos, const void *sin, int S, int B, int H, cudaStream_t stream) {
  if (S <= 0 || B <= 0 || H <= 0) return -1;
  if (!aligned16(dq) || !aligned16(cos) || !aligned16(sin)) return -2;
  const size_t rows = static_cast<size_t>(S) * B * H;
  if (rows * kQKDim * 2 >= (static_cast<size_t>(1) << 32)) return -1;
  constexpr size_t kRowsPerCTA = kQBwdThreads / 32 * 4 * kQBwdRowsPerLane;
  const size_t grid = (rows + kRowsPerCTA - 1) / kRowsPerCTA;
  fused_mla_q_rope_bwd_kernel<<<grid, kQBwdThreads, 0, stream>>>(
      static_cast<uint16_t *>(dq), static_cast<const uint16_t *>(cos), static_cast<const uint16_t *>(sin),
      static_cast<uint32_t>(rows), static_cast<uint32_t>(B * H));
  return cudaGetLastError();
}

}  // extern "C"
