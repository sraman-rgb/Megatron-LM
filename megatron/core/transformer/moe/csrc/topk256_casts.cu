// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused DeepSeek-V3 sigmoid router (256 experts, or the 16-expert proxy model) for
// overlay_topk / overlay_casts fused_topk256.py (ctypes ABI below).
//
// REBUILT ON RHEA (2026-11-11): the original overlay_topk/lib/topk256.cu and
// overlay_casts/lib/topk256_casts.cu were not carried over (the session branch's .gitignore drops
// lib/). This file reconstructs them from
//   * the ctypes ABI and docstrings in overlay_{topk,casts}/mcore/fused_topk256.py,
//   * the TE port of the forward kernel (sraman-rgb/TransformerEngine
//     sraman/router-fused-group-topk-aux-scores, fused_topk_with_aux_loss_scores.cu), used for E=256,
//   * the generic TE 39a8b879 router kernels in the M15 container (fused_topk_forward_simple_kernel
//     with the naive top-k, fused_score_for_moe_aux_loss_forward_simple_kernel, the dense-routing-map
//     fused_topk_with_score_function_backward_kernel, fused_score_for_moe_aux_loss_backward_kernel,
//     fused_moe_aux_loss_backward_kernel), copied literally for E=16 (and for E=256 when
//     TOPK256_GENERIC=1),
//   * torch.topk (sbtopk gather in index order + SmallBitonicSort, ATen SortUtils.cuh) for the order
//     of the dispatcher's compact (token_probs, token_indices).
//
// Forward (one warp per token):
//   sigmoid; seq-aux-loss scores s / (sum s + eps) -> aux_scores [T, E] fp32 and the column counts of
//   their naive top-8 -> aux_counts (+=); group-limited (8 groups / top-4) top-8 of s + bias; probs
//   = bf16(scaling * s_sel / (sum s_sel + eps)); counts of the selected experts -> bias_counts (+=);
//   then (probs, indices) in torch.topk(dense_probs, 8) order -> token_probs [T, 8] (bf16, or
//   float(bf16) for the _f32 entry points), token_indices [T, 8] int64, routed [T, 8] uint8.
// Backward (one warp per token): TE dense top-k bwd (normalisation + sigmoid, routed experts) of
//   the bf16 probs gradient, plus fused_moe_aux_loss bwd (C * tokens_per_expert * g_aux) through the
//   aux-score bwd, each rounded to bf16 and added as autograd does.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {

using bf16 = __nv_bfloat16;
constexpr int kWarp = 32;
constexpr int kTopk = 8;
constexpr int kNumGroups = 8;
constexpr int kGroupTopk = 4;
constexpr float kEps = 1e-20f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

__device__ __forceinline__ float sigmoid_scalar(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_bwd_scalar(float grad, float y) {
  return grad * y * (1.0f - y);
}
__device__ __forceinline__ float normalize_bwd_scalar(float grad, bool routed, float sum_act,
                                                      float sum_grad_act) {
  if (!routed) return 0.0f;
  float denom = sum_act + kEps;
  return grad / denom - sum_grad_act / (denom * denom);
}
__device__ __forceinline__ float warp_allreduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

// ---- TE utils.h, literal copies -----------------------------------------------------------
__device__ inline float warp_reduce_sum_on_shmem(const float *data_ptr, int data_size, int lane_id) {
  float val = lane_id < data_size ? data_ptr[lane_id] : 0.0f;
  for (int i = lane_id + kWarp; i < data_size; i += kWarp) val += data_ptr[i];
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  __syncwarp();
  return val;
}

template <int K>
__device__ inline void naive_topk_and_mask_smallk(const float *scores, int data_size,
                                                  int *topk_indices, float *topk_scores,
                                                  int lane_id) {
  int selected[K];
#pragma unroll
  for (int i = 0; i < K; ++i) selected[i] = -1;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float val = kNegInf;
    int index = (lane_id < data_size) ? lane_id : -1;
    for (int i = lane_id; i < data_size; i += kWarp) {
      bool masked = false;
#pragma unroll
      for (int j = 0; j < k; ++j) masked |= (selected[j] == i);
      if (masked) continue;
      float cur_val = scores[i];
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    for (int s = kWarp / 2; s > 0; s /= 2) {
      auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    float chosen_val = __shfl_sync(0xffffffff, val, 0);
    int chosen_index = __shfl_sync(0xffffffff, index, 0);
    if (lane_id == 0) {
      topk_indices[k] = chosen_index;
      topk_scores[k] = chosen_val;
    }
    selected[k] = chosen_index;
    __syncwarp();
  }
}

// ---- torch.topk(dense_probs, 8) order ------------------------------------------------------
// sbtopk::gatherTopK writes the values > kth in index order, then the values == kth in index order;
// sortKeyValueInplace (k <= 32, not stable) runs SmallBitonicSort: bitonicSortKVInPlace with
// Power2SortSize slots (valid flags, invalid entries sort last) and GTOp.
template <int N>
__device__ __forceinline__ void torch_bitonic_sort(float (&key)[N], int (&val)[N], bool (&valid)[N]) {
  auto bswap = [](float &kA, int &vA, bool &validA, float &kB, int &vB, bool &validB, bool dir) {
    bool swap = ((kA > kB) && validA) || !validB;
    if (swap == dir) {
      float tk = kA; kA = kB; kB = tk;
      int tv = vA; vA = vB; vB = tv;
      bool tb = validA; validA = validB; validB = tb;
    }
  };
#pragma unroll
  for (unsigned size = 2; size < N; size *= 2) {
#pragma unroll
    for (unsigned stride = size / 2; stride > 0; stride /= 2) {
#pragma unroll
      for (unsigned tid = 0; tid < N / 2; ++tid) {
        const bool flag = (tid & (size / 2)) != 0;
        const unsigned pos = 2 * tid - (tid & (stride - 1));
        bswap(key[pos], val[pos], valid[pos], key[pos + stride], val[pos + stride],
              valid[pos + stride], flag);
      }
    }
  }
#pragma unroll
  for (unsigned stride = N / 2; stride > 0; stride /= 2) {
#pragma unroll
    for (unsigned tid = 0; tid < N / 2; ++tid) {
      const unsigned pos = 2 * tid - (tid & (stride - 1));
      bswap(key[pos], val[pos], valid[pos], key[pos + stride], val[pos + stride],
            valid[pos + stride], false);
    }
  }
}

template <typename ProbT>
__device__ __forceinline__ void store_prob(ProbT *p, float v);
template <>
__device__ __forceinline__ void store_prob<bf16>(bf16 *p, float v) {
  *p = __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ void store_prob<float>(float *p, float v) {
  *p = v;
}

// Lane 0 only. sel[k] = selected experts (selection order), pv[k] = their bf16 probs as float.
template <typename ProbT, int SortN>
__device__ void write_dispatcher_topk(const int (&sel)[kTopk], const float (&pv)[kTopk],
                                      int num_experts, size_t token, ProbT *token_probs,
                                      int64_t *token_indices, uint8_t *routed) {
  float kth = pv[0];
#pragma unroll
  for (int k = 1; k < kTopk; ++k) kth = fminf(kth, pv[k]);
  float key[SortN];
  int idx[SortN];
  bool valid[SortN];
#pragma unroll
  for (int i = 0; i < SortN; ++i) {
    key[i] = 0.0f;
    idx[i] = 0;
    valid[i] = false;
  }
  if (kth > 0.0f) {
    // Exactly the 8 selected experts are positive: gather = (> kth by index) ++ (== kth by index).
    int n = 0;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      // index-ordered walk over the 8 selected experts
      int last = -1;
      for (int c = 0; c < kTopk; ++c) {
        int best = 0x7fffffff, bk = -1;
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
          const bool take = pass == 0 ? (pv[k] > kth) : (pv[k] == kth);
          if (take && sel[k] > last && sel[k] < best) {
            best = sel[k];
            bk = k;
          }
        }
        if (bk < 0) break;
        last = best;
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
          if (k == bk) {
            key[n & (SortN - 1)] = pv[k];
            idx[n & (SortN - 1)] = sel[k];
          }
        }
        // n < 8 always; written via the masked index to keep the arrays in registers
#pragma unroll
        for (int i = 0; i < kTopk; ++i) {
          if (i == n) valid[i] = true;
        }
        ++n;
      }
    }
  } else {
    // A selected prob rounded to 0 (|logit| beyond ~87): the gather fills up with zeros in index
    // order, which can include unrouted experts. Never seen in practice; kept for completeness.
    int n = 0;
    for (int pass = 0; pass < 2 && n < kTopk; ++pass) {
      for (int e = 0; e < num_experts && n < kTopk; ++e) {
        float v = 0.0f;
        bool is_sel = false;
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
          if (sel[k] == e) {
            v = pv[k];
            is_sel = true;
          }
        }
        const bool take = pass == 0 ? (is_sel && v > 0.0f) : !(is_sel && v > 0.0f);
        if (take) {
#pragma unroll
          for (int i = 0; i < kTopk; ++i) {
            if (i == n) {
              key[i] = v;
              idx[i] = e;
              valid[i] = true;
            }
          }
          ++n;
        }
      }
    }
  }
  torch_bitonic_sort<SortN>(key, idx, valid);
#pragma unroll
  for (int k = 0; k < kTopk; ++k) {
    store_prob<ProbT>(token_probs + token * kTopk + k, key[k]);
    token_indices[token * kTopk + k] = idx[k];
    bool r = false;
#pragma unroll
    for (int j = 0; j < kTopk; ++j) r |= (sel[j] == idx[k]);
    routed[token * kTopk + k] = r ? 1 : 0;
  }
}

// ============================================================================================
// E = 256 fast forward (TE port, fused_topk_with_aux_loss_scores_forward_kernel)
// ============================================================================================
namespace fast {
constexpr int kNumExperts = 256;
constexpr int kExpertsPerLane = kNumExperts / kWarp;
constexpr int kLanesPerGroup = kNumExperts / kNumGroups / kExpertsPerLane;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ int bitrev5(int x) {
  return static_cast<int>(__brev(static_cast<unsigned>(x)) >> 27);
}
__device__ __forceinline__ int tie_rank(int e) { return (bitrev5(e & 31) << 3) | (e >> 5); }
__device__ __forceinline__ int expert_of_rank(int r) { return bitrev5(r >> 3) | ((r & 7) << 5); }
constexpr int kNoRank = 0x7fffffff;

__device__ __forceinline__ bool precedes(float va, int ra, float vb, int rb) {
  return va > vb || (va == vb && ra < rb);
}
__device__ __forceinline__ void compare_swap(float &va, int &ra, float &vb, int &rb) {
  const bool swap = precedes(vb, rb, va, ra);
  const float tv = swap ? vb : va;
  const int tr = swap ? rb : ra;
  vb = swap ? va : vb;
  rb = swap ? ra : rb;
  va = tv;
  ra = tr;
}
__device__ __forceinline__ void sort8(float (&v)[8], int (&r)[8]) {
  compare_swap(v[0], r[0], v[1], r[1]);
  compare_swap(v[2], r[2], v[3], r[3]);
  compare_swap(v[4], r[4], v[5], r[5]);
  compare_swap(v[6], r[6], v[7], r[7]);
  compare_swap(v[0], r[0], v[2], r[2]);
  compare_swap(v[1], r[1], v[3], r[3]);
  compare_swap(v[4], r[4], v[6], r[6]);
  compare_swap(v[5], r[5], v[7], r[7]);
  compare_swap(v[1], r[1], v[2], r[2]);
  compare_swap(v[5], r[5], v[6], r[6]);
  compare_swap(v[0], r[0], v[4], r[4]);
  compare_swap(v[3], r[3], v[7], r[7]);
  compare_swap(v[1], r[1], v[5], r[5]);
  compare_swap(v[2], r[2], v[6], r[6]);
  compare_swap(v[1], r[1], v[4], r[4]);
  compare_swap(v[3], r[3], v[6], r[6]);
  compare_swap(v[2], r[2], v[4], r[4]);
  compare_swap(v[3], r[3], v[5], r[5]);
  compare_swap(v[3], r[3], v[4], r[4]);
}
__device__ __forceinline__ void pop_front(float (&v)[8], int (&r)[8]) {
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    v[i] = v[i + 1];
    r[i] = r[i + 1];
  }
  v[7] = kNegInf;
  r[7] = kNoRank;
}
template <int width>
__device__ __forceinline__ void lane_group_best(float &v, int &r) {
#pragma unroll
  for (int m = 1; m < width; m <<= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, m);
    const int orank = __shfl_xor_sync(0xffffffffu, r, m);
    if (precedes(ov, orank, v, r)) {
      v = ov;
      r = orank;
    }
  }
}

template <typename ProbT, int SortN>
__global__ void __launch_bounds__(kWarpsPerBlock *kWarp)
    fwd_kernel(const bf16 *__restrict__ logits, const float *__restrict__ expert_bias,
               int num_tokens, float scaling_factor, ProbT *__restrict__ token_probs,
               int64_t *__restrict__ token_indices, uint8_t *__restrict__ routed,
               float *__restrict__ aux_scores, int64_t *__restrict__ aux_counts,
               int64_t *__restrict__ bias_counts) {
  __shared__ float scores_smem[kWarpsPerBlock][kNumExperts];
  __shared__ int aux_count_smem[kNumExperts];
  __shared__ int route_count_smem[kNumExperts];
  for (int i = threadIdx.x; i < kNumExperts; i += blockDim.x) {
    aux_count_smem[i] = 0;
    route_count_smem[i] = 0;
  }
  __syncthreads();

  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int e0 = lane * kExpertsPerLane;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  float *scores = scores_smem[warp];

  if (token < num_tokens) {
    const size_t row = static_cast<size_t>(token) * kNumExperts;
    float s[kExpertsPerLane];
    {
      const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(logits + row + e0));
      const bf16 *vals = reinterpret_cast<const bf16 *>(&raw);
#pragma unroll
      for (int i = 0; i < kExpertsPerLane; ++i) s[i] = __bfloat162float(vals[i]);
    }
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) s[i] = sigmoid_scalar(s[i]);
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) scores[e0 + i] = s[i];
    __syncwarp();

    int rank[kExpertsPerLane];
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) rank[i] = tie_rank(e0 + i);

    {
      float sum = scores[lane];
#pragma unroll
      for (int j = 1; j < kNumExperts / kWarp; ++j) sum += scores[lane + j * 32];
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      float a[kExpertsPerLane];
      int ar[kExpertsPerLane];
#pragma unroll
      for (int i = 0; i < kExpertsPerLane; ++i) {
        a[i] = s[i] / (sum + kEps);
        ar[i] = rank[i];
      }
      float4 *dst = reinterpret_cast<float4 *>(aux_scores + row + e0);
      dst[0] = make_float4(a[0], a[1], a[2], a[3]);
      dst[1] = make_float4(a[4], a[5], a[6], a[7]);
      sort8(a, ar);
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        float v = a[0];
        int r = ar[0];
        lane_group_best<kWarp>(v, r);
        if (r == ar[0]) {
          pop_front(a, ar);
          atomicAdd(&aux_count_smem[expert_of_rank(r)], 1);
        }
      }
    }

    int selected[kTopk];
    {
      float v[kExpertsPerLane];
      int vr[kExpertsPerLane];
#pragma unroll
      for (int i = 0; i < kExpertsPerLane; ++i) {
        v[i] = s[i];
        if (expert_bias != nullptr) v[i] += expert_bias[e0 + i];
        vr[i] = rank[i];
      }
      sort8(v, vr);
      {
        float top1 = v[0];
        int top1_rank = vr[0];
        lane_group_best<kLanesPerGroup>(top1, top1_rank);
        const bool own_top1 = vr[0] == top1_rank;
        float top2 = own_top1 ? v[1] : v[0];
        int top2_rank = own_top1 ? vr[1] : vr[0];
        lane_group_best<kLanesPerGroup>(top2, top2_rank);
        float group_score = 0.0f;
        group_score = group_score + top1;
        group_score = group_score + top2;
        const int group = lane / kLanesPerGroup;
        int preceding = 0;
#pragma unroll
        for (int h = 0; h < kNumGroups; ++h) {
          const float other = __shfl_sync(0xffffffffu, group_score, h * kLanesPerGroup);
          preceding += precedes(other, bitrev5(h), group_score, bitrev5(group)) ? 1 : 0;
        }
        if (preceding >= kGroupTopk) {
#pragma unroll
          for (int i = 0; i < kExpertsPerLane; ++i) {
            v[i] = kNegInf;
            vr[i] = kNoRank;
          }
        }
      }
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        float best = v[0];
        int best_rank = vr[0];
        lane_group_best<kWarp>(best, best_rank);
        const int e = expert_of_rank(best_rank);
        if (best_rank == vr[0]) {
          pop_front(v, vr);
          if (bias_counts != nullptr) atomicAdd(&route_count_smem[e], 1);
        }
        selected[k] = e;
      }
    }

    if (lane == 0) {
      float t[kTopk];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) t[k] = (scores[selected[k]] + 0.0f) + 0.0f;
      const float sum = ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]));
      float pv[kTopk];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const float normalized = scores[selected[k]] / (sum + kEps);
        pv[k] = __bfloat162float(__float2bfloat16_rn(scaling_factor * normalized));
      }
      write_dispatcher_topk<ProbT, SortN>(selected, pv, kNumExperts, token, token_probs,
                                          token_indices, routed);
    }
  }

  __syncthreads();
  for (int i = threadIdx.x; i < kNumExperts; i += blockDim.x) {
    const int aux_count = aux_count_smem[i];
    if (aux_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(aux_counts) + i,
                static_cast<unsigned long long>(aux_count));
    }
    const int route_count = route_count_smem[i];
    if (route_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(bias_counts) + i,
                static_cast<unsigned long long>(route_count));
    }
  }
}
}  // namespace fast

// ============================================================================================
// Generic forward (literal TE simple kernels), E in {16, 256}
// ============================================================================================
namespace generic {
constexpr int kWarpsPerBlock = 4;  // TE kThreadsPerBlock = 128

template <typename ProbT, int E, int SortN>
__global__ void __launch_bounds__(kWarpsPerBlock *kWarp)
    fwd_kernel(const bf16 *__restrict__ logits, const float *__restrict__ expert_bias,
               int num_tokens, float scaling_factor, ProbT *__restrict__ token_probs,
               int64_t *__restrict__ token_indices, uint8_t *__restrict__ routed,
               float *__restrict__ aux_scores, int64_t *__restrict__ aux_counts,
               int64_t *__restrict__ bias_counts) {
  __shared__ float scores_s[kWarpsPerBlock][E];
  __shared__ float sig_s[kWarpsPerBlock][E];  // intermediate_output (sigmoid)
  __shared__ float masked_s[kWarpsPerBlock][E];
  __shared__ float aux_s[kWarpsPerBlock][E];
  __shared__ float group_s[kWarpsPerBlock][kNumGroups];
  __shared__ int tidx_s[kWarpsPerBlock][kTopk];
  __shared__ float tval_s[kWarpsPerBlock][kTopk];
  __shared__ int aux_count_smem[E];
  __shared__ int route_count_smem[E];
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    aux_count_smem[i] = 0;
    route_count_smem[i] = 0;
  }
  __syncthreads();
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  float *scores = scores_s[warp];
  float *sig = sig_s[warp];
  float *masked = masked_s[warp];
  float *aux = aux_s[warp];
  float *group_scores = group_s[warp];
  int *topk_indices = tidx_s[warp];
  float *topk_scores = tval_s[warp];

  if (token < num_tokens) {
    const size_t pos = static_cast<size_t>(token) * E;
    // ---- fused_score_for_moe_aux_loss_forward_simple_kernel (sigmoid)
    for (int i = lane; i < E; i += kWarp) aux[i] = __bfloat162float(logits[pos + i]);
    __syncwarp();
    for (int i = lane; i < E; i += kWarp) aux[i] = 1.0f / (1.0f + expf(-aux[i]));
    __syncwarp();
    {
      const float sum_logits = warp_reduce_sum_on_shmem(aux, E, lane);
      for (int i = lane; i < E; i += kWarp) aux[i] /= (sum_logits + kEps);
      __syncwarp();
    }
    naive_topk_and_mask_smallk<kTopk>(aux, E, topk_indices, topk_scores, lane);
    __syncwarp();
    for (int i = lane; i < kTopk; i += kWarp) atomicAdd(&aux_count_smem[topk_indices[i]], 1);
    for (int i = lane; i < E; i += kWarp) aux_scores[pos + i] = aux[i];
    __syncwarp();

    // ---- fused_topk_forward_simple_kernel (sigmoid, bias, groups, naive top-k)
    for (int i = lane; i < E; i += kWarp) scores[i] = __bfloat162float(logits[pos + i]);
    for (int i = lane; i < E; i += kWarp) masked[i] = kNegInf;
    __syncwarp();
    for (int i = lane; i < E; i += kWarp) scores[i] = 1.0f / (1.0f + expf(-scores[i]));
    __syncwarp();
    for (int i = lane; i < E; i += kWarp) sig[i] = scores[i];
    __syncwarp();
    if (expert_bias != nullptr) {
      for (int i = lane; i < E; i += kWarp) scores[i] += expert_bias[i];
      __syncwarp();
    }
    constexpr int group_size = E / kNumGroups;
    for (int i = 0; i < kNumGroups; i++) {
      naive_topk_and_mask_smallk<kTopk / kGroupTopk>(scores + i * group_size, group_size,
                                                     topk_indices, topk_scores, lane);
      __syncwarp();
      if (lane == 0) {
        float tmp = 0.0;
        for (int j = 0; j < kTopk / kGroupTopk; j++) tmp = tmp + topk_scores[j];
        group_scores[i] = tmp;
      }
      __syncwarp();
    }
    naive_topk_and_mask_smallk<kGroupTopk>(group_scores, kNumGroups, topk_indices, topk_scores,
                                           lane);
    __syncwarp();
    for (int i = 0; i < kGroupTopk; i++) {
      int st = topk_indices[i] * group_size;
      int ed = st + group_size;
      for (int j = st + lane; j < ed; j += kWarp) masked[j] = scores[j];
    }
    __syncwarp();
    naive_topk_and_mask_smallk<kTopk>(masked, E, topk_indices, topk_scores, lane);
    __syncwarp();
    if (expert_bias != nullptr) {
      for (int i = lane; i < kTopk; i += kWarp) topk_scores[i] = sig[topk_indices[i]];
      __syncwarp();
    }
    {
      const float sum_scores = warp_reduce_sum_on_shmem(topk_scores, kTopk, lane);
      for (int i = lane; i < kTopk; i += kWarp) topk_scores[i] = topk_scores[i] / (sum_scores + kEps);
      __syncwarp();
    }
    if (bias_counts != nullptr) {
      for (int i = lane; i < kTopk; i += kWarp) atomicAdd(&route_count_smem[topk_indices[i]], 1);
    }
    if (lane == 0) {
      int sel[kTopk];
      float pv[kTopk];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        sel[k] = topk_indices[k];
        pv[k] = __bfloat162float(__float2bfloat16_rn(scaling_factor * topk_scores[k]));
      }
      write_dispatcher_topk<ProbT, SortN>(sel, pv, E, token, token_probs, token_indices, routed);
    }
    __syncwarp();
  }
  __syncthreads();
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const int aux_count = aux_count_smem[i];
    if (aux_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(aux_counts) + i,
                static_cast<unsigned long long>(aux_count));
    }
    const int route_count = route_count_smem[i];
    if (route_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(bias_counts) + i,
                static_cast<unsigned long long>(route_count));
    }
  }
}
}  // namespace generic

// ============================================================================================
// Backward: one warp per token, lane l owns experts l + 32 j (TE's dense kernels' layout)
// ============================================================================================
template <typename GradT>
__device__ __forceinline__ float load_grad_bf16(const GradT *p);
template <>
__device__ __forceinline__ float load_grad_bf16<bf16>(const bf16 *p) {
  return __bfloat162float(*p);
}
template <>
__device__ __forceinline__ float load_grad_bf16<float>(const float *p) {
  // ToCopyBackward of token_probs.float(): fp32 gradient rounded to bf16
  return __bfloat162float(__float2bfloat16_rn(*p));
}

template <typename GradT, int E>
__global__ void __launch_bounds__(256)
    bwd_kernel(const bf16 *__restrict__ logits, const int64_t *__restrict__ token_indices,
               const uint8_t *__restrict__ routed, const GradT *__restrict__ grad_probs,
               const float *__restrict__ const_buf, const int64_t *__restrict__ global_counts,
               const float *__restrict__ grad_aux, int num_tokens, float scaling_factor,
               bf16 *__restrict__ grad_logits) {
  constexpr int kPerLane = (E + kWarp - 1) / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int token =
      static_cast<int>((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp);
  if (token >= num_tokens) return;
  const size_t row = static_cast<size_t>(token) * E;

  int r_e[kTopk];
  float r_g[kTopk];
#pragma unroll
  for (int k = 0; k < kTopk; ++k) {
    const size_t s = static_cast<size_t>(token) * kTopk + k;
    r_e[k] = routed[s] ? static_cast<int>(token_indices[s]) : -1;
    r_g[k] = load_grad_bf16<GradT>(grad_probs + s);  // dense bf16 grad at that expert
  }
  const float c_coeff = const_buf[0];
  const float g_aux = grad_aux[0];

  float act[kPerLane], g_topk[kPerLane], g_auxs[kPerLane];
  bool rt[kPerLane];
  float sum_act = 0.0f, sum_grad_act = 0.0f;      // top-k bwd (routed only)
  float a_sum_act = 0.0f, a_sum_grad_act = 0.0f;  // aux-score bwd (all experts)
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    const int e = lane + j * kWarp;
    if (E % kWarp != 0 && e >= E) {
      rt[j] = false;
      continue;
    }
    act[j] = sigmoid_scalar(__bfloat162float(logits[row + e]));
    float graw = 0.0f;
    bool r = false;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      if (r_e[k] == e) {
        graw = r_g[k];
        r = true;
      }
    }
    rt[j] = r;
    // fused_topk_with_score_function_backward_kernel, pass 1
    float g = graw * scaling_factor;
    g_topk[j] = g;
    if (r) {
      sum_act += act[j];
      sum_grad_act += g * act[j];
    }
    // fused_moe_aux_loss_backward_kernel value, then score-for-aux bwd pass 1
    const float tpe = static_cast<float>(global_counts[e]);
    const float ga = c_coeff * tpe * g_aux;
    g_auxs[j] = ga;
    a_sum_act += act[j];
    a_sum_grad_act += ga * act[j];
  }
  sum_act = warp_allreduce_sum(sum_act);
  sum_grad_act = warp_allreduce_sum(sum_grad_act);
  a_sum_act = warp_allreduce_sum(a_sum_act);
  a_sum_grad_act = warp_allreduce_sum(a_sum_grad_act);

#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    const int e = lane + j * kWarp;
    if (E % kWarp != 0 && e >= E) continue;
    float g = normalize_bwd_scalar(g_topk[j], rt[j], sum_act, sum_grad_act);
    if (!rt[j]) g = 0.0f;
    g = sigmoid_bwd_scalar(g, act[j]);
    const float topk_part = __bfloat162float(__float2bfloat16_rn(g));
    float ga = normalize_bwd_scalar(g_auxs[j], true, a_sum_act, a_sum_grad_act);
    ga = sigmoid_bwd_scalar(ga, act[j]);
    const float aux_part = __bfloat162float(__float2bfloat16_rn(ga));
    grad_logits[row + e] = __float2bfloat16_rn(__fadd_rn(topk_part, aux_part));
  }
}

int env_int(const char *name, int dflt) {
  const char *v = getenv(name);
  return v ? atoi(v) : dflt;
}

template <typename ProbT>
int fwd_impl(const void *logits, const void *expert_bias, int n, int num_experts,
             float scaling_factor, void *token_probs, void *token_indices, void *routed,
             void *aux_scores, void *aux_counts, void *bias_counts, int /*tok_per_block*/,
             void *stream) {
  if (n < 0) return 3;
  if (n == 0) return 0;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const bool sort16 = env_int("TOPK256_SORTN", 32) == 16;
  auto args = [&](auto kernel, int warps) {
    const int blocks = (n + warps - 1) / warps;
    kernel<<<blocks, warps * kWarp, 0, s>>>(
        reinterpret_cast<const bf16 *>(logits), reinterpret_cast<const float *>(expert_bias), n,
        scaling_factor, reinterpret_cast<ProbT *>(token_probs),
        reinterpret_cast<int64_t *>(token_indices), reinterpret_cast<uint8_t *>(routed),
        reinterpret_cast<float *>(aux_scores), reinterpret_cast<int64_t *>(aux_counts),
        reinterpret_cast<int64_t *>(bias_counts));
  };
  if (num_experts == 256) {
    if (env_int("TOPK256_GENERIC", 0)) {
      if (sort16)
        args(generic::fwd_kernel<ProbT, 256, 16>, generic::kWarpsPerBlock);
      else
        args(generic::fwd_kernel<ProbT, 256, 32>, generic::kWarpsPerBlock);
    } else {
      if (sort16)
        args(fast::fwd_kernel<ProbT, 16>, fast::kWarpsPerBlock);
      else
        args(fast::fwd_kernel<ProbT, 32>, fast::kWarpsPerBlock);
    }
  } else if (num_experts == 16) {
    if (sort16)
      args(generic::fwd_kernel<ProbT, 16, 16>, generic::kWarpsPerBlock);
    else
      args(generic::fwd_kernel<ProbT, 16, 32>, generic::kWarpsPerBlock);
  } else {
    return 1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <typename GradT>
int bwd_impl(const void *logits, const void *token_indices, const void *routed,
             const void *grad_probs, const void *const_buf, const void *global_counts,
             const void *grad_aux, int n, int num_experts, float scaling_factor,
             void *grad_logits, int /*allreduce_up*/, void *stream) {
  if (n < 0) return 3;
  if (n == 0) return 0;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t threads = static_cast<int64_t>(n) * kWarp;
  const int blocks = static_cast<int>((threads + 255) / 256);
  auto launch = [&](auto kernel) {
    kernel<<<blocks, 256, 0, s>>>(
        reinterpret_cast<const bf16 *>(logits), reinterpret_cast<const int64_t *>(token_indices),
        reinterpret_cast<const uint8_t *>(routed), reinterpret_cast<const GradT *>(grad_probs),
        reinterpret_cast<const float *>(const_buf),
        reinterpret_cast<const int64_t *>(global_counts), reinterpret_cast<const float *>(grad_aux),
        n, scaling_factor, reinterpret_cast<bf16 *>(grad_logits));
  };
  if (num_experts == 256)
    launch(bwd_kernel<GradT, 256>);
  else if (num_experts == 16)
    launch(bwd_kernel<GradT, 16>);
  else
    return 1;
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace

extern "C" {
// token_probs bf16 [T, 8]; grad_probs bf16 [T, 8]
int topk256_fwd(const void *logits, const void *expert_bias, int n, int num_experts,
                float scaling_factor, void *token_probs, void *token_indices, void *routed,
                void *aux_scores, void *aux_counts, void *bias_counts, int tok_per_block,
                void *stream) {
  return fwd_impl<bf16>(logits, expert_bias, n, num_experts, scaling_factor, token_probs,
                        token_indices, routed, aux_scores, aux_counts, bias_counts, tok_per_block,
                        stream);
}
int topk256_bwd(const void *logits, const void *token_indices, const void *routed,
                const void *grad_probs, const void *const_buf, const void *global_counts,
                const void *grad_aux, int n, int num_experts, float scaling_factor,
                void *grad_logits, int allreduce_up, void *stream) {
  return bwd_impl<bf16>(logits, token_indices, routed, grad_probs, const_buf, global_counts,
                        grad_aux, n, num_experts, scaling_factor, grad_logits, allreduce_up,
                        stream);
}
// MLPERF_CASTS_TOPK_F32: token_probs fp32 (= float(bf16 prob)); grad_probs fp32, rounded to bf16
int topk256_fwd_f32(const void *logits, const void *expert_bias, int n, int num_experts,
                    float scaling_factor, void *token_probs, void *token_indices, void *routed,
                    void *aux_scores, void *aux_counts, void *bias_counts, int tok_per_block,
                    void *stream) {
  return fwd_impl<float>(logits, expert_bias, n, num_experts, scaling_factor, token_probs,
                         token_indices, routed, aux_scores, aux_counts, bias_counts, tok_per_block,
                         stream);
}
int topk256_bwd_f32(const void *logits, const void *token_indices, const void *routed,
                    const void *grad_probs, const void *const_buf, const void *global_counts,
                    const void *grad_aux, int n, int num_experts, float scaling_factor,
                    void *grad_logits, int allreduce_up, void *stream) {
  return bwd_impl<float>(logits, token_indices, routed, grad_probs, const_buf, global_counts,
                         grad_aux, n, num_experts, scaling_factor, grad_logits, allreduce_up,
                         stream);
}
}
