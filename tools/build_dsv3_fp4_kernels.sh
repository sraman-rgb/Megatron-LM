#!/bin/bash
# Builds the CUDA libraries that the DeepSeek-V3 NVFP4 kernels on this branch load with ctypes.
# Each .so is written next to the Python module that loads it:
#   megatron/core/fusions/libmla_rope2.so          <- fusions/csrc/mla_rope2.cu      (MLPERF_MLA_ROPE2=1)
#   megatron/core/fusions/libcastfuse.so           <- fusions/csrc/castfuse.cu       (MLPERF_CASTFUSE=1)
#   megatron/core/transformer/moe/libtopk256.so    <- moe/csrc/topk256_casts.cu      (MLPERF_FUSED_TOPK256=1)
#   megatron/core/transformer/moe/libtopk256_casts.so  (same binary; MLPERF_CASTS_TOPK_F32=1)
# Usage: bash tools/build_dsv3_fp4_kernels.sh   (needs nvcc; GENCODE overrides the target list)
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
GENCODE=${GENCODE:-"-gencode=arch=compute_100a,code=sm_100a -gencode=arch=compute_103a,code=sm_103a -gencode=arch=compute_107a,code=sm_107a"}
NVCC_FLAGS="-O3 -std=c++17 ${GENCODE} -lineinfo -shared -Xcompiler -fPIC"
F=megatron/core/fusions
MOE=megatron/core/transformer/moe
nvcc ${NVCC_FLAGS} -o ${F}/libmla_rope2.so ${F}/csrc/mla_rope2.cu
nvcc ${NVCC_FLAGS} -o ${F}/libcastfuse.so ${F}/csrc/castfuse.cu
nvcc ${NVCC_FLAGS} -o ${MOE}/libtopk256_casts.so ${MOE}/csrc/topk256_casts.cu
cp ${MOE}/libtopk256_casts.so ${MOE}/libtopk256.so
ls -la ${F}/libmla_rope2.so ${F}/libcastfuse.so ${MOE}/libtopk256.so ${MOE}/libtopk256_casts.so
