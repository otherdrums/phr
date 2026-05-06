"""
PHR fused decode kernel (CUDA) + matmul (cuBLAS via torch).

Drops Triton in favor of hand-tuned cuBLAS matmul (~3× faster on
CUDA-core-only GPUs like GTX 1650).  The decode step is a trivial
CUDA LUT-lookup kernel injected at import time via load_inline.
"""

import torch
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# JIT-compile the CUDA decode kernel (trivial LUT lookup, ~0.2ms for 2M els)
# ---------------------------------------------------------------------------
_cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void decode_packed_kernel(
    const uint8_t* __restrict__ W_p,
    const half* __restrict__ lut,
    half* __restrict__ decoded,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        decoded[idx] = lut[W_p[idx]];
    }
}

torch::Tensor decode_packed_cuda(torch::Tensor W_p, torch::Tensor lut) {
    TORCH_CHECK(W_p.is_contiguous(), "W_p must be contiguous");
    auto decoded = torch::empty(W_p.sizes(), W_p.options().dtype(torch::kHalf));
    int num_elements = W_p.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    decode_packed_kernel<<<blocks, threads>>>(
        W_p.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(lut.data_ptr<at::Half>()),
        reinterpret_cast<half*>(decoded.data_ptr<at::Half>()),
        num_elements
    );
    return decoded;
}
"""

_cpp_source = "torch::Tensor decode_packed_cuda(torch::Tensor W_p, torch::Tensor lut);"

_decode_ext = load_inline(
    name="phr_decode_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_cuda_source,
    functions=["decode_packed_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


def phr_matmul(x, W_p, W_f, lut, bias=None):
    """
    PHR matmul:  out = x @ (W_f + lut[W_p]) + bias

    Strategy (CUDA + cuBLAS):
      1. decode: CUDA LUT lookup     → decoded [K,N] fp16  (~0.2ms)
      2. combine: W_f + decoded      → w_full  [K,N] fp32  (~0.05ms)
      3. matmul: x @ w_full + bias   → out     [M,N] fp32  (~1.4ms cuBLAS)
                                                      Total: ~1.7ms

    Args:
        x:    [M, K] fp32 or bf16 activations
        W_p:  [K, N] uint8 byte indices (contiguous)
        W_f:  [K, N] bf16 residual weights
        lut:  [256]  fp32 lookup table
        bias: [N] optional bias

    Returns:
        [M, N] float32
    """
    assert x.is_cuda and W_p.is_cuda and W_f.is_cuda and lut.is_cuda
    assert W_p.dtype == torch.uint8, f"W_p must be uint8, got {W_p.dtype}"
    assert W_p.is_contiguous(), "W_p must be contiguous"

    M, K = x.shape
    K2, N = W_f.shape
    assert K2 == K, f"W_f rows ({K2}) != x cols ({K})"

    # 1. CUDA LUT lookup → fp16
    decoded = _decode_ext.decode_packed_cuda(W_p, lut.half())

    # 2. W_f (bf16) + decoded (fp16) → fp32 (PyTorch type promotion)
    w_full = W_f + decoded

    # 3. cuBLAS matmul — promote both to fp32 for accuracy
    if x.dtype != w_full.dtype:
        out = x.float() @ w_full
    else:
        out = x @ w_full

    if bias is not None:
        out = out + bias

    return out
