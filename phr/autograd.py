"""
PHR custom autograd function.

Forward:  fused decode+matmul (no weight matrix materialized)
Backward: computes grad_x, grad_w_f, grad_lut (learnable LUT via scatter-add)

When offloading is active (layer_name is not None):
- W_p is NOT saved in ctx — only layer_name + offload_mgr are kept.
- Backward calls mgr.ensure_wp() to re-fetch W_p from pinned CPU memory.
- After backward, mgr.evict_wp() is called to reclaim GPU VRAM.
"""

import torch
from .kernel import phr_matmul


class PHRMatmulFunction(torch.autograd.Function):
    """
    Custom autograd for PHR matmul:  out = x @ (W_f + lut[W_p])

    - W_p: frozen byte indices (no grad)
    - W_f: trainable float residual
    - lut: trainable lookup table (STE — gradient scattered by W_p indices)
    - bias: handled outside (added in PHRLinear.forward, not here)

    Offloading extensions (optional, positional args 5–7):
      evict_cb:   callable fired at end of backward to evict W_p
      layer_name: str — if set, W_p is NOT saved to ctx (offloaded path)
      offload_mgr: OffloadManager instance for re-fetching W_p
    """

    @staticmethod
    def forward(ctx, x, W_p, W_f, lut, evict_cb=None, layer_name=None, offload_mgr=None):
        """
        Args:
            x:    [M, K] input activations   (bf16 or fp32)
            W_p:  [K, N] byte indices        (uint8, frozen)
            W_f:  [K, N] residual weights    (bf16)
            lut:  [256]  lookup table        (fp32, may require grad)
        Returns:
            out: [M, N] float32
        """
        if layer_name is not None:
            # Offloaded path — don't pin W_p in the autograd graph
            ctx.save_for_backward(x, W_f, lut)
            ctx._layer_name = layer_name
            ctx._offload_mgr = offload_mgr
        else:
            ctx.save_for_backward(x, W_p, W_f, lut)

        ctx._evict_cb = evict_cb
        out = phr_matmul(x, W_p, W_f, lut, bias=None)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if hasattr(ctx, '_offload_mgr') and ctx._offload_mgr is not None:
            x, W_f, lut = ctx.saved_tensors
            mgr = ctx._offload_mgr
            mgr.ensure_wp(ctx._layer_name)
            W_p = mgr._wp_params[ctx._layer_name]
        else:
            x, W_p, W_f, lut = ctx.saved_tensors

        # Compute W_full in bf16 to halve temp memory (W_f is bf16, lut is fp32)
        w_quantized_bf16 = lut[W_p.long()].to(torch.bfloat16)
        w_full_bf16 = W_f + w_quantized_bf16

        # grad_x uses bf16 weights (saves VRAM on the matmul intermediate)
        grad_x = (grad_out.to(torch.bfloat16) @ w_full_bf16.T).float()

        # dW_full in fp32 for accurate gradient scatter-add
        dW_full = x.T.float() @ grad_out.float()

        grad_w_f = dW_full.to(W_f.dtype)

        flat_W_p = W_p.flatten().long()
        flat_dW = dW_full.flatten()

        counts = torch.bincount(flat_W_p, minlength=256).float()
        grad_lut = torch.zeros(256, device=W_p.device, dtype=torch.float32)
        grad_lut.scatter_add_(0, flat_W_p, flat_dW)
        grad_lut /= (counts + 1e-8)

        if ctx._evict_cb is not None:
            ctx._evict_cb()

        return grad_x, None, grad_w_f, grad_lut, None, None, None
