"""
PHR custom autograd function.

Forward:  fused decode+matmul (no weight matrix materialized)
Backward: computes grad_x, grad_w_f, grad_lut (learnable LUT via scatter-add)
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
    """

    @staticmethod
    def forward(ctx, x, W_p, W_f, lut):
        """
        Args:
            x:    [M, K] input activations   (bf16 or fp32)
            W_p:  [K, N] byte indices        (uint8, frozen)
            W_f:  [K, N] residual weights    (bf16)
            lut:  [256]  lookup table        (fp32, may require grad)
        Returns:
            out: [M, N] float32
        """
        ctx.save_for_backward(x, W_p, W_f, lut)
        out = phr_matmul(x, W_p, W_f, lut, bias=None)
        return out

    @staticmethod
    def backward(ctx, grad_out):
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
        
        # === Structural fix: per-entry mean gradient ===
        counts = torch.bincount(flat_W_p, minlength=256).float()
        grad_lut = torch.zeros(256, device=W_p.device, dtype=torch.float32)
        grad_lut.scatter_add_(0, flat_W_p, flat_dW)
        grad_lut /= (counts + 1e-8)  # normalize sum to mean
        # ===============================================

        return grad_x, None, grad_w_f, grad_lut
