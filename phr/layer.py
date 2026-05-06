"""
PHRLinear — Packed Hybrid Residual linear layer.

Replaces nn.Linear with:
  - W_p: uint8 byte indices (frozen, 1 byte/weight)
  - W_f: bfloat16 residual (trainable, 2 bytes/weight)
  - lut: float32 lookup table (trainable — learnable codebook)
  - bias: optional bfloat16

Total: 3 bytes/weight + 1 KB LUT overhead vs 4 bytes for float32.
"""

import torch
import torch.nn as nn
from .autograd import PHRMatmulFunction


def _build_codebook():
    """
    256-entry codebook: each byte encodes two 4-bit nibbles.
    Lower nibble = gain g ∈ [0,15] → (g - 7.5)/7.5  ∈ [-1.0, 1.0]
    Upper nibble = scale s ∈ [0,15] → (s + 1)/10.0  ∈ [0.1, 1.6]
    Combined: codebook[i] = gain * scale
    """
    codebook = torch.zeros(256)
    for i in range(256):
        g = i & 0x0F
        s = (i >> 4) & 0x0F
        gain = (g - 7.5) / 7.5
        scale = (s + 1.0) / 10.0
        codebook[i] = gain * scale
    return codebook


class PHRLinear(nn.Module):
    """
    PHR-linear layer with learnable LUT.

    Forward: out = x @ (W_f + lut[W_p]) + bias
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_p = nn.Parameter(
            torch.randint(0, 256, (in_features, out_features), dtype=torch.uint8),
            requires_grad=False,
        )
        self.W_f = nn.Parameter(
            torch.randn(in_features, out_features, dtype=torch.bfloat16) * 0.02
        )
        self.lut = nn.Parameter(torch.zeros(256, dtype=torch.float32))
        self.bias_f = nn.Parameter(
            torch.zeros(out_features, dtype=torch.bfloat16)
        ) if bias else None

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        # Kernel handles bf16 input natively — no .float() copy needed
        out = PHRMatmulFunction.apply(x, self.W_p, self.W_f, self.lut)

        if self.bias_f is not None:
            out = out + self.bias_f

        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_f is not None}"

    @classmethod
    def from_linear(cls, module: nn.Linear):
        """
        Convert a standard nn.Linear into a PHRLinear.

        Quantizes the original weights to the nearest codebook entry,
        stores byte indices in W_p, residual in W_f, and initializes
        the LUT with dequantized codebook values.
        """
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module)}")

        phr = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        device = module.weight.device

        # Build the codebook and compute dequantized values
        codebook = _build_codebook().to(device)

        W_orig = module.weight.detach().t()  # [Out, In] → [In, Out]

        # Scale to use full LUT range
        q99 = torch.quantile(W_orig.abs().float(), 0.99) + 1e-6
        scale = 1.6 / q99
        W_target = torch.clamp(W_orig * scale, -1.6, 1.6)

        # Find nearest codebook entry per weight — chunked to save memory
        codebook_3d = codebook.view(1, 1, 256)  # [1, 1, 256] for broadcasting
        codebook_device = codebook_3d.to(device)
        in_f, out_f = W_orig.shape
        best_indices = torch.empty(in_f, out_f, dtype=torch.uint8, device=device)
        chunk_size = 256

        for start in range(0, in_f, chunk_size):
            end = min(start + chunk_size, in_f)
            W_chunk = W_target[start:end].unsqueeze(-1)  # [chunk, out_f, 1]
            diff = torch.abs(W_chunk - codebook_device)   # [chunk, out_f, 256]
            best_indices[start:end] = torch.argmin(diff, dim=-1).to(torch.uint8)

        # Dequantized values in original scale
        dequant_lut = codebook / scale
        quantized_orig = dequant_lut[best_indices.long()]

        phr.W_p.data = best_indices.contiguous()
        phr.lut.data = dequant_lut.contiguous().to(device=device, dtype=torch.float32)
        phr.W_f.data = (W_orig - quantized_orig).contiguous().to(device=device, dtype=torch.bfloat16)

        if module.bias is not None:
            phr.bias_f.data = module.bias.detach().to(device=device, dtype=torch.bfloat16)

        return phr
