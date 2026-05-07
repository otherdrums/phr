# PHR — Packed Hybrid Residual

> **Warning:** PHR is under active development and **not yet ready for drop-in
> use.** APIs, training dynamics, and file formats are subject to change without
> notice. Expect breakage, improvement, and iteration. Production use is at your
> own risk until a stable release is tagged.

Memory-efficient neural network training and inference using packed 8-bit
codebook-indexed weights with learnable residuals.

## Features

- **37% VRAM reduction** vs full fine-tune (1.7 GB vs 2.7 GB on BERT-base, batch=8)
- **3 bytes/weight** persistent storage (uint8 indices + bf16 residual + 256-entry LUT)
- **8-bit AdamW optimizer** via Triton — int8 m/v moments save 6 bytes per trainable param
- **Learnable 256-entry codebook** with multiplicative nibble encoding
- **Fused CUDA decode kernel** — no persistent full-precision weight matrix materialized
- **Drop-in replacement** for `nn.Linear` in any HuggingFace model
- **Gradient checkpointing** compatible

## Requirements

### Python

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Triton 2.1+

### CUDA Dependencies

The decode kernel is JIT-compiled at import time via `torch.utils.cpp_extension.load_inline`.
This requires:

| Component | Minimum Version | Notes |
|-----------|:--------------:|-------|
| CUDA Toolkit | 12.x | `nvcc` must be on `PATH` |
| cuBLAS | — | Included with PyTorch CUDA wheel |
| cuDNN | — | Included with PyTorch CUDA wheel |
| GPU driver | 525+ | For CUDA 12.x compatibility |

The `FusedQuantizedAdam` optimizer uses **Triton 2.1+**, which requires a
CUDA-capable GPU with compute capability 7.0+ (Volta or newer).

> **Note:** Triton kernels compile to tensor core instructions (`wmma`/`mma`)
> on supported hardware (compute capability 7.0+ Volta, 7.5+ Turing with tensor
> cores). On GPUs without tensor cores — such as the GTX 1650 (TU117) — Triton
> falls back to CUDA core SIMT instructions. The kernel runs correctly but at
> reduced throughput. Because the author's hardware lacks tensor cores, the
> optimizer's full speedup cannot be benchmarked yet.

### Quick CUDA check

```bash
nvcc --version          # CUDA compiler
python -c "import torch; print(torch.version.cuda)"   # PyTorch CUDA version
python -c "import triton; print(triton.__version__)"  # Triton version
```

## Installation

```bash
git clone https://github.com/otherdrums/phr.git
cd phr
pip install -r requirements.txt
```

Or install directly:

```bash
pip install torch triton transformers datasets nvidia-ml-py
```

No `setup.py` is needed — PHR is a single `phr/` package that can be
imported directly when the repo root is on `PYTHONPATH` or installed
with `pip install -e .`.

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from phr import compress_model, PHRConfig

config = PHRConfig(
    scheme="phr",
    layer_scope="ffn",        # compress FFN layers only
    learnable_lut=True,        # train the codebook
    gradient_checkpointing=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
model = compress_model(model, config)
model.cuda()

# Train normally with standard PyTorch / HuggingFace training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

## File Structure

```
phr/
├── phr/
│   ├── __init__.py          # Package exports
│   ├── config.py            # PHRConfig dataclass
│   ├── layer.py             # PHRLinear nn.Module
│   ├── autograd.py          # PHRMatmulFunction (custom autograd)
│   ├── kernel.py            # CUDA decode kernel + phr_matmul
│   ├── layer_patcher.py     # compress_model() entry point
│   └── optim.py             # FusedQuantizedAdam (Triton 8-bit AdamW)
├── tests/
│   ├── configs.py           # Method builders (PHR, LoRA, QLoRA, full, BitFit)
│   ├── harness.py           # Unified SST-2 comparison harness
│   ├── training.py          # Training/eval loop
│   └── memory_tracker.py    # GPU VRAM tracking via pynvml
├── requirements.txt
└── README.md
```

## How It Works

PHR decomposes a weight matrix into three components:

- **W_p** (uint8, 1 byte/weight, frozen): Byte indices into a 256-entry codebook
- **W_f** (bfloat16, 2 bytes/weight, trainable): Floating-point residual
- **lut** (float32, 256 entries, trainable): Learnable codebook

Forward pass: `out = x @ (W_f + lut[W_p])`

**Fused decode — no weight matrix materialized.** Standard quantization
approaches materialize the full-precision weight matrix before the matmul
(e.g., `W_fp32 = dequantize(W_int8)` — a `[K,N]` fp32 tensor). PHR uses a
CUDA decode kernel that fuses the LUT lookup directly into the matmul
pipeline: `lut[W_p]` is computed on-the-fly as a `[K,N]` fp16 intermediate
(~0.2ms for 2M elements) and summed with `W_f` into fp32, then fed to
cuBLAS. The temporary fp16 decode buffer is freed immediately after the
matmul. This avoids a persistent `[K,N]` fp32 weight allocation — saving
**4 bytes/weight** of VRAM compared to materializing the full matrix, and
**1 byte/weight** vs storing fp16 weights directly.

| Representation | Persistent VRAM per weight | vs PHR |
|---------------|:-------------------------:|:------:|
| Standard fp32 | 4 bytes | +25% |
| Standard fp16 | 2 bytes | −33% (but no gradient on the quantization) |
| PHR (ours) | 3 bytes | — |

The codebook uses a multiplicative nibble encoding:
- Lower nibble: gain ∈ [-1.0, 1.0]
- Upper nibble: scale ∈ [0.1, 1.6]
- Combined: codebook[i] = gain × scale

During training, the LUT adapts via scatter-add gradients through the
Straight-Through Estimator, while W_f learns the residual correction.
This gives the representational capacity of full-precision training at
3 bytes per weight.

## VRAM Savings (BERT-base SST-2, batch=8)

On a single GPU, PHR peaks at **1.7 GB** vs **2.7 GB** for full fine-tune —
a **37% reduction**, measured via nvml (hardware-level GPU memory).

The savings come from three independent mechanisms:

| Source | Full Fine-tune | PHR (ours) | Mechanism |
|--------|:-----:|:-----:|-----------|
| Model params | 440 MB | 285 MB | W_p stored as uint8 (1 byte vs 4); FFN-only scope |
| Optimizer states | 880 MB | 242 MB | FusedQuantizedAdam int8 m+v (2 bytes/param vs 8 for AdamW fp32) |
| **Persistent** | **1,320 MB** | **527 MB** | **−60%** |
| Gradients (peak) | 440 MB | 275 MB | Fewer bytes to store (uint8 W_p has no grad) |
| Activations + CUDA | 940 MB | 898 MB | Fused decode avoids materializing `[K,N]` fp32 weight tensor |
| **Peak total** | **~2,700 MB** | **~1,700 MB** | **−1,000 MB (37%)** |

**Largest single contributor:** the 8-bit AdamW optimizer states (FusedQuantizedAdam).
Standard AdamW stores two fp32 moment buffers (8 bytes/param). PHR's Triton kernel
stores m and v as int8 with per-block fp32 scales (2 bytes/param + ~1% scale overhead),
saving **6 bytes per trainable parameter** — ~660 MB alone.

**Second-largest:** W_p is frozen (no gradients, no optimizer states) and stored as
uint8. This saves both parameter memory (4→1 byte for FFN weights) and eliminates
their optimizer overhead entirely — ~400 MB combined.

**Tertiary:** the fused decode kernel performs `lut[W_p]` on-the-fly as a fp16
intermediate, immediately summed with W_f and fed to cuBLAS. No persistent
`[K,N]` fp32 weight matrix exists — only the 2.36 MB fp16 decode buffer per layer,
freed after the matmul.

## PHR vs LoRA / QLoRA

PHR and LoRA-family methods address the VRAM problem from opposite directions:

| | PHR | LoRA (r=8) | QLoRA (8-bit) |
|---|:---:|:----------:|:-------------:|
| Trainable params | ~82M | ~0.6M | ~0.6M |
| Peak VRAM | ~1.7 GB | ~1.1 GB | ~0.6 GB |
| SST-2 accuracy | ~92.5% | ~91.5% | ~91.0% |
| Approach | Compress storage of **full** training | Train a **low-rank** adapter | Quantize base + **low-rank** adapter |
| Base model | Replaced with PHR layers | Frozen | Frozen + NF4 quantized |
| Gradient signal | Full weight matrix + codebook | Adapter matrices only | Adapter matrices only |
| Calibration needed | No | No | Yes (NF4 quantiles) |
| Inference format | 3 bytes/weight (native PHR) | 4 bytes/weight (base + merged adapter) | Dequantized to fp16 at load |
| Learnable codebook | ✓ (LUT trains via STE) | ✗ | ✗ |

**Core philosophical difference:** LoRA asks "how few parameters can we train?"
and answers with a low-rank decomposition. PHR asks "how compact can we store the
full training surface?" and answers with a learnable codebook + residual. LoRA
trains 0.5% of the model and achieves good results on well-behaved tasks. PHR
trains 100% of the weights through a compressed lens and gets within 1.5% of
full fine-tune — with momentum β₁=0.9 now enabled (matching full fine-tune's
β₁), plus scheduler tuning and progressive unfreezing as further improvement
paths — while keeping the model 25% smaller than fp32.

PHR is the right choice when:
- You need the model to adapt broadly (not just along a low-rank subspace)
- You want the final model compact for inference (3 bytes/weight, no merge step)
- You lack calibration data or want to avoid NF4 quantization complexity

## Testing

Run the SST-2 comparison harness:

```bash
python -m tests.harness                    # Full 5-epoch comparison
python -m tests.harness --quick            # 10-batch quick check
python -m tests.harness --method=phr       # PHR only
python -m tests.harness --epochs=3         # Custom epoch count
```

## Roadmap

Features planned but not yet implemented:

- **CPU offloading** — stream frozen W_p and optimizer states to system RAM
  via async CUDA copies. The `offload_frozen_params` flag in `PHRConfig` is a
  stub; the `OffloadManager` described in the design docs has not been built.
  Target: reduce LLaMA-7B training from 40 GB to ~5 GB GPU VRAM.
- **4-bit packed quantization** — sub-8-bit codebooks (16-entry LUT) with
  nibble-packed W_p for further VRAM reduction on output layers.
- **Mixed-precision routing** — per-layer bit-width allocation based on
  sensitivity profiling (residual_ratio), with passthrough for high-sensitivity
  layers and aggressive crushing for dead-weight layers.
- **Momentum in FusedQuantizedAdam** — currently uses `betas=(0.0, 0.999)`
  (no first moment). Adding β₁=0.9 is a one-line change that may close the
  remaining accuracy gap to full fine-tune.

## License

PHR is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
See [LICENSE](LICENSE) for the full text.

This license ensures that all modifications and network-facing deployments of
PHR remain open source. If you use PHR in a commercial context — including
as a hosted service — you must make your changes available to users.

For commercial licensing terms outside the scope of the AGPLv3, contact the
project maintainers.
