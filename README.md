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
- **8-bit AdamW optimizer** (`FusedQuantizedAdam`) via Triton — int8 m/v moments save 6 bytes per trainable param, with full β₁=0.9 momentum matching standard AdamW
- **3-phase LR schedule** — linear warmup → constant hold → cosine decay, with differential LR for body (2e-5) vs classification head (1e-3)
- **CV2LRT adaptive scheduler** — Continuous Velocity to Learning Rate Translation: reads AdamW second-moment velocity (exp_avg_sq) in real time and adapts per-layer learning rates without hand-tuned schedules
- **Learnable 256-entry codebook** with multiplicative nibble encoding
- **Fused CUDA decode kernel** — no persistent full-precision weight matrix materialized
- **Drop-in replacement** for `nn.Linear` in any HuggingFace model
- **Gradient checkpointing** compatible
- **CPU/system RAM offloading** — stream frozen W_p indices and optimizer states
  from pinned system RAM to GPU on demand via `offload=True`

## Requirements

### Python

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Triton 2.1+

### CUDA Dependencies

PHR uses two separate CUDA compilation paths:

| Kernel | Purpose | Compiler | Needs tensor cores? |
|--------|---------|----------|:---:|
| `decode_packed_cuda` | LUT lookup (`lut[W_p]`) | nvcc via `load_inline` | No |
| `_fused_adam_8bit_kernel` | 8-bit AdamW optimizer | Triton `@triton.jit` | No (but helps) |

**Decode kernel** (`kernel.py`): Raw CUDA C++ source is injected into Python at
import time via `torch.utils.cpp_extension.load_inline` and compiled with nvcc.
This kernel runs on any CUDA GPU and uses `--use_fast_math` for throughput.
Requires `nvcc` on `PATH`.

| Component | Minimum Version |
|-----------|:--------------:|
| CUDA Toolkit | 12.x |
| cuBLAS | Included with PyTorch CUDA wheel |
| GPU driver | 525+ |

**Optimizer kernel** (`optim.py`): Triton `@triton.jit` kernel compiled to the
target GPU's instruction set. On hardware with tensor cores (Volta+, Turing 7.5+)
Triton emits `wmma`/`mma` instructions. On GPUs without tensor cores — such as
the GTX 1650 (TU117) — Triton falls back to a CUDA core SIMT compilation path.
Both paths are the same Python kernel; only the generated PTX differs.

> The optimizer's full tensor-core throughput has not been benchmarked because
> the author's hardware lacks tensor cores. The SIMT path is functional and
> believed to benefit from Triton's block-level scheduling, but measurements
> on tensor-core hardware are needed to characterize the designed speedup.

Requires `triton>=2.1.0` and a CUDA-capable GPU with compute capability 7.0+.

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
pip install torch triton transformers datasets nvidia-ml-py peft
```

No `setup.py` is needed — PHR is a single `phr/` package that can be
imported directly when the repo root is on `PYTHONPATH` or installed
with `pip install -e .`.

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from phr import compress_model, PHRConfig, FusedQuantizedAdam
from torch.optim.lr_scheduler import (
    LinearLR, SequentialLR, LambdaLR
)
import math

def _cosine_factor(total, min_factor):
    """Lambda function: 1.0 → min_factor cosine decay."""
    return lambda s: min_factor + 0.5 * (1 - min_factor) * \
        (1 + math.cos(math.pi * min(s / total, 1)))

# 3-phase LR schedule: warmup → hold → cosine decay
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.02 * total_steps)     # 2% warmup
hold_start = int(0.60 * total_steps)       # decay starts at 60%
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        LinearLR(optimizer, start_factor=1.0, end_factor=1.0,
                 total_iters=hold_start - warmup_steps),
        LambdaLR(optimizer, lr_lambda=_cosine_factor(
            total_steps - hold_start, 0.1)),
    ],
    milestones=[warmup_steps, hold_start],
)

config = PHRConfig(
    scheme="phr",
    layer_scope="ffn",        # compress FFN layers only
    learnable_lut=True,        # train the codebook
    gradient_checkpointing=True,
    offload=True,              # enable CPU/system RAM offloading
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
model = compress_model(model, config)
model.cuda()

# Differential LR: head gets 50x higher LR than body
head_params = [p for n, p in model.named_parameters()
               if "classifier" in n or "cls" in n]
body_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "classifier" not in n and "cls" not in n]

optimizer = FusedQuantizedAdam(
    [
        {"params": body_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-3},
    ],
    betas=(0.9, 0.999),   # full momentum matching standard AdamW
)

# 3-phase LR schedule: warmup → hold → linear decay
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.02 * total_steps)     # 2% warmup
hold_start = int(0.60 * total_steps)       # decay starts at 60%
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        LinearLR(optimizer, start_factor=1.0, end_factor=1.0,
                 total_iters=hold_start - warmup_steps),
        LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                 total_iters=total_steps - hold_start),
    ],
    milestones=[warmup_steps, hold_start],
)

# Step scheduler at micro-batch granularity for smooth cosine
for batch in train_loader:
    loss = model(**batch).loss
    loss.backward()
    scheduler.step()                       # per micro-batch
    if (batch_idx + 1) % acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## File Structure

```
phr/
├── phr/
│   ├── __init__.py          # Package exports
│   ├── config.py            # PHRConfig dataclass
│   ├── cv2lrt.py            # CV2LRT — adaptive per-layer LR from gradient velocity
│   ├── offload.py           # OffloadManager (CPU↔GPU tensor streaming)
│   ├── layer.py             # PHRLinear nn.Module
│   ├── autograd.py          # PHRMatmulFunction (custom autograd)
│   ├── kernel.py            # CUDA decode kernel + phr_matmul
│   ├── layer_patcher.py     # compress_model() entry point
│   └── optim.py             # FusedQuantizedAdam (Triton 8-bit AdamW)
├── tests/
│   ├── configs.py           # Method builders (PHR, LoRA, QLoRA, full, BitFit)
│   ├── training_config.py   # TrainingConfig — single source of truth for hyperparams
│   ├── harness.py           # Unified SST-2 comparison harness
│   ├── training.py          # Training/eval loop
│   └── memory_tracker.py    # GPU VRAM tracking via nvidia-ml-py
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
| SST-2 accuracy | ~92.4% | ~91.5% | ~91.0% |
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
trains 100% of the weights through a compressed lens with little to no accuracy
loss while keeping the model 25% smaller than fp32.

## CPU / System RAM Offloading

PHR's three-component weight representation inherently decouples storage
from compute.  Frozen W_p indices never mutate after initialization, so
they can stream from pinned system RAM on demand — only the current
layer's W_p needs GPU memory at any moment.  Combined with
FusedQuantizedAdam's int8 moments, optimizer states live entirely in
system RAM, streaming to GPU only during the optimizer step.

Enable offloading with a single flag:

```python
config = PHRConfig(offload=True)
model = compress_model(model, config)
```

The harness supports it via `--offload`:

```bash
python -m tests.harness --method=phr --offload
```

### Architecture

The `OffloadManager` (`phr/offload.py`) coordinates three mechanisms:

- **W_p streaming** — A small GPU buffer pool reuses tensors for the current
  layer's forward pass.  Pinned CPU memory holds canonical uint8 indices.
  Synchronous default-stream copies avoid races with cuBLAS internal streams.

- **Chunked optimizer state streaming** — m/v/scales are stored as pinned CPU
  tensors grouped into ~100 MB chunks.  During `step()`, each chunk's states
  are copied to GPU via non-blocking transfers, used by the Triton kernel,
  then evicted via double-buffered offload-stream DMA (overlaps with the
  next chunk's compute).

- **Fused C++ CPU AdamW kernel** — A JIT-compiled C++ kernel via `load_inline`
  runs quantized AdamW directly on CPU flat pinned buffers.  Currently too
  slow for standalone use (~2.3s/step vs ~0.08s GPU Triton on BERT-base) but
  preserved for future training-loop overlap.

### Benchmarked Results

Measured on a Quadro T1000 (3.6 GB), batch=2 for BERT-large:

| Model | Offload | Peak VRAM | Throughput |
|-------|---------|-----------|------------|
| BERT-base (0.1B) | off | 1.65 GB | 13.0 sps |
| BERT-base | on | 1.52 GB (−8%) | 12.9 sps |
| BERT-large (0.3B) | off | 3.29 GB | 2.0 sps |
| BERT-large | on | 2.68 GB (−19%) | 1.4 sps |

For BERT-base the savings are modest (~130 MB) since W_p is only 56 MB.
On a 7B model, W_p alone would save ~3.7 GB of GPU VRAM.  The BERT-large
throughput regression (−30%) is bounded by per-chunk stream synchronization;
training-loop restructuring to overlap GPU compute with CPU DMA would
recover this.

## CV2LRT — Adaptive Per-Layer LR Scheduling

> **Experimental — on `feature/cv2lrt` branch.**  Not yet merged to `master`.
> See [feature/cv2lrt](https://github.com/otherdrums/phr/tree/feature/cv2lrt).

CV2LRT (Continuous Velocity to Learning Rate Translation) is a closed-loop
adaptive learning rate controller that replaces hand-tuned schedules entirely.
It reads the AdamW second-moment buffer (`exp_avg_sq`, the "v" state) every
optimizer step, extracts a real-time signal of how quickly each layer is
learning, and dynamically adjusts per-layer learning rates without pausing
or wasting steps.

### How It Works

1. **After every `optimizer.step()`**, the CPU reads the freshly-updated
   `exp_avg_sq` tensor for each parameter and computes its mean.

2. **Velocity computation**: `Δv = v_mean_current − v_mean_previous` captures
   whether the layer's gradients are still climbing (active learning) or have
   flattened out (saturation).

3. **EMA low-pass filter**: The raw step-to-step `Δv` is noisy (SGD is
   stochastic — every micro-batch has different data).  An EMA with β=0.97
   (half-life ~23 steps) smooths out micro-batch jitter, producing a clean
   velocity signal.

4. **Normalize**: Divide the EMA velocity by the current `v_mean` to get the
   *relative* rate of change — this makes the signal comparable across layers
   with different weight magnitudes.

5. **Translate to LR multiplier**: `multiplier = clamp(0.1, 1.0, norm_vel × 10)`.
   When a layer is hungry (high velocity) → multiplier stays at 1.0 (full LR).
   When a layer saturates (velocity → 0) → multiplier decays to 0.1.

The result: every matrix in the network continuously reads its own internal
structural stress and adjusts its own learning rate.  No epochs, no schedules,
no hand-tuning.

### Using CV2LRT

```python
from phr import CV2LRTController

# Create controller (captures base LRs automatically)
cv2lrt = CV2LRTController(optimizer, beta=0.97, min_multiplier=0.1)

# In training loop:
for step, batch in enumerate(train_loader):
    loss = model(**batch).loss
    loss.backward()

    if step < warmup_steps:
        cv2lrt.warmup_step(step, warmup_steps)   # linear warmup
    elif (step + 1) % acc_steps == 0:
        optimizer.step()
        cv2lrt.step()                             # reads v, adjusts LRs
        optimizer.zero_grad()
```

Or via the harness:

```bash
python -m tests.harness --method=phr --cv2lrt
```

### Granularity

CV2LRT supports three grouping levels for independent LR control:

| Level | Groups (bert-base) | Description |
|-------|:---:|---|
| `"matrix"` | ~102 | Each named module gets its own LR |
| `"layer"` | ~14 | Per-encoder-layer grouping |
| `"coarse"` | 2 | Body + head (same as current differential LR) |

Set via `TrainingConfig.cv2lrt_granularity` or the `build_phr_cv2lrt()` builder.

### Int8 State Handling

`FusedQuantizedAdam` stores `exp_avg_sq` as int8 with per-block float32
scales.  CV2LRT auto-detects this format and dequantizes correctly — the
mean of a block-quantized tensor is mathematically identical to the
float32 mean, so the velocity signal is lossless.

## Testing

Run the SST-2 comparison harness:

```bash
python -m tests.harness                    # All methods, 5 epochs
python -m tests.harness --all              # Same — explicit full comparison
python -m tests.harness --quick            # 10-batch quick check
python -m tests.harness --method=phr       # PHR only
python -m tests.harness --method=phr --offload  # PHR with CPU offloading
python -m tests.harness --method=phr --cv2lrt   # PHR with adaptive CV2LRT scheduler
python -m tests.harness --epochs=3         # Custom epoch count
```

After training, generate analysis:

```bash
python -m tests.analyzer                   # Tables + charts in results/_analysis/
```

## Roadmap

Features implemented:

- **CV2LRT adaptive scheduler** — ✅ Continuous Velocity to Learning Rate
  Translation.  Reads AdamW second-moment velocity (`exp_avg_sq`) every optimizer
  step, applies an EMA low-pass filter (β=0.97) to separate signal from
  micro-batch noise, and translates the filtered velocity to per-layer LR
  multipliers in real time.  Fully automatic — no hand-tuned schedules needed.
  Configurable granularity (`"matrix"`, `"layer"`, `"coarse"`).
  ```python
  from phr import CV2LRTController
  cv2lrt = CV2LRTController(optimizer, beta=0.97, min_multiplier=0.1)
  # In training loop:
  if step < warmup_steps:
      cv2lrt.warmup_step(step, warmup_steps)
  elif acc_step:
      optimizer.step()
      cv2lrt.step()
  ```
  ```bash
  python -m tests.harness --method=phr --cv2lrt
  ```

- **CPU offloading** — ✅ Set `offload=True` in `PHRConfig`.  Frozen W_p
  indices are streamed from pinned CPU RAM via a GPU buffer pool.  Optimizer
  m/v/scales are stored as individual pinned CPU tensors and streamed in
  ~100 MB chunks during `step()`.  Double-buffered offload-stream eviction
  DMA overlaps with the next chunk's compute.  Fused C++ CPU AdamW kernel
  is JIT-compiled for future training-loop overlap work.
  ```python
  config = PHRConfig(offload=True)
  model = compress_model(model, config)
  ```
  ```bash
  python -m tests.harness --method=phr --offload
  ```

Features planned but not yet implemented:

- **4-bit packed quantization** — sub-8-bit codebooks (16-entry LUT) with
  nibble-packed W_p for further VRAM reduction on output layers.
- **Mixed-precision routing** — per-layer bit-width allocation based on
  sensitivity profiling (residual_ratio), with passthrough for high-sensitivity
  layers and aggressive crushing for dead-weight layers.

## License

PHR is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
See [LICENSE](LICENSE) for the full text.

This license ensures that all modifications and network-facing deployments of
PHR remain open source. If you use PHR in a commercial context — including
as a hosted service — you must make your changes available to users.

For commercial licensing terms outside the scope of the AGPLv3, contact the
project maintainers.
