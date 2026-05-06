# PHR — Packed Hybrid Residual

Memory-efficient neural network training and inference using packed 8-bit
codebook-indexed weights with learnable residuals.

## Features

- **3 bytes/weight** storage vs 4 bytes for float32 (25% savings)
- **Learnable 256-entry codebook** with multiplicative nibble encoding
- **Fused CUDA decode kernel** for fast LUT lookup
- **8-bit AdamW optimizer** via Triton (FusedQuantizedAdam)
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

### Quick CUDA check

```bash
nvcc --version          # CUDA compiler
python -c "import torch; print(torch.version.cuda)"   # PyTorch CUDA version
python -c "import triton; print(triton.__version__)"  # Triton version
```

## Installation

```bash
git clone https://github.com/your-org/phr.git
cd phr
pip install -r requirements.txt
```

Or install directly:

```bash
pip install torch triton transformers datasets pynvml
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

The codebook uses a multiplicative nibble encoding:
- Lower nibble: gain ∈ [-1.0, 1.0]
- Upper nibble: scale ∈ [0.1, 1.6]
- Combined: codebook[i] = gain × scale

During training, the LUT adapts via scatter-add gradients through the
Straight-Through Estimator, while W_f learns the residual correction.
This gives the representational capacity of full-precision training at
3 bytes per weight.

## Testing

Run the SST-2 comparison harness:

```bash
cd tests
python harness.py                    # Full 5-epoch comparison
python harness.py --quick            # 10-batch quick check
python harness.py --method=phr       # PHR only
python harness.py --epochs=3         # Custom epoch count
```

## License

[Your license here]
