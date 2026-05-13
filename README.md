# PackR Research — Harness, StreamCC, and Experimentation

> **Warning — Early development.**  All components in this repository are under
> active development and not yet ready for production use.  APIs, training
> dynamics, and file formats are subject to change without notice.  Expect
> breakage, improvement, and iteration.

Research infrastructure for the PackR and ZPackR libraries.  Contains the
SST-2/MNLI comparison harness, the StreamCC continuous learning engine,
and experimental two-phase training scripts.

## What's Here

This repository depends on two standalone MIT-licensed libraries:

| Library | Repo | What it provides |
|---------|------|------------------|
| [packr](https://github.com/otherdrums/packr) | `pip install packr` | `PackRLinear`, `FusedQuantizedAdam`, `VelvetController` |
| [zpackr](https://github.com/otherdrums/zpackr) | `pip install zpackr` | `ZPackRLinear`, `WeightDict`, `SuperDict`, `prompt_gate` |

### Components

```
tests/
├── harness.py              SST-2 / MNLI comparison harness (packr mode)
├── configs.py              Method builders: full, lora, qlora, bitfit, phr
├── training.py             Shared train/eval loop
├── training_config.py      Hyperparameter defaults
├── analyzer.py             Post-hoc result tables + charts
├── memory_tracker.py       GPU VRAM tracking via nvml
└── test_zstd_gating.py     ZPackR continuous learning test (SST-2 → MNLI)

streamcc/
├── prompt.py               GLUE dataset → (ids, mask, label, text) encoding
├── stream.py               StreamTrainer — unified forward/train/inference
└── cogitator.py            Batched prompt scheduler with zstd-gated training

results/                    Saved training artifacts and metrics
```

### Two Operating Modes

**PackR mode** (stable harness path):
```bash
python -m tests.harness --method=phr --quick
python -m tests.harness --method=phr --epochs=5
python -m tests.harness --method=phr --cv2lrt
```
Uses `PackRLinear` with 256-entry LUT + bf16 residual.  Optional `VelvetController`
for adaptive per-layer learning rates.

**ZPackR mode** (experimental, on `feature/streamcc-v2`):
```bash
python tests/test_zstd_gating.py
```
Uses `ZPackRLinear` with frozen base + WeightDict-compressed delta.  Block-level
zstd compression ratios replace the LR scheduler entirely.  Three-phase test:
SST-2 → MNLI → re-validate SST-2.

## StreamCC — Continuous Stream Training

StreamCC unifies inference and training into a single token stream processor.
Feeding data *is* training — no mode switch, no separate dataloader, no
external loss computation.

[README_StreamCC_v2.md](README_StreamCC_v2.md) covers the full architecture,
gating chain, metrics, and tuning guidance.

## Requirements

```bash
pip install packr            # Required
pip install zpackr           # For ZPackR mode (feature/streamcc-v2)
pip install zstandard        # For ZPackR mode
pip install transformers datasets peft nvidia-ml-py matplotlib  # Harness
```

## License

PackR Research is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.
See [LICENSE](LICENSE).

The PackR and ZPackR libraries are separate MIT-licensed projects.
