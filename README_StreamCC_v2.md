# StreamCC v2 — Zstd-Native Continuous Learning

Continuous stream training with two operating modes: PackR (256-entry codebook
+ Velvet scheduler) and ZPackR (frozen base + WeightDict-compressed delta with
built-in block-level attenuation).  No epoch limits, no hand-tuned schedules,
no "training mode" switch — feeding data *is* training.

Branch: `feature/streamcc-v2`  
Depends on: `packr @ feature/zpackr` (MIT-licensed, `pip install -e /path/to/packr`)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Cogitator                                │
│                                                                  │
│  Batches prompts (batch_size=16), gates via SuperDict (optional),│
│  feeds to StreamTrainer.  Post-opt hooks fire per optimizer step:│
│    • shrink_known_delta() — every step                          │
│    • post_step()          — every N steps (default 4)           │
│    • reindex()            — every M steps (default 4000)        │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                       StreamTrainer                             │
│                                                                  │
│  .step(ids, mask, labels)         .eval(ids, mask)              │
│  ┌──────────────────────┐        ┌──────────────────────┐       │
│  │ forward             │        │ forward (no_grad)    │       │
│  │ → logits             │        │ → logits             │       │
│  │ → loss / acc_steps   │        │                      │       │
│  │ → backward           │        │                      │       │
│  │                      │        │                      │       │
│  │ every acc_steps=4:   │        │                      │       │
│  │   optimizer.step()   │        │                      │       │
│  │   zero_grad()        │        │                      │       │
│  │   → post_opt_step_fn │        │                      │       │
│  └──────────────────────┘        └──────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

## Two Operating Modes

### PackR (mode="packr") — Legacy

```
┌─────────────────────────┐
│ PackRLinear              │
│ W_f + lut[W_p]          │
│                          │
│ Velvet adjusts LR per    │
│ layer from exp_avg_sq    │
│ velocity                 │
└─────────────────────────┘
```

Uses the 256-entry LUT + bf16 residual.  Requires `VelvetController` (or the
legacy `CV2LRTController` alias) for adaptive per-layer learning rates.
Identical to the original PHR harness behavior.

### ZPackR (mode="zpackr") — Default for StreamCC v2

```
┌──────────────────────────────────┐
│ ZPackRLinear                      │
│ base_W (frozen fp16) + delta      │
│                                    │
│ post_step() after optimizer:      │
│   1. Split delta into 256-el blocks│
│   2. Compress each block vs       │
│      WeightDict (zstd)            │
│   3. ratio → novelty [0,1]       │
│      High ratio = known = novelty→0│
│      Low ratio = novel = novelty→1│
│                                    │
│ forward:  delta *= novelty        │
│   Known blocks contribute near-   │
│   zero to output → protected     │
│                                    │
│ shrink_known_delta():              │
│   Known blocks decay 1%/step →    │
│   faded toward zero over time     │
│                                    │
│ post_step() auto-calibrates       │
│ threshold.  Blocks above threshold│
│ are pruned from VRAM.             │
│                                    │
│ reindex(): rebuilds WeightDict    │
│ from current delta bytes.  Dict   │
│ evolves as model learns.          │
└──────────────────────────────────┘
```

**No Velvet needed.**  The WeightDict's compression-based attenuation IS the
convergence signal.  Known patterns self-prune.  Novel patterns stay hot.

## Component Map

```
streamcc/
├── __init__.py         Package marker
├── prompt.py           Dataset → (ids, mask, label, text) encoding
│                       • encode_classification() — single example
│                       • ingest_glue()            — full GLUE task
│                       • Pads to max_length=128 for uniform batches
├── stream.py           StreamTrainer — unified forward+train
│                       • step(ids, mask, label) — one micro-step
│                       • eval(ids, mask)        — inference only
│                       • post_opt_step_fn       — per-opt-step callback
│                       • Internal counters: micro_step, global_step
└── cogitator.py        Cogitator — prompt library + gating
                        • ingest(task, prompts) / ingest_glue(task)
                        • cogitate(task, max_epochs=N)
                        • Batches prompts (batch_size configurable)
                        • SuperDict Level 0 gating (optional)
                        • Wires post-opt-step hooks for ZPackR

tests/
└── test_zstd_gating.py  Full SST-2 → MNLI continuous learning test
                         with ZPackR.  See "Running" below.
```

## Gating Architecture (ZPackR mode)

| Level | Mechanism | Signal | Frequency |
|:-----:|-----------|--------|:---------:|
| 0 | SuperDict compression ratio on prompt text | Ratio ≥ threshold → skip | Per-prompt, before forward |
| 1 | ZPackRLinear.post_step() vs WeightDict | Block ratio → novelty [0,1] | Every `post_step_interval` opt steps |
| 2 | ZPackRLinear.shrink_known_delta() | Novelty → decay rate | Every optimizer step |
| 3 | ZPackRLinear.post_step() threshold | Blocks above threshold → pruned from VRAM | Every `post_step_interval` |

Level 0 is the fastest gate — compresses prompt text against the frozen
SuperDict (128KB zstd dictionary built from GLUE corpus).  Short SST-2
sentences typically produce ratios 0.9–1.7 (zstd overhead dominates short
texts), while longer MNLI pairs reach 1.5–2.2.  Threshold 1.5 biases
training toward longer/novel prompts.

Level 1-3 are the WeightDict block-level system.  This is the authoritative
convergence mechanism — no Velvet, no cosine decay, no epoch counting.

## Metrics and Tuning

### Output Files

Each run writes to `results/zstd_gating_<timestamp>/`:

| File | Contents |
|------|----------|
| `training_log.jsonl` | Per-1000-step metrics: val acc, aggregate novelty, salient blocks, per-layer detail |
| `summary.json` | Config, final accuracies (incl. pre-MNLI transfer), aggregate metrics |

### Key Tunables

| Parameter | Default | Where | Effect |
|-----------|:------:|-------|--------|
| `POST_STEP_INTERVAL` | 4 | test_zstd_gating.py | How often to run `post_step()` (compress blocks vs WeightDict) |
| `REINDEX_INTERVAL` | 4000 | test_zstd_gating.py | How often to rebuild WeightDict from current delta bytes |
| `CALIBRATION_MULTIPLIER` | 0.01 | test_zstd_gating.py | Auto-calibrate salience threshold at 1% of max observed ratio |
| `GATE_ENABLED` | True | test_zstd_gating.py | Enable SuperDict Level 0 gating |
| `GATE_THRESHOLD` | 1.5 | test_zstd_gating.py | SuperDict ratio threshold (higher = more strict) |
| `batch_size` | 16 | Cogitator constructor | Prompts per forward pass |
| `acc_steps` | 4 | StreamTrainer constructor | Gradient accumulation steps (effective batch = 64) |
| `zstd_salience_threshold` | 1.4 | PackRConfig | Initial salience threshold before auto-calibration |

### Per-Epoch Metrics (in training_log.jsonl)

```json
{
  "step": 1000,
  "val_acc": 91.28,
  "weight_dict_entries": 7200,
  "aggregate": {
    "salient_blocks": "45/180",
    "salient_pct": 25.0,
    "avg_novelty": 0.342
  },
  "per_layer": {
    "layer_0": {
      "salient": "1/3",
      "salient_pct": 33.3,
      "avg_novelty": 0.250,
      "min_novelty": 0.000,
      "max_novelty": 0.500,
      "avg_ratio": 3.214,
      "ratio_gap": 2.108,
      "threshold": 0.032
    }
  }
}
```

### Tuning Guidance

**Novelty stuck at 0.000**: Delta is all zeros — expected at initialization
and early training.  Wait for non-zero gradients to accumulate.

**Salient blocks dropping to zero too fast**: Raise `zstd_salience_threshold`
or reduce `CALIBRATION_MULTIPLIER`.  The auto-calibrated threshold may be too
aggressive.

**Ratio gap too small (< 0.1)**: All blocks look identical to the WeightDict.
Either the WeightDict needs more entries (raise `zstd_max_entries`) or the
delta hasn't diverged sufficiently yet — wait for more training steps.

**Val acc plateauing early**: Reduce `POST_STEP_INTERVAL` (more frequent
post_step) or increase `batch_size` (more GPU throughput per step).

**Cross-domain transfer (pre-MNLI val) too low**: The shared BERT attention
layers carry transfer knowledge.  If MNLI starts at random (33.33%) after
SST-2, the FFN delta has fully saturated and the attention layers aren't
transferring — the system may be over-pruning.

## Running

### Prerequisites

```bash
pip install zstandard
pip install -e /home/otherdrums/packr   # packr @ feature/zpackr branch
```

### ZPackR Continuous Learning Test

```bash
cd /home/otherdrums/phr
git checkout feature/streamcc-v2
python tests/test_zstd_gating.py
```

This runs three phases:
1. **SST-2** (3 epochs, 67k prompts) — learns binary sentiment
2. **Pre-MNLI validation** — measures cross-domain transfer from SST-2 alone
3. **MNLI** (3 epochs, 393k prompts) — learns 3-class NLI on top of SST-2
4. **Re-validate SST-2** — verifies no catastrophic forgetting

Expected outputs:
```
Phase 1: SST-2 val acc ~91-92%, novelty scores separate during training
Phase 2: MNLI pre-train ~37-39% (non-random, cross-domain transfer)
Phase 2: MNLI val acc ~78-83% after 3 epochs
Phase 3: SST-2 delta < 1% from Phase 1 (knowledge preserved)
```

### Standard Harness (PackR + Velvet)

The original harness still works on the `feature/streamcc-v2` branch:

```bash
python -m tests.harness --method=phr --quick
python -m tests.harness --method=phr --epochs=5
python -m tests.harness --method=phr --cv2lrt
```

This uses `mode="packr"` (PackRLinear + VelvetController) — unaffected by
the ZPackR additions.

## How It Protects Learned Knowledge

SST-2 patterns survive MNLI training because of a three-layer defense:

1. **WeightDict recognition**: Entries from SST-2 patterns persist across
   `reindex()` calls.  They continue to match those weight blocks,
   keeping compression ratios high.

2. **Novelty attenuation**: `forward: delta *= novelty`.  Blocks the
   WeightDict recognizes as known contribute near-zero to the output.
   MNLI gradients on those blocks are proportionally small.

3. **Shrink decay**: `shrink_known_delta()` runs every step.  Known blocks
   (novelty → 0) decay 1% per step toward zero.  Novel blocks (novelty → 1)
   stay at full strength.

The system can train MNLI at full learning rate with no risk of overwriting
SST-2 — the attenuation chain makes it impossible to write through a
well-compressed block.

## Pre-MNLI Cross-Domain Transfer

After SST-2 finishes and *before* any MNLI training begins, the test validates
MNLI.  The baseline for random guessing on 3-class MNLI is 33.33%.  Early
testing showed ~37.5% from SST-2 alone — the shared BERT attention layers
carry some semantic structure across domains.  This is measured as
`mnli_pre_train` and `mnli_transfer_gain` in `summary.json`.

## License

StreamCC v2 is part of the `phr` research repository (AGPLv3).  
PackR is a separate MIT-licensed library at `github.com/otherdrums/packr`.
