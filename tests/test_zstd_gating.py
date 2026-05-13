"""StreamCC zstd-native continuous learning test.

ZPackR (mode="zpackr") replaces nn.Linear with frozen base + WeightDict-compressed
delta.  Block-level compression ratios against the WeightDict serve as the
authoritative convergence signal — no Velvet scheduler needed.

Known blocks (high ratio → low novelty): attenuated in forward, decayed over
time, pruned from VRAM below auto-calibrated threshold.
Novel blocks (low ratio → high novelty): kept at full strength, full LR.

Metrics collected per epoch and saved to results/zstd_gating_<run_id>/:
  - Per-layer: salient blocks, avg novelty, threshold, ratios, delta L2 norms
  - Per-task: validation accuracy, SuperDict gating counts
  - Full block-level detail for post-hoc threshold tuning
"""

import torch
import torch.nn as nn
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset

from packr import compress_model
from packr.config import PackRConfig
from packr.optim import FusedQuantizedAdam
from packr.super_dict import load_super_dict
from packr.zpackr_layer import ZPackRLinear

from streamcc.stream import StreamTrainer
from streamcc.cogitator import Cogitator


# ── Output directory ──
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", f"zstd_gating_{RUN_ID}")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Shared validation ──
tok = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)


def _val_loader(task: str, split: str):
    if task == "sst2":
        def tokenize(batch):
            return tok(batch["sentence"], truncation=True, padding="max_length", max_length=128)
    else:
        def tokenize(batch):
            return tok(batch["premise"], batch["hypothesis"],
                       truncation=True, padding="max_length", max_length=128)
    split_key = "validation" if task == "sst2" else "validation_matched"
    ds = load_dataset("glue", task)[split_key].map(tokenize, batched=True)
    ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(ds, batch_size=32, shuffle=False)


def validate(model, task: str) -> float:
    device = next(model.parameters()).device
    model.eval()
    loader = _val_loader(task, "validation" if task == "sst2" else "validation_matched")
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            correct += (out.logits.argmax(-1) == batch["label"].to(device)).sum().item()
            total += batch["label"].size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


# ── ZPackR layer helpers ──

def _zpackr_layers(model):
    return [m for m in model.modules() if isinstance(m, ZPackRLinear)]


def post_step_all(model):
    for layer in _zpackr_layers(model):
        layer.post_step()


def decay_all(model):
    for layer in _zpackr_layers(model):
        layer.decay_delta()


# ── Metrics collection ──

def _log(path, entry):
    entry["timestamp"] = datetime.now().isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def collect_layer_metrics(model):
    """Collect per-layer block metrics for tuning and analysis."""
    metrics = {}
    for i, layer in enumerate(_zpackr_layers(model)):
        info = layer.get_block_ratios()
        if not info:
            continue
        scores = info.get("novelty_scores", [])
        ratios = info.get("ratios", [])
        kept = info.get("salient_count", 0)
        total = info.get("num_blocks", 0)
        threshold = info.get("calibrated_threshold")

        # Per-block detail (full, for threshold tuning)
        block_detail = []
        for blk in range(total):
            r = ratios[blk] if blk < len(ratios) else 1.0
            n = scores[blk] if blk < len(scores) else 1.0
            active = r < (threshold or 1.4)  # whether this block stays salient
            block_detail.append({
                "block": blk,
                "ratio": round(r, 4),
                "novelty": round(n, 4),
                "salient": active,
            })

        name = f"layer_{i}"
        metrics[name] = {
            "salient_blocks": kept,
            "total_blocks": total,
            "salient_pct": round(100.0 * kept / total, 1) if total else 0,
            "avg_novelty": round(sum(scores) / len(scores), 4) if scores else 1.0,
            "min_novelty": round(min(scores), 4) if scores else 1.0,
            "max_novelty": round(max(scores), 4) if scores else 1.0,
            "avg_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 1.0,
            "min_ratio": round(min(ratios), 4) if ratios else 1.0,
            "max_ratio": round(max(ratios), 4) if ratios else 1.0,
            "ratio_gap": round(max(ratios) - min(ratios), 4) if ratios else 0.0,
            "calibrated_threshold": round(threshold, 4) if threshold else None,
            "block_detail": block_detail,
        }
    return metrics


def collect_epoch_metrics(model, cog, task, epoch, val_acc, elapsed_s):
    """Collect full epoch metrics for JSON logging."""
    layer_metrics = collect_layer_metrics(model)

    # Aggregate across layers
    salient_total = sum(m["salient_blocks"] for m in layer_metrics.values())
    blocks_total = sum(m["total_blocks"] for m in layer_metrics.values())
    novelty_vals = [m["avg_novelty"] for m in layer_metrics.values()]
    ratio_gaps = [m["ratio_gap"] for m in layer_metrics.values()]

    state = cog._task_state.get(task)
    zstd_gated = state.zstd_gated_count if state else 0
    zstd_trained = state.zstd_trained_count if state else 0
    steps = state.steps_taken if state else 0

    # WeightDict info
    wd = getattr(model, "weight_dict", None)
    wd_entries = wd.num_entries if wd else 0

    return {
        "phase": task,
        "epoch": epoch,
        "val_acc": round(val_acc, 2),
        "elapsed_s": int(elapsed_s),
        "weight_dict_entries": wd_entries,
        "aggregate": {
            "salient_blocks": f"{salient_total}/{blocks_total}",
            "salient_pct": round(100.0 * salient_total / blocks_total, 1) if blocks_total else 0,
            "avg_novelty": round(sum(novelty_vals) / len(novelty_vals), 4) if novelty_vals else 1.0,
            "avg_ratio_gap": round(sum(ratio_gaps) / len(ratio_gaps), 4) if ratio_gaps else 0.0,
        },
        "zstd_gating": {
            "gated": zstd_gated,
            "trained": zstd_trained,
            "gated_pct": round(100.0 * zstd_gated / max(zstd_gated + zstd_trained, 1), 1),
        },
        "steps_taken": steps,
        "per_layer": layer_metrics,
    }


# ═══════════════════════════════════════════════════════════════
# Build model in zpackr mode
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42)

num_labels = 3
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
    local_files_only=True,
)
model.gradient_checkpointing_enable()

config = PackRConfig(
    mode="zpackr",
    layer_scope="ffn",
    gradient_checkpointing=True,
    zstd_salience_threshold=1.4,
)
model = compress_model(model, config)
model.cuda()

head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and ("classifier" in n or "cls" in n)]
body_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "classifier" not in n and "cls" not in n]

opt = FusedQuantizedAdam([
    {"params": body_params, "lr": 2e-5},
    {"params": head_params, "lr": 1e-3},
], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block_size=256)

stream = StreamTrainer(model, opt, cv2lrt=None, acc_steps=4)
sup = model.super_zstd
cog = Cogitator(stream, super_zstd=sup, zstd_gate_threshold=1.5)

log_path = os.path.join(OUT_DIR, "training_log.jsonl")

print(f"\n  Output: {OUT_DIR}")
print(f"  Config: mode={config.mode}, threshold={config.zstd_salience_threshold}")

# ═══════════════════════════════════════════════════════════════
# Phase 1: SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 1 — SST-2")
print("=" * 60)
n = cog.ingest_glue("sst2", limit=None, seed=42)
print(f"  {n} prompts ingested")

EPOCHS = 3
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\n  --- Epoch {epoch}/{EPOCHS} ---")
    decay_all(model)
    cog.cogitate("sst2", max_epochs=1, use_zstd_gating=True)
    post_step_all(model)

    acc = validate(model, "sst2")
    elapsed = time.time() - t0
    metrics = collect_epoch_metrics(model, cog, "sst2", epoch, acc, elapsed)
    _log(log_path, metrics)

    agg = metrics["aggregate"]
    print(f"  val acc: {acc:.2f}%  novelty: {agg['avg_novelty']:.3f}  "
          f"salient: {agg['salient_blocks']}  gap: {agg['avg_ratio_gap']:.3f}")

val_sst2_1 = validate(stream.model, "sst2")
print(f"\n  SST-2 final: {val_sst2_1:.2f}%")


# ═══════════════════════════════════════════════════════════════
# Phase 2: MNLI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 2 — MNLI")
print("=" * 60)
n = cog.ingest_glue("mnli", limit=None, seed=42)
print(f"  {n} prompts ingested")

for epoch in range(1, EPOCHS + 1):
    print(f"\n  --- Epoch {epoch}/{EPOCHS} ---")
    decay_all(model)
    cog.cogitate("mnli", max_epochs=1, use_zstd_gating=True)
    post_step_all(model)

    acc = validate(model, "mnli")
    elapsed = time.time() - t0
    metrics = collect_epoch_metrics(model, cog, "mnli", epoch, acc, elapsed)
    _log(log_path, metrics)

    agg = metrics["aggregate"]
    print(f"  val acc: {acc:.2f}%  novelty: {agg['avg_novelty']:.3f}  "
          f"salient: {agg['salient_blocks']}  gap: {agg['avg_ratio_gap']:.3f}")

val_mnli = validate(stream.model, "mnli")
print(f"\n  MNLI final: {val_mnli:.2f}%")


# ═══════════════════════════════════════════════════════════════
# Phase 3: Re-validate SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 3 — Re-validate SST-2 (no retraining)")
print("=" * 60)

val_sst2_2 = validate(stream.model, "sst2")
delta = val_sst2_2 - val_sst2_1

# Final metrics snapshot
final_metrics = collect_epoch_metrics(model, cog, "sst2", EPOCHS, val_sst2_2, time.time() - t0)
final_metrics["phase"] = "final"

print(f"\n  SST-2 after Phase 1:  {val_sst2_1:.2f}%")
print(f"  SST-2 after Phase 3:  {val_sst2_2:.2f}%")
print(f"  Delta:                {delta:+.2f}%")
print(f"  MNLI:                 {val_mnli:.2f}%")

# ── Save summary ──
summary = {
    "run_id": RUN_ID,
    "config": {
        "mode": config.mode,
        "salience_threshold": config.zstd_salience_threshold,
        "epochs_per_task": EPOCHS,
        "zstd_gate_threshold": cog.zstd_gate_threshold,
        "body_lr": 2e-5,
        "head_lr": 1e-3,
    },
    "results": {
        "sst2_phase1": round(val_sst2_1, 2),
        "sst2_phase3": round(val_sst2_2, 2),
        "sst2_delta": round(delta, 2),
        "mnli": round(val_mnli, 2),
    },
    "final_layer_metrics": final_metrics["per_layer"],
    "aggregate": final_metrics["aggregate"],
    "zstd_gating": final_metrics["zstd_gating"],
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# HTML-safe version of the training log tabulated
print(f"\n  Saved to: {OUT_DIR}")
print(f"  Files: training_log.jsonl, summary.json")

if abs(delta) < 1.0 and val_mnli > 50.0:
    print(f"\n  PASS — SST-2 preserved ({abs(delta):.2f}% < 1%), MNLI learned ({val_mnli:.1f}%)")
else:
    print(f"\n  Needs investigation — delta={delta:+.2f}%, MNLI={val_mnli:.1f}%")
