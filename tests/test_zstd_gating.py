"""StreamCC zstd-native continuous learning test.

ZPackR (mode="zpackr") replaces nn.Linear with frozen base + WeightDict-compressed
delta.  Block-level compression ratios serve as the convergence signal.

Parameters tuned per ZPackR dev machine best results:
  --post-step-interval 4     run post_step every 4 optimizer steps
  --reindex-interval 4000    rebuild WeightDict every 4000 steps
  --calibration-multiplier 0.01
  --gate --gate-threshold 1.5
  --no-velvet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
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
from packr.zpackr_layer import ZPackRLinear

from streamcc.stream import StreamTrainer
from streamcc.cogitator import Cogitator


# ── Tunables (matching ZPackR dev machine best results) ──
POST_STEP_INTERVAL = 4       # run post_step every N optimizer steps
REINDEX_INTERVAL = 4000      # rebuild WeightDict every N optimizer steps
CALIBRATION_MULTIPLIER = 0.01
GATE_ENABLED = True
GATE_THRESHOLD = 1.5
EPOCHS_PER_TASK = 3

# ── Output ──
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", f"zstd_gating_{RUN_ID}")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Shared validation ──
tok = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)


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


def _make_hooks(model, cog, log_path):
    """Create the post-optimizer-step hook with correct interval gating."""

    def post_opt_step(global_step: int):
        # shrink_known_delta every step (lightweight, in-place on GPU)
        for layer in _zpackr_layers(model):
            layer.shrink_known_delta()

        # post_step every POST_STEP_INTERVAL optimizer steps
        if global_step % POST_STEP_INTERVAL == 0:
            for layer in _zpackr_layers(model):
                layer.post_step(calibration_multiplier=CALIBRATION_MULTIPLIER)

        # reindex WeightDict every REINDEX_INTERVAL steps
        if global_step % REINDEX_INTERVAL == 0:
            for layer in _zpackr_layers(model):
                layer.reindex()

        # Log metrics periodically
        if global_step % 1000 == 0:
            acc = validate(model, cog._current_task if hasattr(cog, '_current_task') else "sst2")
            metrics = collect_step_metrics(model, cog, global_step, acc)
            _log(log_path, metrics)
            agg = metrics["aggregate"]
            print(f"  step {global_step:06d}  val_acc={acc:.2f}%  "
                  f"novelty={agg['avg_novelty']:.3f}  salient={agg['salient_blocks']}")

    return post_opt_step


# ── Metrics ──

def _log(path, entry):
    entry["timestamp"] = datetime.now().isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def collect_layer_metrics(model):
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
        metrics[f"layer_{i}"] = {
            "salient": f"{kept}/{total}",
            "salient_pct": round(100.0 * kept / total, 1) if total else 0,
            "avg_novelty": round(sum(scores) / len(scores), 4) if scores else 1.0,
            "min_novelty": round(min(scores), 4) if scores else 1.0,
            "max_novelty": round(max(scores), 4) if scores else 1.0,
            "avg_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 1.0,
            "ratio_gap": round(max(ratios) - min(ratios), 4) if ratios else 0.0,
            "threshold": round(threshold, 4) if threshold else None,
        }
    return metrics


def collect_step_metrics(model, cog, step, val_acc):
    layer_m = collect_layer_metrics(model)
    salient_t = sum(int(m["salient"].split("/")[0]) for m in layer_m.values())
    blocks_t = sum(int(m["salient"].split("/")[1]) for m in layer_m.values())
    novelty_vals = [m["avg_novelty"] for m in layer_m.values()]
    wd = getattr(model, "weight_dict", None)
    return {
        "step": step,
        "val_acc": round(val_acc, 2),
        "weight_dict_entries": wd.num_entries if wd else 0,
        "aggregate": {
            "salient_blocks": f"{salient_t}/{blocks_t}",
            "salient_pct": round(100.0 * salient_t / blocks_t, 1) if blocks_t else 0,
            "avg_novelty": round(sum(novelty_vals) / len(novelty_vals), 4) if novelty_vals else 1.0,
        },
        "per_layer": layer_m,
    }


# ═══════════════════════════════════════════════════════════════
# Build model
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

log_path = os.path.join(OUT_DIR, "training_log.jsonl")
sup = model.super_zstd

stream = StreamTrainer(model, opt, cv2lrt=None, acc_steps=4)
cog = Cogitator(stream, super_zstd=sup, zstd_gate_threshold=GATE_THRESHOLD, batch_size=16)
cog._current_task = "sst2"  # for early logging

# Attach the step-level hook
stream._post_opt_step_fn = _make_hooks(model, cog, log_path)


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


print(f"\n  Output: {OUT_DIR}")
print(f"  Config: post_step_interval={POST_STEP_INTERVAL} reindex={REINDEX_INTERVAL} "
      f"calibration={CALIBRATION_MULTIPLIER} gate={GATE_ENABLED}({GATE_THRESHOLD})")

# ═══════════════════════════════════════════════════════════════
# Phase 1: SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 1 — SST-2")
print("=" * 60)
n = cog.ingest_glue("sst2", limit=None, seed=42)
print(f"  {n} prompts ingested")

cog._current_task = "sst2"
t0 = time.time()

cog.cogitate("sst2", max_epochs=EPOCHS_PER_TASK)

val_sst2_1 = validate(stream.model, "sst2")
print(f"\n  SST-2 final: {val_sst2_1:.2f}%  ({time.time() - t0:.0f}s)")

# ── Cross-domain transfer check (pre-MNLI) ──
val_mnli_pre = validate(stream.model, "mnli")
transfer_gain = val_mnli_pre - 33.33  # baseline = random (3-class)
print(f"  MNLI (pre-train, SST-2 only): {val_mnli_pre:.2f}%  (+{transfer_gain:+.1f}% from SST-2)")


# ═══════════════════════════════════════════════════════════════
# Phase 2: MNLI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 2 — MNLI")
print("=" * 60)
n = cog.ingest_glue("mnli", limit=None, seed=42)
print(f"  {n} prompts ingested")

cog._current_task = "mnli"

cog.cogitate("mnli", max_epochs=EPOCHS_PER_TASK)

val_mnli = validate(stream.model, "mnli")
print(f"\n  MNLI final: {val_mnli:.2f}%  ({time.time() - t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════
# Phase 3: Re-validate SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 3 — Re-validate SST-2")
print("=" * 60)

val_sst2_2 = validate(stream.model, "sst2")
delta = val_sst2_2 - val_sst2_1

print(f"\n  SST-2 Phase 1: {val_sst2_1:.2f}%")
print(f"  MNLI pre-train: {val_mnli_pre:.2f}%  (+{transfer_gain:+.1f}% from SST-2)")
print(f"  SST-2 Phase 3: {val_sst2_2:.2f}%")
print(f"  Delta:         {delta:+.2f}%")
print(f"  MNLI:          {val_mnli:.2f}%")

summary = {
    "run_id": RUN_ID,
    "config": {
        "post_step_interval": POST_STEP_INTERVAL,
        "reindex_interval": REINDEX_INTERVAL,
        "calibration_multiplier": CALIBRATION_MULTIPLIER,
        "gate_enabled": GATE_ENABLED,
        "gate_threshold": GATE_THRESHOLD,
        "epochs_per_task": EPOCHS_PER_TASK,
        "mode": "zpackr",
    },
    "results": {
        "sst2_phase1": round(val_sst2_1, 2),
        "sst2_phase3": round(val_sst2_2, 2),
        "sst2_delta": round(delta, 2),
        "mnli_pre_train": round(val_mnli_pre, 2),
        "mnli_transfer_gain": round(transfer_gain, 1),
        "mnli": round(val_mnli, 2),
    },
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved: {OUT_DIR}")

if abs(delta) < 1.0 and val_mnli > 50.0:
    print(f"  PASS — SST-2 preserved ({abs(delta):.2f}% < 1%), MNLI learned ({val_mnli:.1f}%)")
else:
    print(f"  Needs investigation")
