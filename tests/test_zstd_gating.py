"""StreamCC zstd-native continuous learning test.

ZPackR (mode="zpackr") replaces nn.Linear with frozen base + WeightDict-compressed
delta.  Block-level compression ratios against the WeightDict serve as the
authoritative convergence signal — no Velvet scheduler needed.

Known blocks (high ratio → low novelty): attenuated in forward, decayed over
time, pruned from VRAM below auto-calibrated threshold.
Novel blocks (low ratio → high novelty): kept at full strength, full LR.
"""

import torch
import torch.nn as nn
import os
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

from streamcc.stream import StreamTrainer
from streamcc.cogitator import Cogitator


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
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            correct += (outputs.logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


# ── ZPackR layer helpers ──

def _zpackr_layers(model):
    """Return all ZPackRLinear layers in the model."""
    from packr.zpackr_layer import ZPackRLinear
    return [m for m in model.modules() if isinstance(m, ZPackRLinear)]


def post_step_all(model):
    """Run post_step() on all ZPackRLinear layers after an optimizer step."""
    for layer in _zpackr_layers(model):
        layer.post_step()


def decay_all(model):
    """Decay known blocks toward zero on all ZPackRLinear layers."""
    for layer in _zpackr_layers(model):
        layer.decay_delta()


def novelty_summary(model):
    """Return per-layer novelty stats for logging."""
    stats = {}
    for i, layer in enumerate(_zpackr_layers(model)):
        info = layer.get_block_ratios()
        if info:
            scores = info.get("novelty_scores", [])
            kept = info.get("salient_count", 0)
            total = info.get("num_blocks", 0)
            avg_novelty = sum(scores) / len(scores) if scores else 1.0
            stats[f"layer_{i}"] = {
                "salient": f"{kept}/{total}",
                "avg_novelty": round(avg_novelty, 3),
            }
    return stats


# ═══════════════════════════════════════════════════════════════
# Build model in zpackr mode (frozen base + WeightDict delta)
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42)

num_labels = 3  # max across both tasks (MNLI has 3)
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

# Fixed LR — no scheduler.  WeightDict handles everything.
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and ("classifier" in n or "cls" in n)]
body_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "classifier" not in n and "cls" not in n]

opt = FusedQuantizedAdam([
    {"params": body_params, "lr": 2e-5},
    {"params": head_params, "lr": 1e-3},
], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block_size=256)

stream = StreamTrainer(model, opt, cv2lrt=None, acc_steps=4)

# SuperDict for optional prompt-level pre-filter
sup = model.super_zstd
cog = Cogitator(stream, super_zstd=sup, zstd_gate_threshold=1.5)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Learn SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 1 — Ingest SST-2")
print("=" * 60)
n = cog.ingest_glue("sst2", limit=None, seed=42)
print(f"  {n} training prompts ingested")

EPOCHS = 3

print("\n" + "=" * 60)
print("  Phase 1 — Cogitate SST-2 (zstd-native, no Velvet)")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    print(f"\n  --- Epoch {epoch}/{EPOCHS} ---")

    # Decay known blocks before forward passes
    decay_all(model)

    cog.cogitate("sst2", max_epochs=1, use_zstd_gating=True)

    # Block-level: compress delta vs WeightDict → update novelty
    post_step_all(model)

    acc = validate(model, "sst2")
    nv = novelty_summary(model)
    avg_nv = sum(v["avg_novelty"] for v in nv.values()) / len(nv) if nv else 0
    print(f"  SST-2 val acc: {acc:.2f}%  avg novelty: {avg_nv:.3f}")
    for name, s in list(nv.items())[:5]:
        print(f"    {name}: salient={s['salient']}  novelty={s['avg_novelty']}")

val_sst2_1 = validate(stream.model, "sst2")
print(f"\n  SST-2 final: {val_sst2_1:.2f}%")


# ═══════════════════════════════════════════════════════════════
# Phase 2: Learn MNLI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 2 — Ingest MNLI")
print("=" * 60)
n = cog.ingest_glue("mnli", limit=None, seed=42)
print(f"  {n} training prompts ingested")

print("\n" + "=" * 60)
print("  Phase 2 — Cogitate MNLI (zstd-native, no Velvet)")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    print(f"\n  --- Epoch {epoch}/{EPOCHS} ---")
    decay_all(model)
    cog.cogitate("mnli", max_epochs=1, use_zstd_gating=True)
    post_step_all(model)

    acc = validate(model, "mnli")
    nv = novelty_summary(model)
    avg_nv = sum(v["avg_novelty"] for v in nv.values()) / len(nv) if nv else 0
    print(f"  MNLI val acc: {acc:.2f}%  avg novelty: {avg_nv:.3f}")

val_mnli = validate(stream.model, "mnli")
print(f"\n  MNLI final: {val_mnli:.2f}%")


# ═══════════════════════════════════════════════════════════════
# Phase 3: Re-validate SST-2 (no retraining, known blocks should
#          still be attenuated/decayed — test survival)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 3 — Re-validate SST-2 (no retraining)")
print("=" * 60)

val_sst2_2 = validate(stream.model, "sst2")
delta = val_sst2_2 - val_sst2_1

print(f"\n  SST-2 after Phase 1:  {val_sst2_1:.2f}%")
print(f"  SST-2 after Phase 3:  {val_sst2_2:.2f}%")
print(f"  Delta:                {delta:+.2f}%")
print(f"  MNLI:                 {val_mnli:.2f}%")

nv = novelty_summary(model)
print(f"\n  Final novelty:")
for name, s in list(nv.items())[:5]:
    print(f"    {name}: salient={s['salient']}  novelty={s['avg_novelty']}")

if abs(delta) < 1.0 and val_mnli > 50.0:
    print(f"\n  PASS — SST-2 preserved ({abs(delta):.2f}% < 1%), MNLI learned ({val_mnli:.1f}%)")
else:
    print(f"\n  Needs investigation — delta={delta:+.2f}%, MNLI={val_mnli:.1f}%")
