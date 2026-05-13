"""StreamCC zstd-gated continuous learning test.

Phase 1: Cogitate on SST-2 → zstd gate filters, CV2LRT converges.
Phase 2: Cogitate on MNLI → zstd gate protects SST-2 patterns.
Phase 3: Re-validate SST-2 → accuracy preserved.
"""

import torch
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset

from packr import VelvetController
from packr.super_dict import load_super_dict
from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam

from streamcc.stream import StreamTrainer
from streamcc.cogitator import Cogitator
from streamcc.prompt import ingest_glue


# ── Shared validation ──
tok = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)


def _val_loader(task: str, split: str):
    ds_path = f"glue/{task}"
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


# ── Build model in zpackr mode ──
torch.manual_seed(42)

num_labels = 3  # max across both tasks (MNLI has 3)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
    local_files_only=True,
)
model.gradient_checkpointing_enable()

config = PackRConfig(
    scheme="phr",
    layer_scope="ffn",
    learnable_lut=True,
    gradient_checkpointing=True,
)
model = compress_model(model, config)
model.cuda()

# Separate params for differential LR
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and ("classifier" in n or "cls" in n)]
body_params = [p for n, p in model.named_parameters()
               if p.requires_grad and "classifier" not in n and "cls" not in n]

opt = FusedQuantizedAdam([
    {"params": body_params, "lr": 2e-5},
    {"params": head_params, "lr": 1e-3},
], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block_size=256)

velvet = VelvetController(opt, beta=0.97, min_multiplier=0.175,
                          max_multiplier=1.0, velocity_scale=10.0)

stream = StreamTrainer(model, opt, velvet, acc_steps=4)

# Load SuperDict for zstd gating
sup = load_super_dict()
cog = Cogitator(stream, super_zstd=sup, zstd_gate_threshold=1.5)


def _val_fn(task_name):
    def _fn(trainer):
        acc = validate(trainer.model, task_name)
        print(f"  -- {task_name} val acc {acc:.2f}%")
        return acc
    return _fn


# ═══════════════════════════════════════════════════════════════
# Phase 1: Learn SST-2
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 1 — Ingest SST-2")
print("=" * 60)
n = cog.ingest_glue("sst2", limit=None, seed=42)
print(f"  {n} training prompts ingested")

print("\n" + "=" * 60)
print("  Phase 1 — Cogitate SST-2 (3 epochs, zstd gating)")
print("=" * 60)

total_micro = len(cog._prompts["sst2"]) * 3
warmup = int(0.02 * total_micro)

cog.cogitate("sst2", max_epochs=3, warmup_steps=warmup,
             val_fn=_val_fn("sst2"), use_zstd_gating=True)

val_sst2_1 = validate(stream.model, "sst2")
print(f"\n  SST-2 final validation: {val_sst2_1:.2f}%")
print(f"  Converged: {'sst2' in cog.converged_tasks}")
print(cog.task_summary())


# ═══════════════════════════════════════════════════════════════
# Phase 2: Learn MNLI
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Phase 2 — Ingest MNLI")
print("=" * 60)
n = cog.ingest_glue("mnli", limit=None, seed=42)
print(f"  {n} training prompts ingested")

print("\n" + "=" * 60)
print("  Phase 2 — Cogitate MNLI (3 epochs, zstd gating)")
print("=" * 60)

total_micro = len(cog._prompts["mnli"]) * 3
warmup = int(0.02 * total_micro)

cog.cogitate("mnli", max_epochs=3, warmup_steps=warmup,
             val_fn=_val_fn("mnli"), use_zstd_gating=True)

val_mnli = validate(stream.model, "mnli")
print(f"\n  MNLI final validation: {val_mnli:.2f}%")
print(f"  Converged: {'mnli' in cog.converged_tasks}")


# ═══════════════════════════════════════════════════════════════
# Phase 3: Verify SST-2 survival
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

print(cog.task_summary())

if abs(delta) < 1.0 and val_mnli > 50.0:
    print(f"\n  PASS — SST-2 preserved ({abs(delta):.2f}% < 1%), MNLI learned ({val_mnli:.1f}%)")
else:
    print(f"\n  Needs investigation — delta={delta:+.2f}%, MNLI={val_mnli:.1f}%")

# Final CV2LRT state
stats = velvet.get_stats()
print(f"\n  Velvet final state:")
for name, g in stats.get("per_group", {}).items():
    if isinstance(g, dict):
        short = name.replace("bert.encoder.layer.", "L")
        print(f"    {short[:35]:35s} multiplier={g.get('multiplier', 0):.4f}  lr={g.get('current_lr', 0):.2e}")
