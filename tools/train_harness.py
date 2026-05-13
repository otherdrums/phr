"""ZPackR Training Harness — drop-in trainer for GLUE tasks with full instrumentation.

Records per-step metrics (loss, super ratio, salience, weight ratios, VRAM,
Velvet multipliers, gate stats) to JSON Lines for analysis and ablation.

Usage:
    from tools.train_harness import ZPackRTrainer, TrainerConfig

    config = TrainerConfig(
        model_name="bert-base-uncased",
        task_name="sst2",
        packr_config=PackRConfig(mode="zpackr"),
        max_steps=2000,
        output_dir="runs/sst2_zpackr",
    )
    trainer = ZPackRTrainer(config)
    results = trainer.run()
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal

import torch
import torch.nn as nn
import numpy as np

# Ensure packr is importable from tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from packr.config import PackRConfig
from packr.layer_patcher import compress_model
from packr.optim import FusedQuantizedAdam
from packr.velvet import VelvetController
from zpackr.prompt_gate import should_train
from zpackr.zpackr_layer import ZPackRLinear


def _git_commit_short():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _make_output_dir(base: str, label: str):
    commit = _git_commit_short()
    ts = _timestamp()
    dirname = f"{ts}_{commit}"
    if label:
        dirname = f"{label}_{dirname}"
    path = os.path.join(base, dirname)
    os.makedirs(path, exist_ok=True)
    return path


# ── GLUE task metadata ──

GLUE_TASKS = {
    "sst2": {"num_labels": 2, "keys": ("sentence", None), "metric": "accuracy"},
    "mnli": {"num_labels": 3, "keys": ("premise", "hypothesis"), "metric": "accuracy"},
    "qnli": {"num_labels": 2, "keys": ("question", "sentence"), "metric": "accuracy"},
    "qqp":  {"num_labels": 2, "keys": ("question1", "question2"), "metric": "accuracy"},
    "rte":  {"num_labels": 2, "keys": ("sentence1", "sentence2"), "metric": "accuracy"},
    "mrpc": {"num_labels": 2, "keys": ("sentence1", "sentence2"), "metric": "accuracy"},
    "cola": {"num_labels": 2, "keys": ("sentence", None), "metric": "matthews_correlation"},
    "stsb": {"num_labels": 1, "keys": ("sentence1", "sentence2"), "metric": "pearson"},
}


# ── Configuration ──

@dataclass
class TrainerConfig:
    """Full training configuration with all tunables exposed."""

    # Task
    model_name: str = "bert-base-uncased"
    task_name: str = "sst2"
    num_labels: Optional[int] = None

    # PackR
    packr_config: PackRConfig = field(default_factory=PackRConfig)

    # Optimization
    lr: float = 2e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    batch_size: int = 16
    max_steps: int = 10000
    grad_accum_steps: int = 1
    max_seq_length: int = 128

    # Velvet
    velvet_enabled: bool = True
    velvet_beta: float = 0.97
    velvet_min_multiplier: float = 0.175
    velvet_max_multiplier: float = 1.0
    velvet_velocity_scale: float = 10.0
    warmup_steps: int = 0

    # Gate
    gate_enabled: bool = True
    gate_threshold: float = 2.0
    gate_skip_forward: bool = False

    # ZPackR
    post_step_interval: int = 4
    reindex_interval: int = 1000

    # Evaluation
    eval_interval: int = 500
    eval_steps: int = 20

    # Checkpoint
    checkpoint_interval: int = 2000

    # Output
    output_dir: str = "runs"
    run_label: str = ""
    seed: int = 42

    def __post_init__(self):
        if self.num_labels is None and self.task_name in GLUE_TASKS:
            self.num_labels = GLUE_TASKS[self.task_name]["num_labels"]


# ── Trainer ──

class ZPackRTrainer:
    """Drop-in trainer for GLUE tasks with full ZPackR instrumentation.

    Records per-step metrics to metrics.jsonl in the output directory.
    Supports both packr and zpackr modes, Velvet, prompt gating,
    checkpointing, and structured ablation runs.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = _make_output_dir(config.output_dir, config.run_label)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._metrics_file = open(os.path.join(self.output_dir, "metrics.jsonl"), "w")
        self._step = 0
        self._global_step = 0
        self._start_time = None
        self._model = None
        self._optimizer = None
        self._velvet = None
        self._tokenizer = None
        self._train_loader = None
        self._eval_dataset = None
        self._metric = None
        self._scaler = None  # for amp
        self._ephemeral = {}  # per-run metrics accumulator
        self._gate_skipped_total = 0
        self._gate_total = 0
        self._zpl_layers = None  # cached list of ZPackRLinear instances
        self._peak_vram = 0      # max VRAM seen during run
        self._gate_cache = {}     # prompt_hash → ratio (LRU, max 1024)
        self._metrics_buffer = [] # batched flush every N steps

        self._log_config()

    def _log_config(self):
        cfg = asdict(self.config)
        cfg["packr_config"] = {
            k: str(v) if k == "scheme" else v
            for k, v in asdict(self.config.packr_config).items()
        }
        cfg["git_commit"] = _git_commit_short()
        cfg["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cfg["gpu_name"] = torch.cuda.get_device_name(0)
            cfg["gpu_arch"] = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    # ── Setup ──

    def setup(self):
        self._log("Setting up model, tokenizer, dataset ...")
        torch.manual_seed(self.config.seed)

        # Suppress expected startup warnings
        import os as _os
        _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # Tokenizer & model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=self.config.num_labels,
        )

        # Compress
        self._log(f"Compressing model (mode={self.config.packr_config.mode}) ...")
        self._model = compress_model(self._model, self.config.packr_config)
        self._model = self._model.to(self.device)

        # Cache ZPackRLinear layers to avoid walking named_modules every step
        if self.config.packr_config.mode == "zpackr":
            from zpackr.zpackr_layer import ZPackRLinear
            self._zpl_layers = [
                (name.replace("bert.encoder.", "enc."), m)
                for name, m in self._model.named_modules()
                if isinstance(m, ZPackRLinear)
            ]
            # WeightDict already seeded from BERT base weights during from_linear().
            # Collect all layer base weights and do ONE combined reindex for the best dict.
            self._log("  Building WeightDict from BERT base weights (all layers) ...")
            all_base_bytes = []
            for _, module in self._zpl_layers:
                wb = module.base_W.cpu().view(torch.uint8).contiguous().view(-1).numpy().tobytes()
                all_base_bytes.append(wb)
            combined_base = b"".join(all_base_bytes)
            # Cache BERT base patterns — they persist across all future reindex calls
            self._model.weight_dict.set_base_samples(combined_base)
            self._model.weight_dict.reindex(combined_base, min_frequency=0.001, min_count=3)
            self._log(f"  WeightDict: {self._model.weight_dict.num_entries} entries from {len(self._zpl_layers)} layers (base cached)")

        # Dataset
        from datasets import load_dataset
        from datasets import logging as ds_logging
        ds_logging.set_verbosity_error()
        task_info = GLUE_TASKS[self.config.task_name]

        raw_dataset = load_dataset("glue", self.config.task_name)
        train_dataset = raw_dataset["train"]

        self._eval_dataset = raw_dataset.get(
            "validation", raw_dataset.get("validation_matched", raw_dataset["train"])
        )

        self._tokenize = self._make_tokenize_fn(task_info["keys"])

        train_dataset = train_dataset.map(
            self._tokenize, batched=True,
            remove_columns=[c for c in train_dataset.column_names if c not in ("label",)]
        )
        self._eval_dataset = self._eval_dataset.map(
            self._tokenize, batched=True,
            remove_columns=[c for c in self._eval_dataset.column_names if c not in ("label",)]
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        self._eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            drop_last=True,
        )

        # Optimizer
        self._optimizer = FusedQuantizedAdam(
            self._model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            block_size=self.config.packr_config.block_size,
        )

        # Velvet
        if self.config.velvet_enabled:
            self._velvet = VelvetController(
                self._optimizer,
                beta=self.config.velvet_beta,
                min_multiplier=self.config.velvet_min_multiplier,
                max_multiplier=self.config.velvet_max_multiplier,
                velocity_scale=self.config.velvet_velocity_scale,
            )

        # Metric
        import evaluate
        self._metric = evaluate.load("glue", self.config.task_name)

        self._log(f"Setup complete.  Output: {self.output_dir}")

    def _make_tokenize_fn(self, keys):
        tokenizer = self._tokenizer
        max_length = self.config.max_seq_length

        def tokenize(examples):
            key1, key2 = keys
            if key2:
                return tokenizer(
                    examples[key1], examples[key2],
                    truncation=True, padding="max_length",
                    max_length=max_length,
                )
            else:
                return tokenizer(
                    examples[key1],
                    truncation=True, padding="max_length",
                    max_length=max_length,
                )
        return tokenize

    # ── Run ──

    def run(self) -> dict:
        self.setup()
        self._start_time = time.perf_counter()
        self._log(f"Starting training ({self.config.max_steps} steps) ...")

        self._model.train()
        train_iter = iter(self._train_loader)

        while self._global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self._train_loader)
                batch = next(train_iter)

            step_start = time.perf_counter()

            # ── Gate: skip-forward mode ──
            if self.config.gate_enabled and self.config.gate_skip_forward:
                if self.config.packr_config.mode == "zpackr":
                    sup = getattr(self._model, "super_zstd", None)
                    if sup is not None:
                        prompt_text = self._tokenizer.decode(
                            batch["input_ids"][0], skip_special_tokens=True
                        )
                        self._gate_total += 1
                        ratio = self._cached_compress(prompt_text.encode("utf-8"), sup)
                        if ratio >= self.config.gate_threshold:
                            self._gate_skipped_total += 1
                            self._global_step += 1
                            self._record_step({"step": self._global_step, "gate_skipped": True})
                            continue

            # ── Forward ──
            labels = batch.pop("label", None)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)

            outputs = self._model(**batch, labels=labels)
            loss = outputs.loss

            loss = loss / self.config.grad_accum_steps

            # ── Gate: skip-backward mode ──
            gate_skipped = False
            if self.config.gate_enabled and not self.config.gate_skip_forward:
                if self.config.packr_config.mode == "zpackr":
                    sup = getattr(self._model, "super_zstd", None)
                    if sup is not None:
                        prompt_text = self._tokenizer.decode(
                            batch["input_ids"][0], skip_special_tokens=True
                        )
                        self._gate_total += 1
                        ratio = self._cached_compress(prompt_text.encode("utf-8"), sup)
                        if ratio >= self.config.gate_threshold:
                            gate_skipped = True
                            self._gate_skipped_total += 1

            if not gate_skipped:
                loss.backward()

                if (self._global_step + 1) % self.config.grad_accum_steps == 0:
                    self._optimizer.step()

                    # Warmup
                    if self.config.warmup_steps > 0 and self._velvet is not None:
                        if self._global_step < self.config.warmup_steps:
                            self._velvet.warmup_step(self._global_step, self.config.warmup_steps)

                    # Velvet
                    if self._velvet is not None:
                        self._velvet.step()

                    # ZPackR post_step
                    if self._zpl_layers is not None:
                        if (self._global_step + 1) % self.config.post_step_interval == 0:
                            for _, module in self._zpl_layers:
                                module.post_step(calibration_multiplier=self.config.packr_config.zstd_calibration_multiplier)

                    self._optimizer.zero_grad()

                    # Decay known blocks toward zero (post-optimizer, pre-next-forward)
                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            if module._gap_enabled:
                                module.shrink_known_delta()

                    # Pre-stage delta for next post_step (D2D + CPU copy overlaps forward)
                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            module.stage_delta_async(None)  # TODO: dedicated stream

            # ── Record step ──
            step_ms = (time.perf_counter() - step_start) * 1000
            self._record_step(self._gather_metrics(loss.item() * self.config.grad_accum_steps, step_ms, gate_skipped))

            # ── Reindex ──
            if self._zpl_layers is not None:
                if (self._global_step + 1) % self.config.reindex_interval == 0:
                    self._run_reindex()

            # ── Eval ──
            if (self._global_step + 1) % self.config.eval_interval == 0:
                self._run_eval()

            # ── Checkpoint ──
            if (self._global_step + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self._global_step += 1

        # Final eval
        self._run_eval()
        self._save_summary()
        self._metrics_file.close()

        elapsed = time.perf_counter() - self._start_time
        self._log(f"Training complete in {elapsed:.1f}s.  Results: {self.output_dir}")
        return self._ephemeral

    # ── Metrics ──

    def _gather_metrics(self, loss: float, step_ms: float, gate_skipped: bool) -> dict:
        metrics = {
            "step": self._global_step + 1,
            "loss": loss,
            "step_ms": step_ms,
            "gate_skipped": gate_skipped,
        }

        # Velvet multipliers
        if self._velvet is not None:
            try:
                stats = self._velvet.get_stats()
                multipliers = {}
                for gname, ginfo in stats.get("per_group", {}).items():
                    multipliers[gname] = round(ginfo.get("multiplier", 1.0), 4)
                metrics["velvet_multipliers"] = multipliers
                metrics["velvet_max_mult"] = max(multipliers.values()) if multipliers else 0
                metrics["velvet_min_mult"] = min(multipliers.values()) if multipliers else 0
            except Exception:
                pass

        # ZPackR salience + weight ratios
        if self._zpl_layers is not None:
            salience = {}
            total_salient_kb = 0
            total_capacity_kb = 0
            thresholds = {}
            for short_name, module in self._zpl_layers:
                kept = module.salient_count
                total = module.num_blocks
                salience[short_name] = {"kept": kept, "total": total, "fraction": round(kept / max(total, 1), 3)}
                total_salient_kb += kept * module.block_size * module.out_features * 2 / 1024
                total_capacity_kb += total * module.block_size * module.out_features * 2 / 1024
                t = module.salience_threshold
                if t is not None:
                    thresholds[short_name] = round(t, 4)
            if salience:
                metrics["salience"] = salience
                metrics["salient_vram_kb"] = round(total_salient_kb, 0)
                metrics["salient_vram_fraction"] = round(total_salient_kb / max(total_capacity_kb, 1), 3)
            if thresholds:
                metrics["salience_thresholds"] = thresholds

            wd = getattr(self._model, "weight_dict", None)
            if wd is not None:
                metrics["weight_dict_entries"] = wd.num_entries

        # VRAM
        if self.device.type == "cuda":
            metrics["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            metrics["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            torch.cuda.reset_peak_memory_stats()

        return metrics

    def _record_step(self, data: dict):
        data["type"] = "step"
        self._metrics_buffer.append(json.dumps(data))
        # Flush every 10 steps (events flush immediately)
        if len(self._metrics_buffer) >= 10:
            self._flush_metrics()
        # Track peak VRAM across the run
        peak = data.get("vram_peak_mb", 0)
        if peak > self._peak_vram:
            self._peak_vram = peak

    def _flush_metrics(self):
        if self._metrics_buffer:
            self._metrics_file.write("\n".join(self._metrics_buffer) + "\n")
            self._metrics_file.flush()
            self._metrics_buffer.clear()

    def _record_event(self, event_type: str, data: dict):
        self._flush_metrics()  # drain buffer before event
        data["type"] = event_type
        data["step"] = self._global_step + 1
        self._metrics_file.write(json.dumps(data) + "\n")
        self._metrics_file.flush()

    def _cached_compress(self, prompt_bytes: bytes, sup) -> float:
        """LRU-cached Super Dict compression for repeated prompts."""
        h = hash(prompt_bytes)
        if h in self._gate_cache:
            return self._gate_cache[h]
        ratio = sup.compress(prompt_bytes)
        if len(self._gate_cache) > 1024:
            # Evict oldest (dict preserves insertion order in Python 3.7+)
            self._gate_cache.pop(next(iter(self._gate_cache)))
        self._gate_cache[h] = ratio
        return ratio
        data["type"] = event_type
        data["step"] = self._global_step + 1
        self._metrics_file.write(json.dumps(data) + "\n")
        self._metrics_file.flush()

    # ── Evaluation ──

    @torch.no_grad()
    def _run_eval(self):
        self._model.eval()
        all_preds = []
        all_labels = []

        eval_loader = torch.utils.data.DataLoader(
            self._eval_dataset, batch_size=self.config.batch_size * 2,
            shuffle=False,
        )

        for i, batch in enumerate(eval_loader):
            if i >= self.config.eval_steps:
                break
            labels = batch.pop("label", None)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)
            outputs = self._model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())

        eval_loss = None
        if all_labels:
            try:
                result = self._metric.compute(predictions=all_preds, references=all_labels)
                eval_loss = result.get(self._metric_name(), 0.0)
            except Exception:
                eval_loss = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        self._record_event("eval", {
            "eval_metric": eval_loss,
            "num_eval_samples": len(all_preds),
        })

        # Compute Super Dict ratio on eval batch
        if self.config.packr_config.mode == "zpackr":
            sup = getattr(self._model, "super_zstd", None)
            if sup is not None:
                sample_text = self._tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                try:
                    ratio = sup.compress(sample_text.encode("utf-8"))
                    self._ephemeral["eval_super_ratio"] = ratio
                except Exception:
                    pass

        self._ephemeral["eval_metric"] = eval_loss
        self._log(f"  Eval at step {self._global_step + 1}: {eval_loss:.4f}")
        self._model.train()

    def _metric_name(self):
        return GLUE_TASKS.get(self.config.task_name, {}).get("metric", "accuracy")

    # ── Reindex ──

    @torch.no_grad()
    def _run_reindex(self):
        new_patterns = {}
        if self._zpl_layers is not None:
            for short_name, module in self._zpl_layers:
                # Collect current delta bytes from this layer
                module._sync_full_delta()
                delta_bytes = module._full_delta.view(torch.uint8).contiguous().view(-1).numpy().tobytes()
                n = self._model.weight_dict.reindex(delta_bytes, min_frequency=0.001, min_count=3, delta_bytes=delta_bytes)
                new_patterns[short_name] = n
                # Reset auto-calibration so this layer recalibrates against the updated dict
                module._salience_threshold = None
        self._record_event("reindex", {"new_patterns": new_patterns})

        wd = getattr(self._model, "weight_dict", None)
        if wd is not None:
            self._log(f"  Reindex at step {self._global_step + 1}: {wd.num_entries} patterns, added {sum(new_patterns.values())} new")

    # ── Checkpoint ──

    def _save_checkpoint(self):
        step_dir = os.path.join(self.checkpoint_dir, f"step_{self._global_step + 1}")
        os.makedirs(step_dir, exist_ok=True)

        # ZPackR layer checkpoints
        if self.config.packr_config.mode == "zpackr":
            from zpackr.checkpoint import save_zpackr_checkpoint
            save_zpackr_checkpoint(self._model, step_dir)

        # Optimizer + Velvet state
        state = {
            "step": self._global_step + 1,
            "optimizer": self._optimizer.state_dict(),
        }
        if self._velvet is not None:
            state["velvet_stats"] = self._velvet.get_stats()
        torch.save(state, os.path.join(step_dir, "trainer_state.pt"))

        self._record_event("checkpoint", {"path": step_dir})

    # ── Summary ──

    def _save_summary(self):
        summary = {
            "total_steps": self._global_step,
            "elapsed_seconds": time.perf_counter() - self._start_time,
            "final_eval_metric": self._ephemeral.get("eval_metric"),
            "peak_vram_mb": self._peak_vram,
            "gate_skipped": self._gate_skipped_total,
            "gate_total": self._gate_total,
            "gate_skip_rate": round(self._gate_skipped_total / max(self._gate_total, 1), 3),
            "output_dir": self.output_dir,
            "config": asdict(self.config),
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._record_event("summary", summary)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZPackR Training Harness")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--mode", default="zpackr", choices=["packr", "zpackr"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--velvet", action="store_true", default=True)
    parser.add_argument("--no-velvet", action="store_false", dest="velvet")
    parser.add_argument("--gate", action="store_true", default=True)
    parser.add_argument("--no-gate", action="store_false", dest="gate")
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--post-step-interval", type=int, default=4)
    parser.add_argument("--reindex-interval", type=int, default=1000)
    parser.add_argument("--calibration-multiplier", type=float, default=0.01)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainerConfig(
        model_name=args.model,
        task_name=args.task,
        packr_config=PackRConfig(
            mode=args.mode,
            zstd_calibration_multiplier=args.calibration_multiplier,
        ),
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        velvet_enabled=args.velvet,
        gate_enabled=args.gate,
        gate_threshold=args.gate_threshold,
        post_step_interval=args.post_step_interval,
        reindex_interval=args.reindex_interval,
        output_dir=args.output_dir,
        run_label=args.label,
        seed=args.seed,
    )
    trainer = ZPackRTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
