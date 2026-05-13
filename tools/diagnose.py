"""ZPackR Diagnostic Trainer — ratio-logging for signal calibration.

Thin wrapper around ZPackRTrainer that adds per-block compression ratio
tracking at each post_step boundary.  Produces a ratio_log.jsonl file
with per-step and per-block signals needed to find the gap→LR relationship.

Usage:
    python tools/diagnose.py --task sst2 --max-steps 500 --post-step-interval 1

Output:
    runs/diagnostic_<ts>_<commit>/
        metrics.jsonl          # standard harness metrics
        ratio_log.jsonl        # per-step ratios + per-block snapshots
        config.json            # full config snapshot
        summary.json           # final summary
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tools.train_harness import ZPackRTrainer, TrainerConfig, GLUE_TASKS
from packr.config import PackRConfig
from zpackr.zpackr_layer import ZPackRLinear


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _git_commit_short():
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class DiagnosticTrainer(ZPackRTrainer):
    """ZPackRTrainer with per-block ratio logging at each post_step."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self._ratio_file = None

    def run(self) -> dict:
        self.setup()

        # Open ratio log file
        ratio_path = os.path.join(self.output_dir, "ratio_log.jsonl")
        self._ratio_file = open(ratio_path, "w")

        self._start_time = time.perf_counter()
        self._log(f"Starting diagnostic training ({self.config.max_steps} steps) ...")

        self._model.train()
        train_iter = iter(self._train_loader)

        while self._global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self._train_loader)
                batch = next(train_iter)

            step_start = time.perf_counter()

            # ── Super Dict ratio (pre-forward gate signal) ──
            super_ratio = 1.0
            sup = getattr(self._model, "super_zstd", None)
            if sup is not None:
                prompt_text = self._tokenizer.decode(
                    batch["input_ids"][0], skip_special_tokens=True
                )
                try:
                    super_ratio = self._cached_compress(prompt_text.encode("utf-8"), sup)
                except Exception:
                    pass

            # ── Gate check (skip-forward mode) ──
            gate_skipped = False
            if self.config.gate_enabled and self.config.gate_skip_forward:
                if sup is not None:
                    self._gate_total += 1
                    # Use already-decoded prompt_text from super_ratio section
                    if super_ratio >= self.config.gate_threshold:
                        self._gate_skipped_total += 1
                        gate_skipped = True
                        self._global_step += 1
                        self._record_step({
                            "step": self._global_step,
                            "gate_skipped": True,
                        })
                        continue

            # ── Forward ──
            labels = batch.pop("label", None)
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            if labels is not None:
                labels = labels.to(self.device)

            outputs = self._model(**batch_gpu, labels=labels)
            loss = outputs.loss / self.config.grad_accum_steps

            # ── Gate check (skip-backward mode) ──
            if self.config.gate_enabled and not self.config.gate_skip_forward:
                if sup is not None:
                    self._gate_total += 1
                    if super_ratio >= self.config.gate_threshold:
                        gate_skipped = True
                        self._gate_skipped_total += 1

            if not gate_skipped:
                loss.backward()

                if (self._global_step + 1) % self.config.grad_accum_steps == 0:
                    self._optimizer.step()

                    if self.config.warmup_steps > 0 and self._velvet is not None:
                        if self._global_step < self.config.warmup_steps:
                            self._velvet.warmup_step(
                                self._global_step, self.config.warmup_steps
                            )

                    if self._velvet is not None:
                        self._velvet.step()

                    if self._zpl_layers is not None:
                        if (self._global_step + 1) % self.config.post_step_interval == 0:
                            for _, module in self._zpl_layers:
                                module.post_step(
                                    calibration_multiplier=self.config.packr_config.zstd_calibration_multiplier
                                )

                    self._optimizer.zero_grad()

                    # Decay known blocks toward zero
                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            if module._gap_enabled:
                                module.shrink_known_delta()

                    # Pre-stage delta for next post_step
                    if self._zpl_layers is not None:
                        for _, module in self._zpl_layers:
                            module.stage_delta_async(None)

            # ── Log ratios every step (cached between post_steps) ──
            if self._zpl_layers is not None:
                self._log_ratios(super_ratio, loss.item(), gate_skipped)

            step_ms = (time.perf_counter() - step_start) * 1000
            self._record_step(self._gather_metrics(
                loss.item() * self.config.grad_accum_steps, step_ms, gate_skipped
            ))

            if self._zpl_layers is not None:
                if (self._global_step + 1) % self.config.reindex_interval == 0:
                    self._run_reindex()

            if (self._global_step + 1) % self.config.eval_interval == 0:
                self._run_eval()

            if (self._global_step + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            self._global_step += 1

        self._run_eval()
        self._save_summary()
        self._metrics_file.close()
        self._ratio_file.close()

        elapsed = time.perf_counter() - self._start_time
        self._log(f"Diagnostic training complete in {elapsed:.1f}s. Output: {self.output_dir}")
        return self._ephemeral

    # ── Ratio logging ──

    def _log_ratios(self, super_ratio: float, loss: float, gate_skipped: bool):
        """Extract and log per-block compression ratios from all ZPackRLinear layers."""
        if self._zpl_layers is None:
            return

        log = {
            "step": self._global_step + 1,
            "loss": loss,
            "super_ratio": super_ratio,
            "gate_skipped": gate_skipped,
            "layers": {},
        }

        all_ratios = []
        all_kept_ratios = []

        for short_name, module in self._zpl_layers:
            data = module.get_block_ratios()
            ratios = data["ratios"]

            layer_info = {
                "ratio_max": max(ratios),
                "ratio_min": min(ratios),
                "ratio_mean": sum(ratios) / len(ratios),
                "calibration_max": data["calibration_max"],
                "calibrated_threshold": data["calibrated_threshold"],
                "salient_count": data["salient_count"],
                "num_blocks": data["num_blocks"],
            }

            # Per-block snapshot
            gaps = data.get("block_gaps", ratios)
            novelties = data.get("novelty_scores")
            if novelties is None:
                novelties = [1.0] * len(ratios)
            layer_info["blocks"] = [
                {
                    "blk": i,
                    "ratio": ratios[i],
                    "gap": gaps[i] if i < len(gaps) else 1.0,
                    "novelty": novelties[i] if i < len(novelties) else 1.0,
                    "delta_l2": round(data["delta_l2"][i], 8),
                    "salient": i < data["salient_count"],
                }
                for i in range(len(ratios))
            ]

            log["layers"][short_name] = layer_info

            all_ratios.extend(ratios)
            # Kept blocks are the first salient_count blocks (compacted view)
            salient_mask = module.block_mask
            for i in range(len(ratios)):
                if salient_mask[i]:
                    all_kept_ratios.append(ratios[i])

        # Aggregate stats
        if all_ratios:
            log["weight_ratio_max"] = max(all_ratios)
            log["weight_ratio_min"] = min(all_ratios)
            log["weight_ratio_mean"] = sum(all_ratios) / len(all_ratios)
        if all_kept_ratios:
            log["weight_ratio_kept_mean"] = sum(all_kept_ratios) / len(all_kept_ratios)

        self._ratio_file.write(json.dumps(log) + "\n")
        self._ratio_file.flush()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ZPackR Diagnostic Trainer — ratio logging for signal calibration"
    )
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--velvet", action="store_true", default=True)
    parser.add_argument("--no-velvet", action="store_false", dest="velvet")
    parser.add_argument("--gate", action="store_true", default=True)
    parser.add_argument("--no-gate", action="store_false", dest="gate")
    parser.add_argument("--gate-threshold", type=float, default=2.0)
    parser.add_argument("--gate-skip-forward", action="store_true", default=False)
    parser.add_argument("--post-step-interval", type=int, default=1)
    parser.add_argument("--reindex-interval", type=int, default=1000)
    parser.add_argument("--calibration-multiplier", type=float, default=0.01)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--label", default="", help="Prefix for output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainerConfig(
        model_name=args.model,
        task_name=args.task,
        packr_config=PackRConfig(
            mode="zpackr",
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
        gate_skip_forward=args.gate_skip_forward,
        post_step_interval=args.post_step_interval,
        reindex_interval=args.reindex_interval,
        output_dir=args.output_dir,
        run_label=args.label or "diagnostic",
        seed=args.seed,
    )
    trainer = DiagnosticTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
