"""Cogitator — prompt scheduler for continuous stream learning.

In ZPackR mode (default supported path), the WeightDict's block-level
compression ratios replace the LR scheduler entirely.  No Velvet needed —
post_step() on ZPackRLinear layers computes novelty per block, and the
forward pass already attenuates known patterns.

Gating architecture:
  Level 0 (data-level):    ZPackR SuperDict compression ratio on prompt text.
                            Novel text stays, familiar text skips training.
  Level 1 (block-level):   ZPackRLinear.post_step() — WeightDict compression
                            ratios → novelty scores.  Forward already attenuates
                            known blocks.  decay_delta() fades them over time.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TaskState:
    """Per-task learning state tracked by the Cogitator."""
    task: str
    epochs_seen: int = 0
    steps_taken: int = 0
    last_loss: float = 0.0
    zstd_gated_count: int = 0      # prompts skipped by zstd gate
    zstd_trained_count: int = 0    # prompts that passed zstd gate

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "epochs_seen": self.epochs_seen,
            "steps_taken": self.steps_taken,
            "last_loss": self.last_loss,
            "zstd_gated_count": self.zstd_gated_count,
            "zstd_trained_count": self.zstd_trained_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        return cls(
            task=d["task"],
            epochs_seen=d.get("epochs_seen", 0),
            steps_taken=d.get("steps_taken", 0),
            last_loss=d.get("last_loss", 0.0),
            zstd_gated_count=d.get("zstd_gated_count", 0),
            zstd_trained_count=d.get("zstd_trained_count", 0),
        )


class Cogitator:
    """Prompt library with zstd-native gating.

    In ZPackR mode, the WeightDict block-level compression is the sole
    convergence signal — no external LR scheduler (Velvet/CV2LRT) is needed.
    Known blocks attenuate in forward, decay toward zero, and are pruned
    from VRAM below the auto-calibrated salience threshold.
    """

    def __init__(
        self,
        stream_trainer,
        super_zstd=None,
        zstd_gate_threshold: float = 2.0,
        post_opt_step_fn=None,
        batch_size: int = 16,
    ):
        self.trainer = stream_trainer
        self.super_zstd = super_zstd
        self.zstd_gate_threshold = zstd_gate_threshold
        self.batch_size = batch_size
        self._post_opt_step_fn = post_opt_step_fn
        self._zstd_gating_active = (super_zstd is not None)

        # Wire post-opt-step hook into trainer
        if post_opt_step_fn is not None and stream_trainer._post_opt_step_fn is None:
            stream_trainer._post_opt_step_fn = post_opt_step_fn

        # Prompt storage: task_name → [(input_ids, attention_mask, label, text), ...]
        self._prompts: Dict[str, List[Tuple]] = {}
        self._task_state: Dict[str, TaskState] = {}
        self._converged_tasks: set = set()

    # ── Ingest ──

    def ingest(self, task: str, prompts: List[Tuple]):
        self._prompts[task] = prompts
        self._task_state[task] = TaskState(task=task)

    def ingest_glue(self, task: str, max_length: int = 128, limit: int = None, seed: int = 42):
        from .prompt import ingest_glue as _ingest
        prompts = _ingest(task, max_length=max_length, limit=limit, seed=seed)
        self.ingest(task, prompts)
        return len(prompts)

    # ── Cogitation ──

    def cogitate(
        self,
        task: str,
        max_epochs: int = 1,
        warmup_steps: int = 0,
        val_fn=None,
        val_interval: int = 0,
        verbose: bool = True,
    ):
        """Run one pass over a task's prompts, batched for GPU throughput.

        Args:
            task:         task name in the prompt library
            max_epochs:   maximum training passes
            warmup_steps: (ignored in zpackr mode — no LR schedule)
            val_fn:       callable(stream_trainer) → float (validation callback)
            val_interval: validate every N steps (0 = never)
            verbose:      print progress
        """
        if task not in self._prompts:
            raise KeyError(f"Unknown task: {task} (ingest first)")

        if task in self._converged_tasks:
            if verbose:
                print(f"  [{task}] already converged — skipping")
            return

        state = self._task_state[task]
        prompts = self._prompts[task]
        total_prompts = len(prompts)
        bs = self.batch_size

        for epoch in range(max_epochs):
            if task in self._converged_tasks:
                break

            self.trainer.reset_stats()

            # Batch prompts for GPU throughput
            for batch_start in range(0, total_prompts, bs):
                batch_end = min(batch_start + bs, total_prompts)
                batch_prompts = prompts[batch_start:batch_end]

                if task in self._converged_tasks:
                    break

                # Collect prompt tuples, optionally filter with zstd gate
                ids_list, mask_list, label_list = [], [], []
                for prompt_tuple in batch_prompts:
                    ids, mask, label = prompt_tuple[0], prompt_tuple[1], prompt_tuple[2]
                    text = prompt_tuple[3] if len(prompt_tuple) > 3 else None

                    if self._zstd_gating_active and self.super_zstd is not None and text is not None:
                        if not self._zstd_should_train(text):
                            state.zstd_gated_count += 1
                            continue
                        state.zstd_trained_count += 1

                    ids_list.append(ids)
                    mask_list.append(mask)
                    label_list.append(label)

                if not ids_list:
                    continue

                # Stack into batch tensors
                batch_ids = torch.stack(ids_list)
                batch_mask = torch.stack(mask_list)
                batch_labels = torch.tensor(label_list)

                # Training step — batch of prompts
                loss, _ = self.trainer.step(batch_ids, batch_mask, batch_labels)
                state.steps_taken += len(ids_list)
                state.last_loss = loss

                i = batch_end
                if verbose and (i) % 5000 == 0:
                    acc = self.trainer.running_acc
                    parts = [f"[{task}] epoch {epoch + 1}/{max_epochs}",
                             f"step {i}/{total_prompts}",
                             f"loss {loss:.4f}  acc {acc:.2f}%"]
                    if self._zstd_gating_active:
                        parts.append(f"zstd(gated={state.zstd_gated_count} trained={state.zstd_trained_count})")
                    print("  " + "  ".join(parts))

                if val_fn is not None and val_interval > 0 and (i) % val_interval == 0:
                    val_acc = val_fn(self.trainer)
                    if verbose:
                        print(f"  [{task}] val acc {val_acc:.2f}%")

            state.epochs_seen += 1

            if val_fn is not None:
                val_acc = val_fn(self.trainer)
                if verbose:
                    print(f"  [{task}] epoch {state.epochs_seen} complete  val acc {val_acc:.2f}%")

    def cogitate_all(self, tasks: List[str] = None, max_epochs: int = 1,
                     warmup_steps: int = 0, val_fn=None, val_interval: int = 0,
                     verbose: bool = True):
        if tasks is None:
            tasks = list(self._prompts.keys())
        for task in tasks:
            if task not in self._converged_tasks:
                self.cogitate(task, max_epochs=max_epochs, warmup_steps=warmup_steps,
                              val_fn=val_fn, val_interval=val_interval,
                              verbose=verbose)

    # ── Gating ──

    def _zstd_should_train(self, text: str) -> bool:
        from zpackr.prompt_gate import should_train
        return should_train(text.encode("utf-8"), self.super_zstd, self.zstd_gate_threshold)

    # ── Persistence ──

    def save_state(self, path: str):
        data = {task: state.to_dict() for task, state in self._task_state.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        for task, d in data.items():
            state = TaskState.from_dict(d)
            self._task_state[task] = state

    # ── Query ──

    @property
    def converged_tasks(self) -> set:
        return self._converged_tasks.copy()

    def mark_converged(self, task: str):
        self._converged_tasks.add(task)

    def task_summary(self) -> str:
        lines = []
        for t, s in self._task_state.items():
            status = "CONVERGED" if t in self._converged_tasks else "ACTIVE"
            parts = [
                f"[{t}] {status}",
                f"epochs={s.epochs_seen}",
                f"steps={s.steps_taken}",
                f"last_loss={s.last_loss:.4f}",
            ]
            if s.zstd_gated_count > 0 or s.zstd_trained_count > 0:
                parts.append(f"zstd(gated={s.zstd_gated_count} trained={s.zstd_trained_count})")
            lines.append("  " + "  ".join(parts))
        return "\n".join(lines) if lines else "  (no tasks ingested)"
