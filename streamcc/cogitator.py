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
    ):
        self.trainer = stream_trainer
        self.super_zstd = super_zstd
        self.zstd_gate_threshold = zstd_gate_threshold

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
        use_zstd_gating: bool = False,
    ):
        """Run one pass over a task's prompts.

        In ZPackR mode, call post_step() and decay_delta() separately
        between cogitate() calls — the Cogitator only handles prompt iteration.
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

        for epoch in range(max_epochs):
            if task in self._converged_tasks:
                break

            self.trainer.reset_stats()

            for i, prompt_tuple in enumerate(prompts):
                if task in self._converged_tasks:
                    break

                ids, mask, label = prompt_tuple[0], prompt_tuple[1], prompt_tuple[2]
                text = prompt_tuple[3] if len(prompt_tuple) > 3 else None

                # Level 0: zstd gating (before forward pass)
                if use_zstd_gating and self.super_zstd is not None and text is not None:
                    if not self._zstd_should_train(text):
                        state.zstd_gated_count += 1
                        continue
                    state.zstd_trained_count += 1

                # Fixed LR training (no Velvet warmup in zpackr mode)
                loss, _ = self.trainer.step(ids, mask, label)
                state.steps_taken += 1
                state.last_loss = loss

                if verbose and (i + 1) % 5000 == 0:
                    acc = self.trainer.running_acc
                    parts = [f"[{task}] epoch {epoch + 1}/{max_epochs}",
                             f"step {i + 1}/{total_prompts}",
                             f"loss {loss:.4f}  acc {acc:.2f}%"]
                    if use_zstd_gating:
                        parts.append(f"zstd(gated={state.zstd_gated_count} trained={state.zstd_trained_count})")
                    print("  " + "  ".join(parts))

                if val_fn is not None and val_interval > 0 and (i + 1) % val_interval == 0:
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
                     verbose: bool = True, use_zstd_gating: bool = False):
        if tasks is None:
            tasks = list(self._prompts.keys())
        for task in tasks:
            if task not in self._converged_tasks:
                self.cogitate(task, max_epochs=max_epochs, warmup_steps=warmup_steps,
                              val_fn=val_fn, val_interval=val_interval,
                              verbose=verbose, use_zstd_gating=use_zstd_gating)

    # ── Gating ──

    def _zstd_should_train(self, text: str) -> bool:
        from packr.prompt_gate import should_train
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
