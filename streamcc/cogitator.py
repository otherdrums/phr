"""Cogitator — CV2LRT + zstd-gated prompt scheduler for continuous stream learning.

Three-level gating architecture:
  Level 0 (data-level):    ZPackR SuperDict compression ratio on prompt text.
                            Novel text stays, familiar text skips training.
  Level 1 (parameter-group): CV2LRT drops LR per group when v velocity flatlines.
  Level 2 (prompt-level):    Cogitator skips backward for converged tasks.
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
    converged: bool = False
    cv2lrt_multipliers: Dict[str, float] = field(default_factory=dict)
    zstd_gated_count: int = 0      # prompts skipped by zstd gate
    zstd_trained_count: int = 0    # prompts that passed zstd gate

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "epochs_seen": self.epochs_seen,
            "steps_taken": self.steps_taken,
            "last_loss": self.last_loss,
            "converged": self.converged,
            "cv2lrt_multipliers": self.cv2lrt_multipliers,
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
            converged=d.get("converged", False),
            cv2lrt_multipliers=d.get("cv2lrt_multipliers", {}),
            zstd_gated_count=d.get("zstd_gated_count", 0),
            zstd_trained_count=d.get("zstd_trained_count", 0),
        )


class Cogitator:
    """Prompt library with CV2LRT + zstd-gated training scheduling.

    Three-level gating architecture:
      Level 0 (data-level):    ZPackR SuperDict compression ratio on prompt text.
                                Novel text stays, familiar text skips training.
      Level 1 (parameter-group): CV2LRT drops LR per group when v velocity
                                  flatlines.  Built into the optimizer step.
      Level 2 (prompt-level):    Cogitator skips backward entirely for prompts
                                  whose task has converged via CV2LRT check.
    """

    def __init__(
        self,
        stream_trainer,                   # StreamTrainer instance
        convergence_multiplier: float = None,
        super_zstd=None,                  # ZPackRSuperDict for zstd gating
        zstd_gate_threshold: float = 2.0,  # ratio threshold for should_train
    ):
        self.trainer = stream_trainer
        self.super_zstd = super_zstd
        self.zstd_gate_threshold = zstd_gate_threshold

        # Default CV2LRT convergence threshold
        if convergence_multiplier is not None:
            self._threshold = convergence_multiplier
        elif stream_trainer.cv2lrt is not None:
            self._threshold = 2.0 * stream_trainer.cv2lrt.min_m
        else:
            self._threshold = 0.35

        # Prompt storage: task_name → [(input_ids, attention_mask, label, text), ...]
        self._prompts: Dict[str, List[Tuple]] = {}
        self._task_state: Dict[str, TaskState] = {}
        self._converged_tasks: set = set()

    # ── Ingest ──

    def ingest(self, task: str, prompts: List[Tuple]):
        """Register a list of (input_ids, attention_mask, label) prompts for a task.

        Args:
            task:    task name (e.g. "sst2", "mnli")
            prompts: list of (input_ids: Tensor[L], attention_mask: Tensor[L], label: int)
        """
        self._prompts[task] = prompts
        self._task_state[task] = TaskState(task=task)

    def ingest_glue(self, task: str, max_length: int = 128, limit: int = None, seed: int = 42):
        """Convenience: ingest a GLUE task via prompt.ingest_glue."""
        from .prompt import ingest_glue as _ingest
        prompts = _ingest(task, max_length=max_length, limit=limit, seed=seed)
        self.ingest(task, prompts)
        return len(prompts)

    # ── Cogitation ──

    def cogitate(
        self,
        task: str,
        max_epochs: int = 3,
        warmup_steps: int = 0,
        val_fn=None,
        val_interval: int = 0,
        verbose: bool = True,
        use_zstd_gating: bool = False,
    ):
        """Run training passes over a task until convergence or max_epochs.

        Args:
            task:             task name in the prompt library
            max_epochs:       maximum training epochs
            warmup_steps:     CV2LRT warmup micro-steps
            val_fn:           callable(stream_trainer) → float (validation callback)
            val_interval:     validate every N steps (0 = epoch boundaries only)
            verbose:          print progress
            use_zstd_gating:  enable Level 0 zstd compression ratio gating
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
                if self._should_skip(task):
                    continue

                ids, mask, label = prompt_tuple[0], prompt_tuple[1], prompt_tuple[2]
                text = prompt_tuple[3] if len(prompt_tuple) > 3 else None

                # Level 0: zstd gating (before forward pass — raw text only)
                if use_zstd_gating and self.super_zstd is not None and text is not None:
                    if not self._zstd_should_train(text):
                        state.zstd_gated_count += 1
                        continue
                    state.zstd_trained_count += 1

                global_step = state.steps_taken
                if global_step < warmup_steps and self.trainer.cv2lrt is not None:
                    self.trainer.warmup_step(global_step, warmup_steps)

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

            self._check_convergence(task)

            if task in self._converged_tasks:
                if verbose:
                    print(f"  [{task}] CONVERGED after {state.epochs_seen} epoch(s) "
                          f"({state.steps_taken} steps)")
                break

    def cogitate_background(self, tasks: List[str] = None):
        """One pass over non-converged tasks, prioritizing by CV2LRT velocity.

        Tasks with higher average CV2LRT multiplier (more learning headroom)
        get processed first.  Converged tasks are skipped entirely.
        """
        if tasks is None:
            tasks = list(self._prompts.keys())

        active = [t for t in tasks if t not in self._converged_tasks]
        if not active:
            return

        velocities = {}
        for t in active:
            s = self._cv2lrt_snapshot()
            multipliers = [v for v in s.values()]
            velocities[t] = sum(multipliers) / len(multipliers) if multipliers else 1.0

        active.sort(key=velocities.get, reverse=True)
        for t in active:
            self.cogitate(t, max_epochs=1, verbose=False)

    def cogitate_all(self, tasks: List[str] = None, max_epochs: int = 3,
                     warmup_steps: int = 0, val_fn=None, val_interval: int = 0,
                     verbose: bool = True):
        """Cogitate sequentially over all listed (or all ingested) tasks."""
        if tasks is None:
            tasks = list(self._prompts.keys())
        for task in tasks:
            if task not in self._converged_tasks:
                self.cogitate(task, max_epochs=max_epochs, warmup_steps=warmup_steps,
                              val_fn=val_fn, val_interval=val_interval, verbose=verbose)

    # ── Gating ──

    def _zstd_should_train(self, text: str) -> bool:
        """Level 0: zstd compression ratio gate.

        Returns True if the text is novel (should train), False if familiar.
        """
        from packr.prompt_gate import should_train
        return should_train(text.encode("utf-8"), self.super_zstd, self.zstd_gate_threshold)

    def _should_skip(self, task: str) -> bool:
        """Prompt-level gating: skip backward if task has fully converged."""
        return task in self._converged_tasks

    def _check_convergence(self, task: str):
        """Mark a task as converged if all CV2LRT param groups are below threshold.

        When all groups have multiplier <= threshold, the model has "nothing new
        to learn" from this task — CV2LRT has throttled LRs to near-minimum.
        """
        snapshot = self._cv2lrt_snapshot()
        if not snapshot:
            return

        all_low = all(m <= self._threshold for m in snapshot.values())
        if all_low:
            self._converged_tasks.add(task)
            state = self._task_state[task]
            state.converged = True
            state.cv2lrt_multipliers = snapshot

    def _cv2lrt_snapshot(self) -> Dict[str, float]:
        """Return {group_name: current_multiplier} from CV2LRT."""
        if self.trainer.cv2lrt is None:
            return {}
        stats = self.trainer.cv2lrt.get_stats()
        return {
            name: g.get("multiplier", 1.0)
            for name, g in stats.get("per_group", {}).items()
            if isinstance(g, dict)
        }

    # ── Persistence ──

    def save_state(self, path: str):
        """Save task states and convergence info as JSON."""
        data = {
            task: state.to_dict()
            for task, state in self._task_state.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str):
        """Restore task states from a JSON file saved by save_state()."""
        with open(path, "r") as f:
            data = json.load(f)
        for task, d in data.items():
            state = TaskState.from_dict(d)
            self._task_state[task] = state
            if state.converged:
                self._converged_tasks.add(task)

    # ── Query ──

    @property
    def converged_tasks(self) -> set:
        return self._converged_tasks.copy()

    @property
    def active_tasks(self) -> list:
        return [t for t in self._prompts if t not in self._converged_tasks]

    def task_summary(self) -> str:
        """Human-readable task state summary."""
        lines = []
        for t, s in self._task_state.items():
            status = "CONVERGED" if s.converged else "ACTIVE"
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
