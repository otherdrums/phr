"""CV2LRT — Continuous Velocity to Learning Rate Translation.

Closed-loop adaptive learning rate controller that reads AdamW second-moment
statistics (exp_avg_sq / v) every optimizer step and adjusts per-param-group
LRs in real time based on the filtered velocity of gradient variance.

Core insight: when a layer is actively learning, its exp_avg_sq climbs
(positive velocity → LR stays hot).  When a layer saturates, exp_avg_sq
flattens (velocity → 0 → LR decays).  The Exponential Moving Average (EMA)
acts as a low-pass filter separating signal from micro-batch noise.

Works with both standard torch.optim.AdamW (float32 state) and
FusedQuantizedAdam (int8 block-quantized state) via auto-detection.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional


class CV2LRTController:
    """Continuous Velocity to Learning Rate Translation controller.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose param_groups will be adapted in real time.
    beta : float
        EMA smoothing coefficient for velocity (0 < beta < 1).
        Higher = more smoothing, slower to react.  Default 0.97.
    min_multiplier : float
        Minimum LR multiplier applied when velocity is near zero
        (layer is saturating).  Default 0.175.
    max_multiplier : float
        Maximum LR multiplier applied when velocity is high
        (layer is actively learning).  Default 1.0.
    velocity_scale : float
        Scaling factor that maps normalized velocity to the [0,1] range
        before clamping to [min_multiplier, max_multiplier].
        Higher = more aggressive LR reduction.  Default 10.0.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.97,
        min_multiplier: float = 0.175,
        max_multiplier: float = 1.0,
        velocity_scale: float = 10.0,
    ):
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not 0 <= min_multiplier <= max_multiplier:
            raise ValueError(
                f"min_multiplier ({min_multiplier}) must be <= "
                f"max_multiplier ({max_multiplier})"
            )

        self._optimizer = optimizer
        self.beta = beta
        self.min_m = min_multiplier
        self.max_m = max_multiplier
        self.vel_scale = velocity_scale

        # Per-parameter tracking, keyed by id(p)
        self._prev_v_mean: Dict[int, float] = {}   # v_mean from previous step
        self._ema_velocity: Dict[int, float] = {}   # EMA-filtered Δv
        self._base_lr: Dict[int, float] = {}        # group_idx → base_lr

        # Capture base LRs immediately (before warmup or any scheduling
        # modifies them).  These are immutable reference values.
        for group_idx, group in enumerate(self._optimizer.param_groups):
            self._base_lr[group_idx] = group["lr"]

        self._step_count = 0
        self._stats: Dict[str, list] = {
            "step": [],
            "group": [],
            "v_mean": [],
            "velocity": [],
            "multiplier": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def warmup_step(self, global_step: int, warmup_steps: int) -> None:
        """Apply linear warmup LR schedule (called every micro-batch).

        Ramps each param group's LR from 10% to 100% of its base_lr
        over ``warmup_steps`` micro-batches.
        """
        factor = 0.1 + 0.9 * (global_step / max(warmup_steps, 1))
        for group_idx, group in enumerate(self._optimizer.param_groups):
            group["lr"] = self._base_lr[group_idx] * factor

    @torch.no_grad()
    def step(self) -> None:
        """Called **after** ``optimizer.step()``.

        Reads the freshly-updated ``v`` (exp_avg_sq) states, computes
        the filtered velocity, and translates it to per-group LR multipliers.

        The first call is observation-only — it seeds ``prev_v_mean``
        without adjusting LRs.  All subsequent calls apply the
        velocity→LR translation.  Base LRs are captured at construction
        time so warmup cannot corrupt them.
        """
        self._step_count += 1
        is_first = self._step_count == 1

        for group_idx, group in enumerate(self._optimizer.param_groups):
            multipliers: List[float] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._optimizer.state.get(p)
                if state is None or "v" not in state:
                    continue

                pid = id(p)

                # ---- dequantize and compute mean of v ----
                v_mean = self._dequantize_v_mean(state["v"], state.get("v_scale"))

                if is_first or pid not in self._prev_v_mean:
                    # First call: seed state, use max multiplier
                    self._prev_v_mean[pid] = v_mean
                    self._ema_velocity[pid] = 0.0
                    multipliers.append(self.max_m)
                    continue

                # ---- compute velocity ----
                prev = self._prev_v_mean[pid]
                self._prev_v_mean[pid] = v_mean
                delta = v_mean - prev

                # ---- EMA filter ----
                ema = (
                    self.beta * self._ema_velocity[pid]
                    + (1.0 - self.beta) * delta
                )
                self._ema_velocity[pid] = ema

                # ---- normalize by current v_mean (relative rate of change) ----
                norm_vel = abs(ema) / (v_mean + 1e-12)

                # ---- translate to multiplier ----
                multiplier = self.min_m + (self.max_m - self.min_m) * min(
                    1.0, norm_vel * self.vel_scale
                )
                multipliers.append(multiplier)

            # Apply per-param-group multiplier
            if multipliers:
                group_m = sum(multipliers) / len(multipliers)
                group["lr"] = self._base_lr[group_idx] * group_m

                # Collect stats for the *first* param in group (representative)
                sample_id = id(group["params"][0])
                self._stats["step"].append(self._step_count)
                self._stats["group"].append(group_idx)
                self._stats["v_mean"].append(
                    self._prev_v_mean.get(sample_id, 0.0)
                )
                self._stats["velocity"].append(
                    self._ema_velocity.get(sample_id, 0.0)
                )
                self._stats["multiplier"].append(group_m)
                self._stats["lr"].append(group["lr"])

    def get_stats(self) -> dict:
        """Return a snapshot of internal velocity/multiplier statistics.

        Useful for heartbeat logging and post-hoc analysis.
        """
        # Per-group latest values
        per_group = {}
        for group_idx, group in enumerate(self._optimizer.param_groups):
            gname = group.get("name", f"group_{group_idx}")
            per_group[gname] = {
                "base_lr": self._base_lr.get(group_idx),
                "current_lr": group["lr"],
                "multiplier": group["lr"] / max(self._base_lr.get(group_idx, 1e-12), 1e-12),
            }

        return {
            "step_count": self._step_count,
            "per_group": per_group,
            "history": {
                "steps": self._stats["step"][-100:],  # last 100
                "multipliers": self._stats["multiplier"][-100:],
                "velocities": self._stats["velocity"][-100:],
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dequantize_v_mean(
        self, v: torch.Tensor, v_scale: Optional[torch.Tensor]
    ) -> float:
        """Compute the mean of the exp_avg_sq state tensor.

        Handles two formats:
        - Float32 (standard AdamW): ``v`` is the raw floating-point tensor.
        - Int8 block-quantized (FusedQuantizedAdam): ``v`` is int8 with
          per-block float32 ``v_scale``.  Dequantized as:
          ``v_fp = v_i8.float().view(num_blocks, block_size) * v_scale``.
        """
        if v.dtype == torch.float32:
            return v.mean().item()

        # Int8 block-quantized path
        if v_scale is None:
            return v.float().mean().item()

        N = v.numel()
        num_blocks = v_scale.numel()
        block_size = (N + num_blocks - 1) // num_blocks

        # Reshape to [num_blocks, block_size], dequantize, mean
        v_r = v.float()
        if N < num_blocks * block_size:
            # There's padding — handle gracefully
            return v_r.mean().item()

        v_r = v_r.view(num_blocks, block_size)
        block_means = v_r.mean(dim=1) * v_scale
        return block_means.mean().item()
