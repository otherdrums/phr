"""Training hyperparameter defaults — single source of truth.

All tunable parameters live here as a plain dataclass that can be
serialized to JSON and saved alongside results for reproducibility.
"""

from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    # ── Optimizer ──
    body_lr: float = 2e-5
    head_lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    block_size: int = 256

    # ── LR schedule (fractions of total micro-batches) ──
    warmup_fraction: float = 0.02
    hold_fraction: float = 0.60
    warmup_start_factor: float = 0.1
    decay_eta_min: float = 0.1

    # ── Training loop ──
    epochs: int = 5
    batch_size: int = 8
    acc_steps: int = 4
    max_seq_length: int = 128
    val_interval: int = 100

    # ── PHR model (used by build_phr) ──
    layer_scope: str = "ffn"
    learnable_lut: bool = True
    gradient_checkpointing: bool = True
    scheme: str = "phr"

    def to_dict(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d
