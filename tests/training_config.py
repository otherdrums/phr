"""Training hyperparameter defaults — single source of truth.

All tunable parameters live here as a plain dataclass that can be
serialized to JSON and saved alongside results for reproducibility.

Method-specific overrides use METHOD_CONFIGS to ensure fair comparison
against state-of-the-art baselines. Defaults are PHR-specific.
"""

from dataclasses import dataclass, asdict

# ── Per-method hyperparameters against SOTA baselines ──
# LoRA: Hu et al. 2021 (arXiv:2106.09685) — lr=4e-4..5e-4, α=r, Q+V targets
# QLoRA: Dettmers et al. 2023 (arXiv:2305.14314) — same LoRA params
# BitFit: Ben-Zaken et al. 2022 (arXiv:2106.10199) — lr=1e-3..3e-3
# Full: Devlin et al. 2019 — lr=2e-5, wd=0.01, same LR head+body

METHOD_CONFIGS = {
    "full":   {"body_lr": 2e-5, "head_lr": 2e-5, "weight_decay": 0.01},
    "phr":    {"body_lr": 2e-5, "head_lr": 1e-3, "weight_decay": 0.0},
    "lora":   {"body_lr": 5e-4, "head_lr": 5e-4, "weight_decay": 0.0},
    "qlora":  {"body_lr": 5e-4, "head_lr": 5e-4, "weight_decay": 0.0},
    "bitfit": {"body_lr": 2e-3, "head_lr": 2e-3, "weight_decay": 0.0},
}


def method_lr_config(method: str) -> dict:
    """Return (body_lr, head_lr, weight_decay) for a given method."""
    return METHOD_CONFIGS.get(method, METHOD_CONFIGS["full"])


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

    # ── CV2LRT adaptive scheduler ──
    cv2lrt_enabled: bool = False
    cv2lrt_beta: float = 0.97
    cv2lrt_min_multiplier: float = 0.1
    cv2lrt_max_multiplier: float = 1.0
    cv2lrt_velocity_scale: float = 10.0
    cv2lrt_granularity: str = "matrix"   # "coarse" | "layer" | "matrix"

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
