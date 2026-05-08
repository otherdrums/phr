"""Configuration for PHR memory-efficient training."""

from dataclasses import dataclass, field
from typing import Optional, Literal


SchemeType = Literal["phr"]


@dataclass
class PHRConfig:
    """Configuration for PHR-based memory-efficient fine-tuning.

    Args:
        scheme:                 Compression scheme (currently only "phr")
        learnable_lut:          Whether the LUT codebook is trainable
        layer_scope:            Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing on the backbone
        use_8bit_optimizer:     Use FusedQuantizedAdam (Triton 8-bit Adam)
        offload_level:          GPU→CPU offloading aggressiveness:
                                  0 = none (default, all GPU-resident)
                                  1 = W_p streaming (frozen indices from pinned CPU)
                                  2 = + optimizer state storage offload (m/v on CPU)
                                  3 = + optimizer compute offload (AdamW on CPU)
        wp_prefetch_depth:      Layers ahead to prefetch W_p (default 1 = double-buffer)
        block_size:             Quantization block size for 8-bit optimizer
    """

    scheme: SchemeType = "phr"
    learnable_lut: bool = True
    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    offload_level: int = 0
    wp_prefetch_depth: int = 1
    block_size: int = 256
