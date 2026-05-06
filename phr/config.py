"""Configuration for PHR memory-efficient training."""

from dataclasses import dataclass, field
from typing import Optional, Literal


SchemeType = Literal["phr"]


@dataclass
class PHRConfig:
    """Configuration for PHR-based memory-efficient fine-tuning.

    Args:
        scheme:           Compression scheme (currently only "phr")
        learnable_lut:    Whether the LUT codebook is trainable
        layer_scope:      Which linear layers to replace
        gradient_checkpointing: Enable gradient checkpointing on the backbone
        use_8bit_optimizer:  Use FusedQuantizedAdam (Triton 8-bit Adam)
        offload_frozen_params: Offload W_p and LUT to CPU when not needed
        block_size:       Quantization block size for 8-bit optimizer
    """

    scheme: SchemeType = "phr"
    learnable_lut: bool = True
    layer_scope: Literal["ffn", "attention", "all"] = "ffn"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    offload_frozen_params: bool = False
    block_size: int = 256


LowRamConfig = PHRConfig
