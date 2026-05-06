"""
PHR — Packed Hybrid Residual for memory-efficient neural network training.

Usage:
    from phr import compress_model, PHRConfig

    config = PHRConfig(scheme="phr", learnable_lut=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model = compress_model(model, config)
    # train normally with standard PyTorch / HuggingFace loop
"""

from .kernel import phr_matmul
from .autograd import PHRMatmulFunction
from .layer import PHRLinear
from .layer_patcher import compress_model
from .config import PHRConfig, LowRamConfig, SchemeType
from .optim import FusedQuantizedAdam

__all__ = [
    "phr_matmul",
    "PHRMatmulFunction",
    "PHRLinear",
    "compress_model",
    "PHRConfig",
    "LowRamConfig",
    "SchemeType",
    "FusedQuantizedAdam",
]
