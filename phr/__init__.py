"""
PHR — Packed Hybrid Residual for memory-efficient neural network training.

Usage:
    from phr import compress_model, PHRConfig

    config = PHRConfig(scheme="phr", learnable_lut=True, offload=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model = compress_model(model, config)
    # train normally with standard PyTorch / HuggingFace loop
"""

from .kernel import phr_matmul
from .autograd import PHRMatmulFunction
from .layer import PHRLinear
from .layer_patcher import compress_model
from .config import PHRConfig, SchemeType
from .optim import FusedQuantizedAdam
from .offload import OffloadManager
from .cv2lrt import CV2LRTController

__all__ = [
    "phr_matmul",
    "PHRMatmulFunction",
    "PHRLinear",
    "compress_model",
    "PHRConfig",
    "SchemeType",
    "FusedQuantizedAdam",
    "OffloadManager",
    "CV2LRTController",
]
