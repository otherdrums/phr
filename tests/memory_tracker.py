"""VRAM tracking — PyTorch view + real GPU view (pynvml)."""

import torch
import pynvml


pynvml.nvmlInit()
_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_used_mb():
    """Real GPU memory used (matches nvidia-smi / btop)."""
    info = pynvml.nvmlDeviceGetMemoryInfo(_handle)
    return info.used / (1024 ** 2)


class MemoryTracker:
    def __init__(self):
        self._torch_peak = 0
        self._nvml_peak = 0
        self._torch_model = 0
        self._nvml_model = 0

    def reset(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self._torch_peak = 0
        self._nvml_peak = torch.cuda.memory_allocated() / (1024 ** 2)
        self._nvml_peak = gpu_used_mb()

    def snapshot_model(self):
        torch.cuda.synchronize()
        self._torch_model = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self._nvml_model = gpu_used_mb()

    def step(self):
        torch.cuda.synchronize()
        self._torch_peak = max(self._torch_peak, torch.cuda.max_memory_allocated() / (1024 ** 2))
        self._nvml_peak = max(self._nvml_peak, gpu_used_mb())

    def peak_torch_gb(self):
        return self._torch_peak / 1024

    def peak_nvml_gb(self):
        return self._nvml_peak / 1024

    def summary(self):
        return {
            "torch_model_mb": self._torch_model,
            "nvml_model_mb": self._nvml_model,
            "peak_torch_gb": self._torch_peak / 1024,
            "peak_nvml_gb": self._nvml_peak / 1024,
        }
