"""Layer patcher — replaces nn.Linear layers with PHRLinear in any HF model."""

import torch.nn as nn
from .layer import PHRLinear
from .config import PHRConfig
from .offload import OffloadManager


def compress_model(model: nn.Module, config: PHRConfig = None):
    """
    Replace nn.Linear layers in a model with PHR-compressed equivalents.

    Returns:
        model: nn.Module with PHRLinear layers.
        The OffloadManager (if active) is attached as model._offload_manager.
    """
    if config is None:
        config = PHRConfig()

    phr_layers = []  # ordered (name, PHRLinear) for offload sequencing

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not _matches_scope(name, config.layer_scope):
            continue

        phr = PHRLinear.from_linear(module)
        phr.lut.requires_grad_(config.learnable_lut)

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], phr)

        phr_layers.append((name, phr))

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    if config.offload_level >= 1 and phr_layers:
        # Model must be on CUDA before we can capture W_p for offloading.
        # If not already, move it now (subsequent .cuda() calls are no-ops).
        if next(model.parameters()).is_cpu:
            model.cuda()

        mgr = OffloadManager(prefetch_depth=config.wp_prefetch_depth)
        layer_names = []
        for name, phr in phr_layers:
            mgr.register_wp(name, phr.W_p)
            phr.attach_offload(mgr, name)
            layer_names.append(name)
        mgr.set_layer_sequence(layer_names)
        model._offload_manager = mgr

    return model


def _matches_scope(name: str, scope: str) -> bool:
    """Check if a module path matches the target scope."""
    if scope == "all":
        return True
    if scope == "ffn":
        return _is_ffn(name)
    if scope == "attention":
        return _is_attention(name) and not _is_ffn(name)
    return False


def _is_ffn(name: str) -> bool:
    """Check if a layer is part of a feed-forward network."""
    ffn_intermediate = ["intermediate", "fc1", "mlp.up", "ffn.up", "dense_h_to_4h"]
    ffn_output = ["output.dense", "fc2", "mlp.down", "ffn.down", "dense_4h_to_h"]

    name_lower = name.lower()
    # FFN intermediate (e.g. encoder.layer.X.intermediate.dense)
    if any(m in name_lower for m in ffn_intermediate):
        return True
    # FFN output (e.g. encoder.layer.X.output.dense) — NOT attention output
    if any(m in name_lower for m in ffn_output):
        if "attention" not in name_lower:
            return True
    return False


def _is_attention(name: str) -> bool:
    """Check if a layer name corresponds to attention projection."""
    attn_markers = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
    name_lower = name.lower()
    return any(m in name_lower for m in attn_markers)


def _enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing on the backbone, if supported."""
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        pass
