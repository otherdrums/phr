"""Method-specific model builders for the SST-2 comparison."""

import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from packr import compress_model, PHRConfig, FusedQuantizedAdam
from .training_config import TrainingConfig, method_lr_config

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

_MODEL_KWARGS = {"local_files_only": True}
_cfg = TrainingConfig()


def build_full_finetune(num_labels=2, seed=42):
    """Vanilla BERT — all parameters trainable."""
    torch.manual_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()
    return model, None


def build_bitfit(num_labels=2, seed=42):
    """Bias-only fine-tuning — freeze all except biases."""
    torch.manual_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()
    for name, param in model.named_parameters():
        if "bias" not in name and "classifier" not in name:
            param.requires_grad = False
    return model, None


def build_lora(num_labels=2, seed=42):
    """LoRA adapters on attention Q+V projections, r=8, α=r per Hu et al. 2021."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model, None


def build_qlora(num_labels=2, seed=42):
    """8-bit quantized BERT with LoRA adapters, r=8, α=r."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    torch.manual_seed(seed)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["classifier"],
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model, None


def build_phr(offload=False, num_labels=2, seed=42):
    """PHR-compressed FFN layers with 8-bit Adam."""
    torch.manual_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    phr_cfg = PHRConfig(
        scheme=_cfg.scheme,
        layer_scope=_cfg.layer_scope,
        learnable_lut=_cfg.learnable_lut,
        gradient_checkpointing=_cfg.gradient_checkpointing,
        offload=offload,
    )
    model = compress_model(model, phr_cfg)

    # Separate params for differential LR
    head_params = []
    phr_params = []
    for n, p in model.named_parameters():
        if "classifier" in n or "cls" in n:
            head_params.append(p)
        elif p.requires_grad:
            phr_params.append(p)

    optimizer = FusedQuantizedAdam(
        [
            {"params": phr_params, "lr": _cfg.body_lr},
            {"params": head_params, "lr": _cfg.head_lr},
        ],
        betas=_cfg.betas,
        eps=_cfg.eps,
        weight_decay=_cfg.weight_decay,
        block_size=_cfg.block_size,
    )
    if offload and hasattr(model, '_offload_manager'):
        optimizer.enable_offload(model._offload_manager)
    return model, optimizer


def build_optimizer(model, method_name, prebuilt_optimizer=None):
    """Create the appropriate optimizer for a given method.

    Uses method-specific LRs and weight_decay from METHOD_CONFIGS.
    Non-PHR methods use same LR for body and head (no differential LR)
    per standard practice in LoRA, QLoRA, and BitFit papers.
    """
    if prebuilt_optimizer is not None:
        return prebuilt_optimizer

    mc = method_lr_config(method_name)

    head_params = []
    body_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n or "cls" in n:
            head_params.append(p)
        else:
            body_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": body_params, "lr": mc["body_lr"]},
            {"params": head_params, "lr": mc["head_lr"]},
        ],
        betas=_cfg.betas,
        eps=_cfg.eps,
        weight_decay=mc["weight_decay"],
    )


def _param_group_key(name, granularity="matrix"):
    """Return a group key for a named parameter based on granularity.

    granularity values:
      - "coarse":  single "body" key for all non-head params
      - "layer":   "layer_{N}" for encoder layer N, "embeddings", "pooler", "head"
      - "matrix":  the module path (parameter name with last segment stripped)
    """
    if "classifier" in name or "cls" in name:
        return "head"

    if granularity == "coarse":
        return "body"

    # Extract encoder layer index for "layer" granularity
    if granularity == "layer":
        import re
        m = re.search(r"encoder\.layer\.(\d+)", name)
        if m:
            return f"layer_{m.group(1)}"
        if "embedding" in name.lower():
            return "embeddings"
        if "pooler" in name.lower():
            return "pooler"
        return "body"

    # "matrix" granularity: strip last .-segment (W_f, weight, bias, etc.)
    granularity = "matrix"  # default
    return ".".join(name.split(".")[:-1])


def build_phr_cv2lrt(offload=False, num_labels=2, seed=42):
    """PHR-compressed FFN layers with per-module parameter groups for CV2LRT."""
    torch.manual_seed(seed)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    phr_cfg = PHRConfig(
        scheme=_cfg.scheme,
        layer_scope=_cfg.layer_scope,
        learnable_lut=_cfg.learnable_lut,
        gradient_checkpointing=_cfg.gradient_checkpointing,
        offload=offload,
    )
    model = compress_model(model, phr_cfg)

    granularity = _cfg.cv2lrt_granularity

    # Collect params by group key
    head_params = []
    groups: dict = {}  # key → [params]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = _param_group_key(n, granularity)
        if key == "head":
            head_params.append(p)
        else:
            groups.setdefault(key, []).append(p)

    # Build param_groups list with per-group LR
    param_groups = []
    for key, params in sorted(groups.items()):
        param_groups.append({
            "params": params,
            "lr": _cfg.body_lr,
            "name": key,
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": _cfg.head_lr,
            "name": "head",
        })

    optimizer = FusedQuantizedAdam(
        param_groups,
        betas=_cfg.betas,
        eps=_cfg.eps,
        weight_decay=_cfg.weight_decay,
        block_size=_cfg.block_size,
    )
    if offload and hasattr(model, '_offload_manager'):
        optimizer.enable_offload(model._offload_manager)
    return model, optimizer


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


METHODS = [
    ("phr",     "PHR (ours)",       build_phr),
    ("full",    "Full Fine-tune",   build_full_finetune),
    ("bitfit",  "BitFit",           build_bitfit),
    ("lora",    "LoRA (r=8)",       build_lora),
    ("qlora",   "QLoRA (8-bit)",    build_qlora),
]
