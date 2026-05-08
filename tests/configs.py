"""Method-specific model builders for the SST-2 comparison."""

import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from phr import compress_model, PHRConfig, FusedQuantizedAdam
from .training_config import TrainingConfig, method_lr_config

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

SHARED_SEED = 42
_MODEL_KWARGS = {"local_files_only": True}
_cfg = TrainingConfig()


def build_full_finetune():
    """Vanilla BERT — all parameters trainable."""
    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()
    return model, None


def build_bitfit():
    """Bias-only fine-tuning — freeze all except biases."""
    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()
    for name, param in model.named_parameters():
        if "bias" not in name and "classifier" not in name:
            param.requires_grad = False
    return model, None


def build_lora():
    """LoRA adapters on attention Q+V projections, r=8, α=r per Hu et al. 2021."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
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


def build_qlora():
    """8-bit quantized BERT with LoRA adapters, r=8, α=r."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    torch.manual_seed(SHARED_SEED)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["classifier"],
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
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


def build_phr(offload_level=0):
    """PHR-compressed FFN layers with 8-bit Adam."""
    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    phr_cfg = PHRConfig(
        scheme=_cfg.scheme,
        layer_scope=_cfg.layer_scope,
        learnable_lut=_cfg.learnable_lut,
        gradient_checkpointing=_cfg.gradient_checkpointing,
        offload_level=offload_level,
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
    if offload_level >= 1 and hasattr(model, '_offload_manager'):
        optimizer.enable_offload(model._offload_manager, offload_level)
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


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


METHODS = [
    ("phr",     "PHR (ours)",       build_phr),
    ("full",    "Full Fine-tune",   build_full_finetune),
    ("bitfit",  "BitFit",           build_bitfit),
    ("lora",    "LoRA (r=8)",       build_lora),
    ("qlora",   "QLoRA (8-bit)",    build_qlora),
]
