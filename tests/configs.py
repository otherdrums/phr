"""Method-specific model builders for the SST-2 comparison."""

import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from phr import compress_model, PHRConfig, FusedQuantizedAdam

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

SHARED_SEED = 42
_MODEL_KWARGS = {"local_files_only": True}


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
    """LoRA adapters on attention Q+V projections, r=8."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model, None


def build_qlora():
    """8-bit quantized BERT with LoRA adapters, r=8."""
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
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model, None


def build_phr():
    """PHR-compressed FFN layers with 8-bit Adam."""
    torch.manual_seed(SHARED_SEED)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, ignore_mismatched_sizes=True,
        **_MODEL_KWARGS,
    )
    config = PHRConfig(
        scheme="phr",
        layer_scope="ffn",
        learnable_lut=True,
        gradient_checkpointing=True,
    )
    model = compress_model(model, config)

    # Separate params for differential LR
    head_params = []
    phr_params = []
    other_params = []
    for n, p in model.named_parameters():
        if "classifier" in n or "cls" in n:
            head_params.append(p)
        elif p.requires_grad:
            phr_params.append(p)

    optimizer = FusedQuantizedAdam(
        [
            {"params": phr_params, "lr": 2e-5},
            {"params": head_params, "lr": 1e-3},
        ],
        betas=(0.9, 0.999),
    )
    return model, optimizer


def build_optimizer(model, method_name, prebuilt_optimizer=None):
    """Create the appropriate optimizer for a given method."""
    if prebuilt_optimizer is not None:
        return prebuilt_optimizer

    head_params = []
    body_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n or "cls" in n:
            head_params.append(p)
        else:
            body_params.append(p)

    if method_name == "full":
        return torch.optim.AdamW(
            [
                {"params": body_params, "lr": 2e-5},
                {"params": head_params, "lr": 1e-3},
            ],
            betas=(0.9, 0.999),
        )
    else:
        return torch.optim.AdamW(
            [
                {"params": body_params, "lr": 2e-5},
                {"params": head_params, "lr": 1e-3},
            ],
            betas=(0.9, 0.999),
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
