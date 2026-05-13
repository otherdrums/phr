"""Prompt encoding — dataset examples → tokenized prompts for unified stream training.

Each prompt is a (input_ids, attention_mask, label) tuple matching the format
the harness uses internally. The label is kept explicit (not embedded in tokens)
so StreamTrainer.step() can directly replicate the harness micro-step.
"""

import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import BertTokenizerFast
    return BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)


def encode_classification(
    text_a: str,
    text_b: str = None,
    label: int = None,
    max_length: int = 128,
):
    """Tokenize a single-sequence or pair-sequence classification example.

    Returns:
        input_ids:       [L] int tensor
        attention_mask:  [L] int tensor (padding mask)
        label:           int or None
    """
    tokenizer = _get_tokenizer()

    if text_b is not None:
        # Pair classification (MNLI)
        encoded = tokenizer(
            text_a, text_b,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        # Single sequence classification (SST-2)
        encoded = tokenizer(
            text_a,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    return (
        encoded["input_ids"][0],
        encoded["attention_mask"][0],
        label,
    )


def ingest_glue(task: str, max_length: int = 128, limit: int = None, seed: int = 42):
    """Convert a GLUE benchmark task into a list of (input_ids, attention_mask, label, text) prompts.

    Args:
        task:       "sst2" or "mnli"
        max_length: max token sequence length
        limit:      cap number of training examples (None = all)
        seed:       random seed for shuffling

    Returns:
        List of (input_ids: Tensor[L], attention_mask: Tensor[L], label: int, text: str)
        for train split.
    """
    from datasets import load_dataset

    ds = load_dataset("glue", task)

    prompts = []
    for ex in ds["train"]:
        if limit is not None and len(prompts) >= limit:
            break

        if task == "sst2":
            text = ex["sentence"]
            ids, mask, label = encode_classification(
                text, label=ex["label"], max_length=max_length
            )
        elif task == "mnli":
            text = ex["premise"] + " [SEP] " + ex["hypothesis"]
            ids, mask, label = encode_classification(
                ex["premise"], ex["hypothesis"], label=ex["label"], max_length=max_length
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        prompts.append((ids, mask, label, text))

    # Shuffle deterministically
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(prompts), generator=rng).tolist()
    prompts = [prompts[i] for i in perm]

    return prompts


def num_labels(task: str) -> int:
    """Return the number of output classes for a task."""
    return {"sst2": 2, "mnli": 3}[task]
