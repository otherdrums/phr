"""Shared training/eval loop — heartbeat every 100 batches, real VRAM only."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .memory_tracker import MemoryTracker, gpu_used_mb


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    epoch,
    device,
    acc_steps=4,
    val_loader=None,
    val_steps=100,
    tracker=None,
    criterion=None,
    scheduler=None,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if tracker is None:
        tracker = MemoryTracker()

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    val_accuracies = {}
    epoch_start = time.time()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=ids, attention_mask=mask)
        loss = criterion(outputs.logits, labels) / acc_steps
        loss.backward()

        total_loss += loss.item() * acc_steps
        correct += (outputs.logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % acc_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            tracker.step()

        # Heartbeat every 100 micro-batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - epoch_start
            running_loss = total_loss / (batch_idx + 1)
            running_acc = 100.0 * correct / total
            vram = gpu_used_mb()
            print(
                f"  step {batch_idx+1:05d} | "
                f"loss {running_loss:.4f} | acc {running_acc:.2f}% | "
                f"VRAM {vram:.0f}MB | {elapsed:.0f}s"
            )

        # Validate on schedule
        if (batch_idx + 1) % val_steps == 0 and val_loader is not None:
            acc = evaluate(model, val_loader, device)
            val_accuracies[batch_idx + 1] = acc
            print(f"  -- val acc {acc:.2f}%")

    if (batch_idx + 1) % acc_steps != 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        tracker.step()

    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    peak_vram = gpu_used_mb() / 1024

    return train_loss, train_acc, val_accuracies, peak_vram


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=ids, attention_mask=mask)
        correct += (outputs.logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return 100.0 * correct / total if total > 0 else 0.0
