"""Shared training/eval loop — heartbeat every 100 batches, real VRAM only."""

import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .memory_tracker import MemoryTracker, gpu_used_mb


def _append_log(log_path, entry):
    if log_path is None:
        return
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    log_path=None,
    cv2lrt=None,
    warmup_steps=0,
    steps_per_epoch=None,
    is_regression=False,
    is_cola=False,
):
    if criterion is None:
        criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    if tracker is None:
        tracker = MemoryTracker()

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    val_accuracies = {}
    epoch_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    global_step = None  # will be computed per-iteration if cv2lrt is active

    for batch_idx, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=ids, attention_mask=mask)

        if is_regression:
            labels_r = labels.float().unsqueeze(-1)
            loss = criterion(outputs.logits, labels_r) / acc_steps
        else:
            loss = criterion(outputs.logits, labels) / acc_steps
        loss.backward()

        # ---- LR scheduling ----
        global_step = None
        if scheduler is not None:
            scheduler.step()
        elif cv2lrt is not None and steps_per_epoch is not None:
            global_step = (epoch - 1) * steps_per_epoch + batch_idx
            if global_step < warmup_steps:
                cv2lrt.warmup_step(global_step, warmup_steps)

        total_loss += loss.item() * acc_steps
        if is_regression:
            correct += 0
            total += labels.size(0)
        else:
            correct += (outputs.logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        if (batch_idx + 1) % acc_steps == 0:
            optimizer.step()
            # CV2LRT engages after warmup completes
            if cv2lrt is not None and (
                global_step is None or global_step >= warmup_steps
            ):
                cv2lrt.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.step()

        # Heartbeat every 100 micro-batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - epoch_start
            running_loss = total_loss / (batch_idx + 1)
            running_acc = 100.0 * correct / total
            vram = gpu_used_mb()
            entry = {
                "epoch": epoch,
                "step": batch_idx + 1,
                "event": "heartbeat",
                "loss": round(running_loss, 5),
                "vram_mb": vram,
                "elapsed_s": int(elapsed),
            }
            if is_regression:
                entry["train_mse"] = round(running_loss, 5)
            else:
                entry["train_acc"] = round(running_acc, 2)
            if cv2lrt is not None:
                entry["cv2lrt"] = cv2lrt.get_stats()
            velvet_str = ""
            if cv2lrt is not None:
                muls = [g["multiplier"] for g in entry["cv2lrt"]["per_group"].values()]
                if muls:
                    velvet_str = f" | velvet μ={sum(muls)/len(muls):.2f} ↓{min(muls):.2f}"
            metric_label = f"mse {running_loss:.4f}" if is_regression else f"acc {running_acc:.2f}%"
            print(
                f"  step {batch_idx+1:05d} | "
                f"loss {running_loss:.4f} | {metric_label} | "
                f"VRAM {vram:.0f}MB | {elapsed:.0f}s{velvet_str}"
            )
            _append_log(log_path, entry)

        # Validate on schedule
        if (batch_idx + 1) % val_steps == 0 and val_loader is not None:
            if is_regression:
                corr = evaluate_regression(model, val_loader, device)
                val_accuracies[batch_idx + 1] = corr
                entry = {
                    "epoch": epoch,
                    "step": batch_idx + 1,
                    "event": "val",
                    "val_corr": round(corr, 2),
                }
                if cv2lrt is not None:
                    entry["cv2lrt"] = cv2lrt.get_stats()
                vstr = ""
                if cv2lrt is not None:
                    muls = [g["multiplier"] for g in entry["cv2lrt"]["per_group"].values()]
                    if muls:
                        vstr = f" | velvet μ={sum(muls)/len(muls):.2f}"
                print(f"  -- val corr {corr:.2f}{vstr}")
            elif is_cola:
                mcc = evaluate_mcc(model, val_loader, device)
                val_accuracies[batch_idx + 1] = mcc
                entry = {
                    "epoch": epoch,
                    "step": batch_idx + 1,
                    "event": "val",
                    "val_mcc": round(mcc, 2),
                }
                if cv2lrt is not None:
                    entry["cv2lrt"] = cv2lrt.get_stats()
                vstr = ""
                if cv2lrt is not None:
                    muls = [g["multiplier"] for g in entry["cv2lrt"]["per_group"].values()]
                    if muls:
                        vstr = f" | velvet μ={sum(muls)/len(muls):.2f}"
                print(f"  -- val mcc {mcc:.2f}{vstr}")
            else:
                acc = evaluate(model, val_loader, device)
                val_accuracies[batch_idx + 1] = acc
                entry = {
                    "epoch": epoch,
                    "step": batch_idx + 1,
                    "event": "val",
                    "val_acc": round(acc, 2),
                }
                if cv2lrt is not None:
                    entry["cv2lrt"] = cv2lrt.get_stats()
                vstr = ""
                if cv2lrt is not None:
                    muls = [g["multiplier"] for g in entry["cv2lrt"]["per_group"].values()]
                    if muls:
                        vstr = f" | velvet μ={sum(muls)/len(muls):.2f}"
                print(f"  -- val acc {acc:.2f}%{vstr}")
            _append_log(log_path, entry)

    if (batch_idx + 1) % acc_steps != 0:
        optimizer.step()
        if cv2lrt is not None and (global_step is None or global_step >= warmup_steps):
            cv2lrt.step()
        optimizer.zero_grad(set_to_none=True)
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


@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=ids, attention_mask=mask)
        preds.extend(outputs.logits.squeeze(-1).cpu().tolist())
        gts.extend(labels.cpu().tolist())
    model.train()
    from scipy.stats import pearsonr, spearmanr
    pearson, _ = pearsonr(gts, preds)
    spearman, _ = spearmanr(gts, preds)
    return pearson * 100


@torch.no_grad()
def evaluate_mcc(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=ids, attention_mask=mask)
        preds.extend(outputs.logits.argmax(-1).cpu().tolist())
        gts.extend(labels.cpu().tolist())
    model.train()
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(gts, preds) * 100
