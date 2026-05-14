"""Unified fine-tuning comparison harness."""

# === Silence everything ===
import sys, os, gc, io, time, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import subprocess
from contextlib import redirect_stderr
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset
import transformers, datasets
transformers.logging.set_verbosity_error()
datasets.disable_progress_bar()

from .configs import METHODS, build_optimizer, count_trainable, build_phr_velvet
from .training import train_one_epoch, evaluate, evaluate_regression, evaluate_mcc
from .memory_tracker import MemoryTracker, gpu_used_mb
from .training_config import TrainingConfig
from torch.optim.lr_scheduler import LinearLR, SequentialLR, LambdaLR
from packr import VelvetController
import math

_TASK_META = {
    "sst2": {
        "num_labels": 2,
        "dataset_name": "sst2",
        "train_samples": 42190,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["sentence"],
    },
    "mnli": {
        "num_labels": 3,
        "dataset_name": "mnli",
        "train_samples": 392702,
        "val_splits": ["validation_matched", "validation_mismatched"],
        "task_type": "classification",
        "input_cols": ["premise", "hypothesis"],
    },
    "cola": {
        "num_labels": 2,
        "dataset_name": "cola",
        "train_samples": 8551,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["sentence"],
    },
    "mrpc": {
        "num_labels": 2,
        "dataset_name": "mrpc",
        "train_samples": 3668,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["sentence1", "sentence2"],
    },
    "qqp": {
        "num_labels": 2,
        "dataset_name": "qqp",
        "train_samples": 363846,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["question1", "question2"],
    },
    "qnli": {
        "num_labels": 2,
        "dataset_name": "qnli",
        "train_samples": 104743,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["question", "sentence"],
    },
    "rte": {
        "num_labels": 2,
        "dataset_name": "rte",
        "train_samples": 2490,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["sentence1", "sentence2"],
    },
    "wnli": {
        "num_labels": 2,
        "dataset_name": "wnli",
        "train_samples": 635,
        "val_splits": ["validation"],
        "task_type": "classification",
        "input_cols": ["sentence1", "sentence2"],
    },
    "stsb": {
        "num_labels": 1,
        "dataset_name": "stsb",
        "train_samples": 5749,
        "val_splits": ["validation"],
        "task_type": "regression",
        "input_cols": ["sentence1", "sentence2"],
    },
}


def _cosine_factor(total_steps: int, min_factor: float):
    """Return a LambdaLR-compatible function: 1.0 → min_factor over total_steps."""
    def fn(step: int) -> float:
        if step >= total_steps:
            return min_factor
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * step / total_steps))
    return fn


from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=os.path.dirname(os.path.dirname(__file__)),
).decode().strip()
RUN_ID = f"{_COMMIT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _save_model(model, method_key, metrics, idle_vram, num_labels=2, task="sst2", seed=42):
    """Save model, metrics, and baked standard HuggingFace model."""
    import json
    from packr import PHRLinear

    out_dir = os.path.join(RESULTS_DIR, f"{task}_{method_key}_seed{seed}_{RUN_ID}")
    os.makedirs(out_dir, exist_ok=True)

    # Save state dict
    torch.save(model.state_dict(), os.path.join(out_dir, "state_dict.pt"))

    # Save metrics
    metrics["idle_vram_mb"] = idle_vram
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Bake to standard HuggingFace model (all methods)
    _bake_model(model, method_key, out_dir, num_labels)

    # PHR-specific: dump layer stats
    if method_key == "packr":
        _dump_phr_stats(model, out_dir)

    print(f"  Saved: {out_dir}")


def _bake_model(model, method_key, out_dir, num_labels=2):
    """Produce a standard HuggingFace baked/ model from the trained model."""
    import torch.nn as nn
    from packr import PHRLinear

    baked_dir = os.path.join(out_dir, "baked")
    os.makedirs(baked_dir, exist_ok=True)

    if method_key == "packr":
        _bake_phr(model, out_dir)
    elif method_key in ("lora", "qlora"):
        try:
            merged = model.merge_and_unload()
            merged.save_pretrained(baked_dir)
        except Exception:
            # bitsandbytes merged model .state_dict() may fail — build clean model
            from transformers import BertForSequenceClassification
            merged = model.merge_and_unload()
            clean = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels, ignore_mismatched_sizes=True,
                local_files_only=True,
            )
            merged_sd = {}
            for name, param in merged.named_parameters():
                merged_sd[name] = param.detach().cpu().clone()
            for name, buf in merged.named_buffers():
                merged_sd[name] = buf.detach().cpu().clone()
            clean.load_state_dict(merged_sd, strict=False)
            clean.save_pretrained(baked_dir)
        print(f"  Baked standard model: {baked_dir}")
    else:
        model.save_pretrained(baked_dir)
        print(f"  Baked standard model: {baked_dir}")


def _bake_phr(model, out_dir):
    """Materialize PHR weights → standard nn.Linear and save via save_pretrained."""
    import copy
    import torch.nn as nn
    from packr import PHRLinear

    baked_dir = os.path.join(out_dir, "baked")
    os.makedirs(baked_dir, exist_ok=True)
    baked = copy.deepcopy(model)

    for name, module in list(baked.named_modules()):
        if isinstance(module, PHRLinear):
            w_full = (module.W_f + module.lut[module.W_p.long()]).t().contiguous()
            linear = nn.Linear(module.in_features, module.out_features,
                              bias=module.bias_f is not None)
            linear.weight.data = w_full.to(linear.weight.dtype)
            if module.bias_f is not None:
                linear.bias.data = module.bias_f.to(linear.bias.dtype)

            parent = baked
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], linear)

    baked.save_pretrained(baked_dir)
    print(f"  Baked standard model: {baked_dir}")


def _dump_phr_stats(model, out_dir):
    """Per-layer PHR statistics: LUT usage, residual magnitude, viability."""
    import json
    from packr import PHRLinear

    layer_stats = {}
    for name, m in model.named_modules():
        if isinstance(m, PHRLinear):
            W_p = m.W_p.long()
            W_f = m.W_f.float()
            lut_vals = m.lut[W_p]
            residual_abs = W_f.abs().mean().item()
            lut_abs = lut_vals.abs().mean().item()
            # LUT entry usage histogram
            usage = torch.bincount(W_p.flatten(), minlength=256).cpu().tolist()
            used_entries = sum(1 for u in usage if u > 0)

            layer_stats[name] = {
                "shape": [m.in_features, m.out_features],
                "lut_entries_used": used_entries,
                "lut_usage_histogram": usage,
                "mean_abs_residual": residual_abs,
                "mean_abs_lut": lut_abs,
                "residual_ratio": residual_abs / max(lut_abs, 1e-10),
            }

    with open(os.path.join(out_dir, "layer_stats.json"), "w") as f:
        json.dump(layer_stats, f, indent=2)
    print(f"  Layer stats: {len(layer_stats)} layers")


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run(quick=False, method_filter=None, epochs=5, offload=False, velvet=False, task="sst2", seed=42):
    cfg = TrainingConfig(epochs=epochs)
    if velvet:
        cfg.velvet_enabled = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    idle_vram = gpu_used_mb()
    print(f"\n{'='*55}")
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB)")
    print(f"  Idle VRAM: {idle_vram:.0f} MB")
    if quick:
        print(f"  MODE: quick (10 batches)")
    if method_filter:
        print(f"  Filter: {method_filter}")
    print(f"{'='*55}\n")

    # ---- Shared data ----
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)

    meta = _TASK_META[task]
    num_labels = meta["num_labels"]

    is_regression = meta.get("task_type") == "regression"
    is_cola = task == "cola"
    input_cols = meta["input_cols"]

    def tok(batch):
        texts = [batch[col] for col in input_cols]
        return tokenizer(
            *texts, truncation=True, padding="max_length", max_length=128
        )

    dataset = load_dataset("glue", meta["dataset_name"])
    dataset = dataset.map(tok, batched=True).with_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True, num_workers=0)
    val_loaders = {s: DataLoader(dataset[s], batch_size=32, shuffle=False, num_workers=0) for s in meta["val_splits"]}
    val_loader = val_loaders[meta["val_splits"][0]]

    EPOCHS = epochs
    ACC_STEPS = 4
    train_steps = len(train_loader)
    val_steps = max(train_steps // 10, 50)

    if quick:
        print(f"Task: {task} ({num_labels} labels), {train_steps} train batches, 10 batch quick test\n")
    else:
        print(f"Task: {task} ({num_labels} labels), Train batches: {train_steps}, Effective batch: 32, Epochs: {EPOCHS}\n")

    # ---- Filter methods ----
    methods_to_run = METHODS
    if method_filter:
        methods_to_run = [(k, n, f) for k, n, f in METHODS if k == method_filter]

    results = {}

    for method_key, method_name, build_fn in methods_to_run:
        print(f"{'='*60}")
        print(f"  {method_name}")
        print(f"{'='*60}")

        _cleanup()
        method_idle_vram = gpu_used_mb()
        tracker = MemoryTracker()
        tracker.reset()

        model = None
        optimizer = None

        try:
            # Suppress model loading progress bars (stderr)
            with redirect_stderr(io.StringIO()):
                if method_key == "packr" and cfg.velvet_enabled:
                    model, prebuilt_opt = build_phr_velvet(offload=offload, num_labels=num_labels, seed=seed)
                elif method_key == "phr":
                    model, prebuilt_opt = build_fn(offload=offload, num_labels=num_labels, seed=seed)
                else:
                    model, prebuilt_opt = build_fn(num_labels=num_labels, seed=seed)
            # Skip model.to() when offloading — compress_model handles CUDA move
            if not hasattr(model, '_offload_manager'):
                model.to(device)
            tracker.snapshot_model()

            optimizer = build_optimizer(model, method_key, prebuilt_opt)
            tracker.step()

            trainable = count_trainable(model)
            model_vram = gpu_used_mb() - method_idle_vram
            print(f"  Params:      {trainable/1e6:.2f} M")
            print(f"  Model VRAM:  {model_vram:.0f} MB")

            steps_per_epoch = len(train_loader)
            total_micro_steps = steps_per_epoch * cfg.epochs
            warmup_steps = int(cfg.warmup_fraction * total_micro_steps)

            velvet_ctrl = None
            scheduler = None

            if cfg.velvet_enabled and method_key == "packr":
                velvet_ctrl = VelvetController(
                    optimizer,
                    beta=None,
                    min_multiplier=None,
                    max_multiplier=cfg.velvet_max_multiplier,
                    velocity_scale=None,
                    train_samples=meta["train_samples"],
                )
                vs = velvet_ctrl.get_stats()
                print(f"  Velvet:      enabled (β={vs['beta']:.3f}, "
                      f"v_ref_β={vs.get('v_ref_beta',0):.3f}, "
                      f"vel_scale={vs['vel_scale']:.1f}, "
                      f"min_m={vs['min_multiplier']:.3f}, "
                      f"obs={vs.get('observation_steps',1)}, "
                      f"gran={cfg.velvet_granularity}, "
                      f"{len(optimizer.param_groups)} groups, "
                      f"{meta['train_samples']} samples)")
            else:
                hold_start = int(cfg.hold_fraction * total_micro_steps)
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[
                        LinearLR(optimizer, start_factor=cfg.warmup_start_factor, end_factor=1.0, total_iters=warmup_steps),
                        LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=hold_start - warmup_steps),
                        LambdaLR(optimizer, lr_lambda=_cosine_factor(total_micro_steps - hold_start, cfg.decay_eta_min)),
                    ],
                    milestones=[warmup_steps, hold_start],
                )

            total_time = 0
            final_val_acc = 0
            final_train_acc = 0
            best_val_acc = 0
            peak_torch = 0
            peak_nvml = 0
            oom = False
            mismatched_acc = None

            if velvet and method_key == "packr":
                method_key = "packr_va"

            out_dir = os.path.join(RESULTS_DIR, f"{task}_{method_key}_seed{seed}_{RUN_ID}")
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, "training_log.jsonl")

            if quick:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                t0 = time.time()
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= 10:
                        break
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(input_ids=ids, attention_mask=mask)
                    if is_regression:
                        loss = nn.functional.mse_loss(outputs.logits, labels.float().unsqueeze(-1)) / ACC_STEPS
                    else:
                        loss = nn.functional.cross_entropy(outputs.logits, labels) / ACC_STEPS
                    loss.backward()
                    if (batch_idx + 1) % ACC_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        tracker.step()
                total_time = time.time() - t0
                peak_torch = tracker.peak_torch_gb()
                peak_nvml = tracker.peak_nvml_gb()
                # Quick eval on subset
                model.eval()
                if is_regression:
                    quick_val = 0
                    for i, batch in enumerate(val_loader):
                        if i >= 5:
                            break
                        ids = batch["input_ids"].to(device)
                        mask = batch["attention_mask"].to(device)
                        with torch.no_grad():
                            outputs = model(input_ids=ids, attention_mask=mask)
                    final_val_acc = 0
                else:
                    correct_q = 0
                    total_q = 0
                    for i, batch in enumerate(val_loader):
                        if i >= 5:
                            break
                        ids = batch["input_ids"].to(device)
                        mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)
                        with torch.no_grad():
                            outputs = model(input_ids=ids, attention_mask=mask)
                        correct_q += (outputs.logits.argmax(-1) == labels).sum().item()
                        total_q += labels.size(0)
                    final_val_acc = 100.0 * correct_q / total_q if total_q > 0 else 0
                best_val_acc = final_val_acc
                model.train()
                print(f"  Quick:       {total_time:.1f}s, val {final_val_acc:.2f}%")
            else:
                criterion = nn.MSELoss() if is_regression else None
                for epoch in range(1, EPOCHS + 1):
                    t0 = time.time()
                    train_loss, train_acc, val_accs, vram_torch = train_one_epoch(
                        model, train_loader, optimizer, epoch, device,
                        acc_steps=ACC_STEPS, val_loader=val_loader,
                        val_steps=val_steps, tracker=tracker,
                        scheduler=scheduler, cv2lrt=velvet_ctrl,
                        warmup_steps=warmup_steps,
                        steps_per_epoch=steps_per_epoch,
                        log_path=log_path,
                        criterion=criterion,
                        is_regression=is_regression,
                        is_cola=is_cola,
                    )
                    epoch_time = time.time() - t0
                    total_time += epoch_time
                    final_train_acc = train_acc
                    peak_torch = max(peak_torch, vram_torch)
                    peak_nvml = max(peak_nvml, tracker.peak_nvml_gb())
                    cur_val = list(val_accs.values())[-1] if val_accs else 0
                    best_val_acc = max(best_val_acc, cur_val)
                    if is_cola:
                        metric_name = "MCC"
                    elif is_regression:
                        metric_name = "Corr"
                    else:
                        metric_name = "Acc"
                    print(f"  Epoch {epoch} | Loss {train_loss:.4f} | "
                          f"{metric_name} {train_acc:.2f}% | Val {cur_val:.2f}% | "
                          f"Best {best_val_acc:.2f}% | {epoch_time:.0f}s")

                if is_cola:
                    final_val_acc = evaluate_mcc(model, val_loader, device)
                elif is_regression:
                    final_val_acc = evaluate_regression(model, val_loader, device)
                else:
                    final_val_acc = evaluate(model, val_loader, device)
                best_val_acc = max(best_val_acc, final_val_acc)
                if task == "mnli":
                    mismatched_acc = evaluate(model, val_loaders["validation_mismatched"], device)
                    best_val_acc = max(best_val_acc, mismatched_acc)
                    print(f"  MNLI Mismatched: {mismatched_acc:.2f}%")

            if is_cola:
                val_key = "val_mcc"
            elif is_regression:
                val_key = "val_corr"
            else:
                val_key = "val_acc"
            result_entry = {
                "name": method_name,
                "trainable_m": trainable / 1e6,
                "train_acc": final_train_acc,
                val_key: best_val_acc,
                "vram_gb": peak_nvml,
                "total_time_s": total_time,
                "status": "OK",
                "task_type": meta["task_type"],
            }
            if mismatched_acc is not None:
                result_entry["val_acc_mismatched"] = mismatched_acc
            results[method_key] = result_entry
            if is_cola:
                metric_label = "MCC"
            elif is_regression:
                metric_label = "Corr"
            else:
                metric_label = "Val"
            print(f"  >> {metric_label}: {final_val_acc:.2f}% | Best: {best_val_acc:.2f}% | VRAM: {peak_nvml:.2f} GB")

            # Save model + metrics + PHR artifacts
            save_metrics = {
                val_key: best_val_acc,
                f"final_{val_key}": final_val_acc,
                "val_acc": best_val_acc,
                "final_val_acc": final_val_acc,
                "train_acc": final_train_acc,
                "peak_vram_gb": peak_nvml,
                "trainable_params_m": trainable / 1e6,
                "total_time_s": total_time,
                "epochs": EPOCHS,
                "offload": offload,
                "format_version": 2,
                "training_config": cfg.to_dict(),
                "task": task,
                "task_type": meta["task_type"],
                "num_labels": num_labels,
                "seed": seed,
            }
            if mismatched_acc is not None:
                save_metrics["val_acc_mismatched"] = mismatched_acc
            if velvet_ctrl is not None:
                vs = velvet_ctrl.get_stats()
                save_metrics["velvet"] = {
                    "enabled": True,
                    "beta": vs["beta"],
                    "min_multiplier": vs["min_multiplier"],
                    "max_multiplier": vs["max_multiplier"],
                    "velocity_scale": vs["vel_scale"],
                    "auto_tuned": vs.get("auto_tuned", False),
                    "granularity": cfg.velvet_granularity,
                    "num_groups": len(optimizer.param_groups),
                    "final_stats": vs,
                }
            _save_model(model, method_key, save_metrics, method_idle_vram, num_labels=num_labels, task=task, seed=seed)

        except torch.cuda.OutOfMemoryError as e:
            print(f"  >> OOM")
            _cleanup()
            results[method_key] = {"name": method_name, "status": "OOM"}
        except Exception as e:
            print(f"  >> ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            results[method_key] = {"name": method_name, "status": f"ERR: {type(e).__name__}"}
        finally:
            del model, optimizer
            _cleanup()

    # ---- Tables ----
    task_label = task.upper()
    print(f"\n\n{'='*100}")
    print(f"  FULL-WEIGHT TRAINING  ({task_label} - same parameters, different storage)")
    print(f"{'='*100}")
    _print_table({k: r for k, r in results.items() if k in ("full", "packr", "packr_va")})

    print(f"\n{'='*100}")
    print(f"  PARAMETER-EFFICIENT TRAINING  ({task_label} - fewer trainable parameters)")
    print(f"{'='*100}")
    _print_table({k: r for k, r in results.items() if k in ("lora", "qlora", "bitfit")})

    print(f"\n  Acc/GB = Val Accuracy / Peak VRAM GB (higher = more memory-efficient)\n")


def _print_table(entries):
    if not entries:
        print("  (no results)")
        return
    first = list(entries.values())[0]
    if first.get("task_type") == "regression":
        val_label = "Val Corr"
    elif "val_mcc" in first:
        val_label = "Val MCC"
    else:
        val_label = "Val Acc"
    header = (
        f"{'Method':<22} {'Status':<8} {'Train M':>8} "
        f"{'VRAM':>8} {val_label:>9} {'Time':>7} {'Acc/GB':>8}"
    )
    print(header)
    print("-" * 90)
    for r in entries.values():
        if r["status"] == "OK":
            val = r.get("val_corr", r.get("val_acc", 0))
            eff = val / max(r["vram_gb"], 0.01)
            print(
                f"{r['name']:<22} {'OK':<8} {r['trainable_m']:>7.1f}M "
                f"{r['vram_gb']:>6.2f} GB {val:>7.2f}% "
                f"{r['total_time_s']:>5.0f}s {eff:>7.1f}"
            )
            if "val_acc_mismatched" in r:
                print(f"  {'':22} {'':8} {'':>8} {'':>8} Mismatched: {r['val_acc_mismatched']:>5.2f}%")
        else:
            print(f"{r['name']:<22} {r['status']:<8} {'-':>8} {'-':>8} {'-':>9} {'-':>7} {'-':>8}")
    print()


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_all = "--all" in sys.argv
    offload = "--offload" in sys.argv
    velvet = "--velvet" in sys.argv
    method_filter = None
    epochs = 5
    task = "sst2"
    seed = 42
    for arg in sys.argv:
        if arg.startswith("--method="):
            method_filter = arg.split("=")[1]
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=")[1])
        elif arg.startswith("--task="):
            task = arg.split("=")[1]
        elif arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
    if run_all:
        method_filter = None
    run(quick=quick, method_filter=method_filter, epochs=epochs, offload=offload, velvet=velvet, task=task, seed=seed)
