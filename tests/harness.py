"""Unified SST-2 fine-tuning comparison harness."""

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
from contextlib import redirect_stderr
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset
import transformers, datasets
transformers.logging.set_verbosity_error()
datasets.disable_progress_bar()

from .configs import METHODS, build_optimizer, count_trainable
from .training import train_one_epoch, evaluate
from .memory_tracker import MemoryTracker, gpu_used_mb
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_model(model, method_key, metrics, idle_vram):
    """Save model, metrics, and PHR-specific layer stats after training."""
    import json
    from phr import PHRLinear

    out_dir = os.path.join(RESULTS_DIR, f"sst2_{method_key}_seed42_{RUN_ID}")
    os.makedirs(out_dir, exist_ok=True)

    # Save state dict
    torch.save(model.state_dict(), os.path.join(out_dir, "state_dict.pt"))

    # Save metrics
    metrics["idle_vram_mb"] = idle_vram
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # PHR-specific: bake to standard model + dump layer stats
    if method_key == "phr":
        _dump_phr_stats(model, out_dir)
        _bake_phr(model, out_dir)

    print(f"  Saved: {out_dir}")


def _bake_phr(model, out_dir):
    """Materialize PHR weights → standard nn.Linear and save via save_pretrained."""
    import torch.nn as nn
    from phr import PHRLinear

    # Clone to avoid mutating the original
    baked = model

    for name, module in list(baked.named_modules()):
        if isinstance(module, PHRLinear):
            # Materialize
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

    baked_dir = os.path.join(out_dir, "baked")
    baked.save_pretrained(baked_dir)
    print(f"  Baked standard model: {baked_dir}")


def _dump_phr_stats(model, out_dir):
    """Per-layer PHR statistics: LUT usage, residual magnitude, viability."""
    import json
    from phr import PHRLinear

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


def run(quick=False, method_filter=None, epochs=5):
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

    def tok(batch):
        return tokenizer(
            batch["sentence"], truncation=True, padding="max_length", max_length=128
        )

    dataset = load_dataset("glue", "sst2")
    dataset = dataset.map(tok, batched=True).with_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset["validation"], batch_size=32, shuffle=False, num_workers=0)

    EPOCHS = epochs
    ACC_STEPS = 4
    train_steps = len(train_loader)
    val_steps = max(train_steps // 10, 50)

    print(f"Train batches: {train_steps}, Effective batch: 32, Epochs: {EPOCHS}\n")

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
        tracker = MemoryTracker()
        tracker.reset()

        model = None
        optimizer = None

        try:
            # Suppress model loading progress bars (stderr)
            with redirect_stderr(io.StringIO()):
                model, prebuilt_opt = build_fn()
            model.to(device)
            tracker.snapshot_model()

            optimizer = build_optimizer(model, method_key, prebuilt_opt)
            tracker.step()

            trainable = count_trainable(model)
            model_vram = gpu_used_mb() - idle_vram
            print(f"  Params:      {trainable/1e6:.2f} M")
            print(f"  Model VRAM:  {model_vram:.0f} MB")

            steps_per_epoch = (train_steps + ACC_STEPS - 1) // ACC_STEPS
            total_opt_steps = steps_per_epoch * EPOCHS
            warmup_steps = min(total_opt_steps // 10, 300)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                    LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_opt_steps - warmup_steps),
                ],
                milestones=[warmup_steps],
            )

            total_time = 0
            final_val_acc = 0
            final_train_acc = 0
            best_val_acc = 0
            peak_torch = 0
            peak_nvml = 0
            oom = False

            if quick:
                model.train()
                optimizer.zero_grad()
                t0 = time.time()
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= 10:
                        break
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(input_ids=ids, attention_mask=mask)
                    loss = torch.nn.functional.cross_entropy(outputs.logits, labels) / ACC_STEPS
                    loss.backward()
                    if (batch_idx + 1) % ACC_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        tracker.step()
                total_time = time.time() - t0
                peak_torch = tracker.peak_torch_gb()
                peak_nvml = tracker.peak_nvml_gb()
                # Quick eval on subset
                quick_val = 0
                model.eval()
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
                for epoch in range(1, EPOCHS + 1):
                    t0 = time.time()
                    train_loss, train_acc, val_accs, vram_torch = train_one_epoch(
                        model, train_loader, optimizer, epoch, device,
                        acc_steps=ACC_STEPS, val_loader=val_loader,
                        val_steps=val_steps, tracker=tracker,
                        scheduler=scheduler,
                    )
                    epoch_time = time.time() - t0
                    total_time += epoch_time
                    final_train_acc = train_acc
                    peak_torch = max(peak_torch, vram_torch)
                    peak_nvml = max(peak_nvml, tracker.peak_nvml_gb())
                    cur_val = list(val_accs.values())[-1] if val_accs else 0
                    best_val_acc = max(best_val_acc, cur_val)
                    print(f"  Epoch {epoch} | Loss {train_loss:.4f} | "
                          f"Acc {train_acc:.2f}% | Val {cur_val:.2f}% | "
                          f"Best {best_val_acc:.2f}% | {epoch_time:.0f}s")

                final_val_acc = evaluate(model, val_loader, device)
                best_val_acc = max(best_val_acc, final_val_acc)

            results[method_key] = {
                "name": method_name,
                "trainable_m": trainable / 1e6,
                "train_acc": final_train_acc,
                "val_acc": best_val_acc,
                "vram_gb": peak_nvml,
                "total_time_s": total_time,
                "status": "OK",
            }
            print(f"  >> Val: {final_val_acc:.2f}% | Best: {best_val_acc:.2f}% | VRAM: {peak_nvml:.2f} GB")

            # Save model + metrics + PHR artifacts
            _save_model(model, method_key, {
                "val_acc": best_val_acc,
                "final_val_acc": final_val_acc,
                "train_acc": final_train_acc,
                "peak_vram_gb": peak_nvml,
                "trainable_params_m": trainable / 1e6,
                "total_time_s": total_time,
                "epochs": EPOCHS,
            }, idle_vram)

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
    print(f"\n\n{'='*100}")
    print("  FULL-WEIGHT TRAINING  (same parameters, different storage)")
    print(f"{'='*100}")
    _print_table({k: r for k, r in results.items() if k in ("full", "phr")})

    print(f"\n{'='*100}")
    print("  PARAMETER-EFFICIENT TRAINING  (fewer trainable parameters)")
    print(f"{'='*100}")
    _print_table({k: r for k, r in results.items() if k in ("lora", "qlora", "bitfit")})

    print(f"\n  Acc/GB = Val Accuracy / Peak VRAM GB (higher = more memory-efficient)\n")


def _print_table(entries):
    if not entries:
        print("  (no results)")
        return
    header = (
        f"{'Method':<22} {'Status':<8} {'Train M':>8} "
        f"{'VRAM':>8} {'Val Acc':>9} {'Time':>7} {'Acc/GB':>8}"
    )
    print(header)
    print("-" * 90)
    for r in entries.values():
        if r["status"] == "OK":
            eff = r["val_acc"] / max(r["vram_gb"], 0.01)
            print(
                f"{r['name']:<22} {'OK':<8} {r['trainable_m']:>7.1f}M "
                f"{r['vram_gb']:>6.2f} GB {r['val_acc']:>7.2f}% "
                f"{r['total_time_s']:>5.0f}s {eff:>7.1f}"
            )
        else:
            print(f"{r['name']:<22} {r['status']:<8} {'-':>8} {'-':>8} {'-':>9} {'-':>7} {'-':>8}")
    print()


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    method_filter = None
    epochs = 5
    for arg in sys.argv:
        if arg.startswith("--method="):
            method_filter = arg.split("=")[1]
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=")[1])
    run(quick=quick, method_filter=method_filter, epochs=epochs)
