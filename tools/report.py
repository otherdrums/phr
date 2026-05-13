"""ZPackR Results Analyzer — load, compare, and visualize training metrics.

Usage:
    python tools/report.py runs/sst2_zpackr
    python tools/report.py runs/ --compare
    python tools/report.py runs/ablation_sweep/ --ablation
"""

import os
import sys
import json
import glob
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_run(path: str) -> dict:
    """Load a single experiment's metrics and config.

    Returns dict with keys: config, metrics (DataFrame-like list), summary, path.
    """
    metrics_path = os.path.join(path, "metrics.jsonl")
    config_path = os.path.join(path, "config.json")
    summary_path = os.path.join(path, "summary.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.jsonl in {path}")

    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    config = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    return {
        "path": os.path.abspath(path),
        "config": config,
        "metrics": metrics,
        "summary": summary,
    }


def load_runs(base_dir: str) -> list[dict]:
    """Load all experiment subdirectories under base_dir."""
    runs = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "metrics.jsonl")):
            try:
                runs.append(load_run(full))
            except Exception:
                pass
    return runs


def metric_series(runs: list[dict], key: str, default=np.nan) -> list:
    """Extract a scalar metric series from step-type entries."""
    series = []
    for run in runs:
        values = []
        for m in run["metrics"]:
            if m.get("type") == "step":
                values.append(m.get(key, default))
        series.append(values)
    return series


def final_metric(runs: list[dict], key: str, default=None):
    """Extract a final value from each run's summary or last eval."""
    values = []
    for run in runs:
        if run["summary"] and key in run["summary"]:
            values.append(run["summary"][key])
        else:
            values.append(default)
    return values


def compare_table(runs: list[dict]) -> str:
    """Generate a markdown comparison table across runs."""
    if not runs:
        return "No runs found."

    headers = ["Run", "Mode", "Steps", "Final Metric", "Gate Skip%", "VRAM (MB)"]
    rows = []

    for i, run in enumerate(runs):
        cfg = run.get("config", {})
        summary = run.get("summary", {})

        mode = cfg.get("packr_config", {}).get("mode", "?")
        name = os.path.basename(run["path"])[:40]
        steps = summary.get("total_steps", "?")
        metric = summary.get("final_eval_metric")
        if isinstance(metric, float):
            metric = f"{metric:.4f}"
        gate_rate = summary.get("gate_skip_rate", 0)
        if isinstance(gate_rate, float):
            gate_rate = f"{gate_rate:.1%}"

        # Get last step VRAM
        vram = "?"
        for m in reversed(run["metrics"]):
            if m.get("type") == "step" and "vram_allocated_mb" in m:
                vram = f"{m['vram_allocated_mb']:.0f}"
                break

        rows.append([name, mode, str(steps), str(metric), gate_rate, vram])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-|-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    return "\n".join(lines)


def plot_loss(runs: list[dict], output_path: Optional[str] = None):
    """Plot loss curves for multiple runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for run in runs:
        name = os.path.basename(run["path"])[:30]
        steps = []
        losses = []
        for m in run["metrics"]:
            if m.get("type") == "step" and "loss" in m:
                steps.append(m["step"])
                losses.append(m["loss"])
        if steps:
            ax.plot(steps, losses, label=name, alpha=0.7, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    else:
        out = os.path.join(os.path.dirname(runs[0]["path"]), "loss_plot.png") if runs else "loss_plot.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(f"  Saved loss plot to {out}")
    plt.close(fig)


def plot_salience(run: dict, output_path: Optional[str] = None):
    """Plot per-layer salience fraction evolution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    # Collect per-layer salience over time
    layer_data = defaultdict(lambda: {"steps": [], "fractions": []})
    for m in run["metrics"]:
        if m.get("type") == "step" and "salience" in m:
            step = m["step"]
            for layer, info in m["salience"].items():
                layer_data[layer]["steps"].append(step)
                layer_data[layer]["fractions"].append(info.get("fraction", 1.0))

    if not layer_data:
        print("  No salience data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for layer, data in sorted(layer_data.items()):
        ax.plot(data["steps"], data["fractions"], label=layer, alpha=0.7, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Salient Fraction")
    ax.set_title("Per-Layer Salience Evolution")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    else:
        out = os.path.join(os.path.dirname(run["path"]), "salience_plot.png")
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(f"  Saved salience plot to {out}")
    plt.close(fig)


def plot_ablation_summary(ablation_dir: str, metric: str = "eval_metric", output_path: Optional[str] = None):
    """Generate a heatmap/table summarizing ablation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    runs = load_runs(ablation_dir)
    if not runs:
        print(f"No runs found in {ablation_dir}")
        return

    # Extract parameter values and final metric
    rows = []
    for run in runs:
        cfg = run.get("config", {})
        summary = run.get("summary", {})
        val = summary.get(f"final_{metric}", summary.get(metric))
        rows.append({
            "mode": cfg.get("packr_config", {}).get("mode", "?"),
            "gate": cfg.get("gate_enabled", False),
            "velvet": cfg.get("velvet_enabled", True),
            "threshold": cfg.get("packr_config", {}).get("zstd_salience_threshold", 2.0),
            "metric": val,
            "vram": summary.get("vram_allocated_mb", 0),
        })

    if not rows:
        return

    # Print summary table
    print(f"\nAblation Summary ({len(runs)} runs):")
    print(f"{'Mode':<8} {'Gate':<6} {'Velvet':<8} {'Thresh':<8} {'Metric':<10}")
    print("-" * 45)
    for r in sorted(rows, key=lambda x: (x["mode"], x.get("threshold", 0))):
        m = r.get("metric", "?")
        print(f"{r['mode']:<8} {str(r['gate']):<6} {str(r['velvet']):<8} {r.get('threshold', '?'):<8} {str(m)[:10]:<10}")

    # Scatter plot: threshold vs metric
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode, marker in [("packr", "o"), ("zpackr", "s")]:
        pts = [r for r in rows if r["mode"] == mode]
        if pts:
            xs = [r.get("threshold", 2.0) for r in pts]
            ys = [r.get("metric", 0) or 0 for r in pts]
            axes[0].scatter(xs, ys, marker=marker, label=mode, s=80)

    axes[0].set_xlabel("Salience Threshold")
    axes[0].set_ylabel(metric)
    axes[0].set_title(f"Ablation: Threshold vs {metric}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # VRAM comparison
    modes = list(set(r["mode"] for r in rows))
    vrams = {m: [r.get("vram", 0) or 0 for r in rows if r["mode"] == m] for m in modes}
    axes[1].bar(modes, [np.mean(v) for v in vrams.values()])
    axes[1].set_title("Average VRAM by Mode")
    axes[1].set_ylabel("VRAM (MB)")

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
    else:
        out = os.path.join(ablation_dir, "ablation_summary.png")
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(f"  Saved ablation plot to {out}")
    plt.close(fig)


def generate_report(path: str):
    """Generate a full HTML report for a run or directory of runs."""
    if os.path.isdir(path):
        runs = load_runs(path)
    else:
        runs = [load_run(path)]

    if not runs:
        print(f"No runs found at {path}")
        return

    print(f"\nLoaded {len(runs)} run(s):")
    for run in runs:
        name = os.path.basename(run["path"])
        steps = run.get("summary", {}).get("total_steps", "?")
        metric = run.get("summary", {}).get("final_eval_metric", "?")
        print(f"  {name}: {steps} steps, final metric={metric}")

    print(f"\n{compare_table(runs)}")

    if len(runs) <= 5:
        plot_loss(runs)
        for run in runs[:3]:
            plot_salience(run)


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZPackR Results Analyzer")
    parser.add_argument("path", help="Path to experiment directory or runs/ root")
    parser.add_argument("--compare", action="store_true", help="Compare all runs in directory")
    parser.add_argument("--ablation", action="store_true", help="Ablation summary mode")
    parser.add_argument("--metric", default="eval_metric", help="Metric to use for ablation")
    parser.add_argument("--output", "-o", default=None, help="Output path for plots")
    args = parser.parse_args()

    if args.ablation:
        plot_ablation_summary(args.path, metric=args.metric, output_path=args.output)
    else:
        generate_report(args.path)


if __name__ == "__main__":
    main()
