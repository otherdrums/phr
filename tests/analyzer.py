"""Result analyzer — cross-method comparison with tables, metrics, and plots.

Generates industry-standard comparisons from results/ directories.
Outputs go to results/_analysis/ by default.

Usage:
    python -m tests.analyzer
    python -m tests.analyzer --results-dir results/
    python -m tests.analyzer --no-plot
"""

import sys, os, json, re, math
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# ── Data model ──

@dataclass
class RunData:
    method: str
    task: str
    seed: int
    commit: str
    timestamp: str
    directory: str
    val_acc: float = 0.0
    final_val_acc: float = 0.0
    val_acc_mismatched: float = 0.0
    train_acc: float = 0.0
    peak_vram_gb: float = 0.0
    trainable_params_m: float = 0.0
    total_time_s: float = 0.0
    epochs: int = 5
    idle_vram_mb: float = 0.0
    offload: bool = False
    training_config: dict = field(default_factory=dict)
    layer_stats: dict = field(default_factory=dict)

    # Computed
    acc_per_gb: float = 0.0
    acc_per_m_params: float = 0.0
    samples_per_sec: float = 0.0
    vram_saved_vs_full_pct: float = 0.0
    acc_gap_vs_full_pct: float = 0.0


def _parse_dirname(dirname: str):
    """Parse {task}_{method}_seed{seed}_{commit}_{timestamp}"""
    m = re.match(r'^(\w+?)_(.+)_seed(\d+)_([a-f0-9]+)_(\d{8}_\d{6})$', dirname)
    if not m:
        return None, None, None, None, None
    task, method, seed_str, commit, timestamp = m.groups()
    return task, method, int(seed_str), commit, timestamp


def load_runs(results_dir: Path) -> list[RunData]:
    """Scan results/ for run directories and load their metrics."""
    runs: list[RunData] = []
    full_vram = None

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        task, method, seed, commit, timestamp = _parse_dirname(d.name)
        if method is None:
            continue

        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        rd = RunData(
            method=method,
            task=task,
            seed=seed,
            commit=commit,
            timestamp=timestamp,
            directory=str(d),
            val_acc=metrics.get("val_acc", 0),
            final_val_acc=metrics.get("final_val_acc", 0),
            val_acc_mismatched=metrics.get("val_acc_mismatched", 0),
            train_acc=metrics.get("train_acc", 0),
            peak_vram_gb=metrics.get("peak_vram_gb", 0),
            trainable_params_m=metrics.get("trainable_params_m", 0),
            total_time_s=metrics.get("total_time_s", 0),
            epochs=metrics.get("epochs", 5),
            idle_vram_mb=metrics.get("idle_vram_mb", 0),
            offload=metrics.get("offload", False),
            training_config=metrics.get("training_config", {}),
        )

        # Derived metrics
        rd.acc_per_gb = rd.val_acc / max(rd.peak_vram_gb, 0.01)
        rd.acc_per_m_params = rd.val_acc / max(rd.trainable_params_m, 0.001)
        _TRAIN_SAMPLES = {"sst2": 42190, "mnli": 392702,
                         "cola": 8551, "mrpc": 3668, "qqp": 363846,
                         "qnli": 104743, "rte": 2490, "wnli": 635, "stsb": 5749}
        rd.samples_per_sec = (_TRAIN_SAMPLES.get(rd.task, 42190) * rd.epochs) / max(rd.total_time_s, 1)

        # Load layer_stats for PackR runs
        layer_path = d / "layer_stats.json"
        if layer_path.exists():
            with open(layer_path) as f:
                rd.layer_stats = json.load(f)

        runs.append(rd)

    # Compute relative metrics vs full fine-tune
    full_runs = [r for r in runs if r.method == "full"]
    if full_runs:
        full_run = full_runs[0]
        full_vram = full_run.peak_vram_gb
        full_acc = full_run.val_acc
        for rd in runs:
            if rd.method != "full":
                rd.vram_saved_vs_full_pct = ((full_vram - rd.peak_vram_gb) / full_vram) * 100
                rd.acc_gap_vs_full_pct = rd.val_acc - full_acc

    return runs


# ── Markdown tables ──

def _b(val):
    return f"**{val}**"


def generate_comparison_table(runs: list[RunData]) -> str:
    """Primary comparison table sorted by val_acc descending."""
    sorted_runs = sorted(runs, key=lambda r: r.val_acc, reverse=True)

    lines = [
        "## Method Comparison (sorted by validation accuracy)",
        "",
        "| Method | Task | Val Acc | Train Acc | VRAM (GB) | Params (M) | Time (h) | Offload | Acc/GB | Acc/M | ",
        "|--------|------|--------:|----------:|:--------:|:---------:|:--------:|:------:|:------:|:-----:|",
    ]

    best_val = max(r.val_acc for r in sorted_runs)
    best_vram = min(r.peak_vram_gb for r in sorted_runs)
    best_efficiency = max(r.acc_per_gb for r in sorted_runs)

    for rd in sorted_runs:
        val = f"{_b(f'{rd.val_acc:.2f}%')}" if rd.val_acc == best_val else f"{rd.val_acc:.2f}%"
        vram = f"{_b(f'{rd.peak_vram_gb:.2f}')}" if rd.peak_vram_gb == best_vram else f"{rd.peak_vram_gb:.2f}"
        eff = f"{_b(f'{rd.acc_per_gb:.1f}')}" if rd.acc_per_gb == best_efficiency else f"{rd.acc_per_gb:.1f}"
        off = "✓" if rd.offload else "—"
        line = (
            f"| {rd.method} | {rd.task} | {val} | {rd.train_acc:.2f}% | "
            f"{vram} GB | {rd.trainable_params_m:.1f}M | "
            f"{rd.total_time_s / 3600:.2f} | {off} | {eff} | {rd.acc_per_m_params:.2f} |"
        )
        lines.append(line)

    return "\n".join(lines)


def generate_efficiency_table(runs: list[RunData]) -> str:
    """Efficiency-focused table with relative savings vs full fine-tune."""
    sorted_runs = sorted(runs, key=lambda r: r.vram_saved_vs_full_pct, reverse=True)

    lines = [
        "",
        "## Efficiency vs Full Fine-tune",
        "",
        "| Method | Val Acc | VRAM Saved | Acc Gap | Acc/GB |",
        "|--------|--------:|:---------:|:-------:|:------:|",
    ]

    for rd in sorted_runs:
        saved = f"{rd.vram_saved_vs_full_pct:.0f}%" if rd.vram_saved_vs_full_pct > 0 else "—"
        gap = f"{rd.acc_gap_vs_full_pct:+.2f}%" if rd.method != "full" else "—"
        lines.append(
            f"| {rd.method} | {rd.val_acc:.2f}% | {saved} | {gap} | {rd.acc_per_gb:.1f} |"
        )

    return "\n".join(lines)


def generate_packr_layer_table(runs: list[RunData]) -> str:
    """Per-layer PackR statistics comparison."""
    packr_runs = [r for r in runs if r.method == "packr" and r.layer_stats]
    if len(packr_runs) < 2:
        return ""

    # Sort by val_acc: better run = "new", worse run = "old"
    packr_runs.sort(key=lambda r: r.val_acc, reverse=True)
    new_run, old_run = packr_runs[0], packr_runs[1]

    lines = [
        "",
        "## PackR Layer Statistics (old vs current)",
        "",
        "| Layer | Old residual_ratio | New residual_ratio | Δ Ratio | Old LUT used | New LUT used |",
        "|-------|-------------------:|-------------------:|:------:|:----------:|:----------:|",
    ]

    layer_names = list(new_run.layer_stats.keys())

    for name in layer_names:
        old_stats = old_run.layer_stats.get(name, {})
        new_stats = new_run.layer_stats.get(name, {})
        if not new_stats:
            continue
        old_ratio = old_stats.get("residual_ratio", 0)
        new_ratio = new_stats.get("residual_ratio", 0)
        delta = new_ratio - old_ratio
        old_used = old_stats.get("lut_entries_used", 0)
        new_used = new_stats.get("lut_entries_used", 0)
        short = name.replace("bert.encoder.layer.", "L").replace(".intermediate.dense", ".int").replace(".output.dense", ".out")
        lines.append(
            f"| {short} | {old_ratio:.4f} | {new_ratio:.4f} | "
            f"{delta:+.4f} | {old_used} | {new_used} |"
        )

    return "\n".join(lines)


def generate_config_table(runs: list[RunData]) -> str:
    """Training configuration comparison."""
    lines = [
        "",
        "## Training Configuration by Run",
        "",
        "| Method | Commit | Betas | Body LR | Head LR | Scheduler |",
        "|--------|--------|:-----:|:-------:|:-------:|-----------|",
    ]
    for rd in runs:
        cfg = rd.training_config
        betas = cfg.get("betas", ["?", "?"])
        scheduler = cfg.get("scheduler", "none")
        lines.append(
            f"| {rd.method} | `{rd.commit}` | ({betas[0]}, {betas[1]}) | "
            f"{cfg.get('body_lr', '?')} | {cfg.get('head_lr', '?')} | {scheduler} |"
        )
    return "\n".join(lines)


def generate_full_markdown(runs: list[RunData]) -> str:
    sections = [
        generate_comparison_table(runs),
        generate_efficiency_table(runs),
        generate_config_table(runs),
        generate_packr_layer_table(runs),
        "",
        "---",
        "",
        "*Charts: see `_analysis/` directory for PNG plots.*",
    ]
    return "\n".join(s for s in sections if s)


# ── JSON summary ──

def generate_json_summary(runs: list[RunData]) -> dict:
    packr_runs = [r for r in runs if r.method == "packr" and r.layer_stats]
    packr_layer_summary = {}
    if packr_runs:
        latest = packr_runs[0]
        layer_avg = {"residual_ratio": 0, "lut_entries_used": 0}
        n = 0
        for stats in latest.layer_stats.values():
            layer_avg["residual_ratio"] += stats.get("residual_ratio", 0)
            layer_avg["lut_entries_used"] += stats.get("lut_entries_used", 0)
            n += 1
        if n > 0:
            layer_avg["residual_ratio"] /= n
            layer_avg["lut_entries_used"] /= n
        packr_layer_summary = {
            "num_layers": n,
            "avg_residual_ratio": round(layer_avg["residual_ratio"], 5),
            "avg_lut_entries_used": round(layer_avg["lut_entries_used"], 1),
            "lut_sparsity_pct": round((256 - layer_avg["lut_entries_used"]) / 2.56, 1),
        }

    return {
        "runs": [
            {
                "method": r.method,
                "commit": r.commit,
                "timestamp": r.timestamp,
                "val_acc": r.val_acc,
                "final_val_acc": r.final_val_acc,
                "val_acc_mismatched": r.val_acc_mismatched,
                "train_acc": r.train_acc,
                "peak_vram_gb": r.peak_vram_gb,
                "trainable_params_m": r.trainable_params_m,
                "total_time_s": r.total_time_s,
                "total_time_h": round(r.total_time_s / 3600, 2),
                "epochs": r.epochs,
                "offload": r.offload,
                "acc_per_gb": round(r.acc_per_gb, 2),
                "acc_per_m_params": round(r.acc_per_m_params, 2),
                "samples_per_sec": round(r.samples_per_sec, 1),
                "vram_saved_vs_full_pct": round(r.vram_saved_vs_full_pct, 1),
                "acc_gap_vs_full_pct": round(r.acc_gap_vs_full_pct, 2),
                "training_config": r.training_config,
            }
            for r in runs
        ],
        "packr_layer_summary": packr_layer_summary,
        "_notes": {
            "acc_gap_vs_full_pct": "Positive = better than full fine-tune, negative = worse",
            "acc_per_gb": "Val accuracy per GB VRAM — higher = more memory-efficient",
            "samples_per_sec": "Training throughput (task-specific train set size)",
        },
    }


# ── Plots ──

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
    })
    return plt


def plot_val_acc_bar(runs: list[RunData], output_dir: Path):
    plt = _setup_mpl()
    sorted_runs = sorted(runs, key=lambda r: r.val_acc, reverse=True)
    methods = [r.method for r in sorted_runs]
    vals = [r.val_acc for r in sorted_runs]
    colors = ["#2ecc71" if m == "packr" else "#3498db" if m == "full" else "#95a5a6" for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.5)

    full_acc = next((r.val_acc for r in sorted_runs if r.method == "full"), None)
    if full_acc:
        ax.axhline(y=full_acc, color="#e74c3c", linestyle="--", linewidth=1, label=f"Full fine-tune ({full_acc:.1f}%)")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy by Method")
    ax.set_ylim(min(vals) - 1.5, max(vals) + 2)
    if full_acc:
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "val_acc_bar.png")
    plt.close(fig)


def plot_efficiency_scatter(runs: list[RunData], output_dir: Path):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(9, 6))

    for rd in runs:
        color = "#2ecc71" if rd.method == "packr" else "#e74c3c" if rd.method == "full" else "#3498db"
        marker = "s" if rd.method == "packr" else "o"
        size = max(rd.trainable_params_m * 1.5, 40) if rd.trainable_params_m > 0 else 60
        ax.scatter(rd.val_acc, rd.acc_per_gb, s=size, c=color, marker=marker,
                   edgecolors="white", linewidth=0.5, zorder=5, alpha=0.85)
        ax.annotate(rd.method, (rd.val_acc, rd.acc_per_gb),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Validation Accuracy (%)")
    ax.set_ylabel("Accuracy per GB VRAM (Acc/GB)")
    ax.set_title("Memory Efficiency vs Accuracy (bubble size = trainable parameters)")
    fig.savefig(output_dir / "efficiency_scatter.png")
    plt.close(fig)


def plot_packr_layer_profile(runs: list[RunData], output_dir: Path):
    """Line chart: residual_ratio per layer for PackR runs."""
    packr_runs = [r for r in runs if r.method == "packr" and r.layer_stats]
    if not packr_runs:
        return

    plt = _setup_mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    colors = ["#2ecc71", "#3498db"]
    labels = [f"PackR ({r.commit[:7]})" for r in packr_runs]

    for idx, rd in enumerate(packr_runs):
        layer_names = list(rd.layer_stats.keys())
        ratios = [rd.layer_stats[n].get("residual_ratio", 0) for n in layer_names]
        entries = [rd.layer_stats[n].get("lut_entries_used", 0) for n in layer_names]
        x = range(len(layer_names))

        ax1.plot(x, ratios, marker=".", color=colors[idx], label=labels[idx], markersize=4)
        ax2.plot(x, entries, marker=".", color=colors[idx], label=labels[idx], markersize=4)

    ax1.set_ylabel("Residual Ratio (W_f / LUT)")
    ax1.set_title("PackR Layer Profile: Residual Contribution vs LUT Entries Used")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("LUT Entries Used (of 256)")
    ax2.set_xlabel("Layer Index (0–23, alternating intermediate/output)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / "packr_layer_profile.png")
    plt.close(fig)


def plot_lut_heatmap(runs: list[RunData], output_dir: Path):
    packr_runs = [r for r in runs if r.method == "packr" and r.layer_stats]
    if not packr_runs:
        return
    rd = packr_runs[0]  # latest PackR run

    plt = _setup_mpl()
    import numpy as np

    layer_names = list(rd.layer_stats.keys())
    n_layers = len(layer_names)
    matrix = np.zeros((256, n_layers))

    for j, name in enumerate(layer_names):
        hist = rd.layer_stats[name].get("lut_usage_histogram", [])
        for i in range(min(256, len(hist))):
            matrix[i, j] = hist[i]

    matrix_log = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix_log, aspect="auto", cmap="YlOrRd", origin="lower",
                   extent=[0, n_layers, 0, 256])

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("LUT Entry Index (0–255)")
    ax.set_title("PackR LUT Entry Usage Heatmap (log scale)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log(1 + usage count)")

    # Mark dead entries
    dead_rows = np.where(matrix.sum(axis=1) == 0)[0]
    for row in dead_rows:
        ax.axhline(y=row + 0.5, color="#e74c3c", linewidth=0.3, alpha=0.5)

    fig.savefig(output_dir / "lut_heatmap.png")
    plt.close(fig)


def plot_dead_entries(runs: list[RunData], output_dir: Path):
    packr_runs = [r for r in runs if r.method == "packr" and r.layer_stats]
    if not packr_runs:
        return
    rd = packr_runs[0]

    plt = _setup_mpl()
    import numpy as np

    layer_names = list(rd.layer_stats.keys())
    dead_counts = np.zeros(256, dtype=int)

    for name in layer_names:
        hist = rd.layer_stats[name].get("lut_usage_histogram", [])
        for i in range(min(256, len(hist))):
            if hist[i] == 0:
                dead_counts[i] += 1

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#e74c3c" if c > len(layer_names) / 2 else "#f39c12" if c > 0 else "#2ecc71" for c in dead_counts]
    bars = ax.bar(range(256), dead_counts, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(y=len(layer_names) / 2, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("LUT Entry Index (0–255)")
    ax.set_ylabel("Number of Layers Where Entry is Dead")
    ax.set_title(f"PackR Dead LUT Entries Across {len(layer_names)} Layers "
                 f"({sum(1 for c in dead_counts if c == len(layer_names))} entries dead in ALL layers)")

    fig.savefig(output_dir / "dead_entries.png")
    plt.close(fig)


def generate_all_plots(runs: list[RunData], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_val_acc_bar(runs, output_dir)
    plot_efficiency_scatter(runs, output_dir)
    plot_packr_layer_profile(runs, output_dir)
    plot_lut_heatmap(runs, output_dir)
    plot_dead_entries(runs, output_dir)


# ── Main ──

def run(results_dir: str = None, output_dir: str = None, skip_plots: bool = False):
    base = Path(results_dir or os.path.join(os.path.dirname(__file__), "..", "results"))
    out = Path(output_dir or (base / "_analysis"))
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(base)
    if not runs:
        print(f"No run directories found in {base}")
        return

    print(f"Loaded {len(runs)} runs:")
    for r in runs:
        print(f"  {r.method:8s}  val={r.val_acc:.2f}%  VRAM={r.peak_vram_gb:.2f}GB  "
              f"params={r.trainable_params_m:.1f}M  commit={r.commit}")

    md = generate_full_markdown(runs)
    (out / "comparison.md").write_text(md)
    print(f"\nMarkdown comparison → {out / 'comparison.md'}")

    summary = generate_json_summary(runs)
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary        → {out / 'metrics.json'}")

    if not skip_plots:
        try:
            generate_all_plots(runs, out)
            print(f"Charts              → {out}/")
        except ImportError as e:
            print(f"[SKIP] Plots not generated — {e}")
            print("       Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    skip_plots = "--no-plot" in sys.argv
    run(skip_plots=skip_plots)
