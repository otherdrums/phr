"""ZPackR Calibration Tool — offline boundary search for ratio threshold.

Reads a diagnostic ratio_log.jsonl and replays candidate salience thresholds
against the recorded per-block compression ratios and delta norms.  Measures
how well each threshold separates "zero-delta" (can prune) from "non-zero
delta" (must keep) blocks without access to GPU.

Produces a recommended threshold and confusion-matrix breakdown.

Usage:
    python tools/calibrate.py runs/sig_probe1_*/ratio_log.jsonl

    # Test specific thresholds
    python tools/calibrate.py runs/sig_probe1_*/ratio_log.jsonl \
        --thresholds 1.5,2.0,3.0,4.0

    # Test gap multipliers  
    python tools/calibrate.py runs/sig_probe1_*/ratio_log.jsonl \
        --multipliers 0.1,0.2,0.5,0.8

    # Include plots
    python tools/calibrate.py runs/sig_probe1_*/ratio_log.jsonl --plot
"""

import os
import sys
import json
import math
import glob
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_ratio_log(path: str) -> List[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _extract_block_signals(entries: List[dict]) -> dict:
    """Extract all per-block (ratio, delta_l2, layer, step) signals from log.

    Returns dict with keys: steps, layers, blocks (list of block dicts).
    """
    steps = []
    block_records = []  # [{step, layer, blk, ratio, delta_l2, salient}]

    for entry in entries:
        step = entry.get("step", 0)
        steps.append(step)
        layers = entry.get("layers", {})
        for layer_name, layer_data in layers.items():
            for block in layer_data.get("blocks", []):
                dl2 = block.get("delta_l2")
                if dl2 is None:
                    dl2 = 0.0
                block_records.append({
                    "step": step,
                    "layer": layer_name,
                    "blk": block["blk"],
                    "ratio": block["ratio"],
                    "delta_l2": dl2,
                    "gap": block.get("gap", block["ratio"]),
                    "salient": block.get("salient", True),
                })

    return {
        "steps": sorted(set(steps)),
        "block_records": block_records,
    }


def _compute_confusion(
    block_records: List[dict],
    threshold_strategy: str,
    threshold_value: float,
    zero_delta_eps: float = 0.001,
) -> dict:
    """Compute confusion matrix for a given threshold across all blocks.

    zero_delta_eps: delta_l2 below this → considered "should be pruned" (near-zero).
    """
    tp = tn = fp = fn = 0
    events = []

    for br in block_records:
        ratio = br["ratio"]
        delta_l2 = br.get("delta_l2", 0.0)
        if delta_l2 is None:
            delta_l2 = 0.0
        is_zero = delta_l2 < zero_delta_eps

        # Decision: ratio >= threshold → prune, ratio < threshold → keep
        keep = ratio < threshold_value

        if is_zero and not keep:
            tn += 1  # correctly pruned
        elif is_zero and keep:
            fp += 1  # zero block kept (wasted VRAM, but no signal lost)
        elif not is_zero and keep:
            tp += 1  # non-zero block kept (correct)
        else:  # not is_zero and not keep
            fn += 1  # non-zero block pruned (LOST SIGNAL — worst)
            events.append(br)

    total = tp + tn + fp + fn
    return {
        "strategy": threshold_strategy,
        "threshold": threshold_value,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "fn_count": fn,
        "fn_samples": events[:5],
        "kept_fraction": (tp + fp) / max(total, 1),
        "vram_mult": (tp + fp) / max(total, 1),
    }


def _compute_per_step_confusion(
    block_records: List[dict],
    threshold: float,
    zero_delta_eps: float = 0.001,
) -> List[dict]:
    """Compute confusion matrix per step."""
    by_step = defaultdict(list)
    for br in block_records:
        by_step[br["step"]].append(br)

    results = []
    for step in sorted(by_step):
        c = _compute_confusion(by_step[step], "fixed", threshold, zero_delta_eps)
        c["step"] = step
        results.append(c)
    return results


def find_best_threshold(
    block_records: List[dict],
    thresholds: Optional[List[float]] = None,
    multipliers: Optional[List[float]] = None,
    zero_delta_eps: float = 0.001,
) -> List[dict]:
    """Test candidate thresholds and return ranked results."""
    results = []

    if thresholds:
        for t in thresholds:
            results.append(_compute_confusion(block_records, "fixed", t, zero_delta_eps))

    if multipliers:
        # Gap-based: threshold = max_ratio * multiplier
        all_ratios = [br["ratio"] for br in block_records]
        max_ratio = max(all_ratios) if all_ratios else 1.0
        for m in multipliers:
            t = max_ratio * m
            strategy = f"max_ratio * {m}"
            results.append(_compute_confusion(block_records, strategy, t, zero_delta_eps))

    results.sort(key=lambda r: (-r["f1"], r["fn"]))
    return results


def _novelty_from_gap(gap: float, gap_max: float, gap_min: float) -> float:
    """Map gap → novelty [0, 1].  High gap → low novelty (known)."""
    span = max(gap_max - gap_min, 1e-8)
    return max(0.0, min(1.0, (gap_max - gap) / span))


def sweep_decay_rates(
    block_records: List[dict],
    decay_rate_options: Optional[List[float]] = None,
    zero_delta_eps: float = 0.001,
) -> List[dict]:
    """Sweep decay_rate values against recorded gap data.

    For each decay_rate, computes: how fast would known blocks shrink?
    Measures mean novelty for zero-delta blocks (should be near 0)
    and non-zero blocks (should be near 1.0).
    """
    if decay_rate_options is None:
        decay_rate_options = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]

    by_step = defaultdict(list)
    for br in block_records:
        by_step[br["step"]].append(br)

    results = []
    for dr in decay_rate_options:
        zero_novelties = []
        nonzero_novelties = []
        zero_decays = []

        for step, blocks in by_step.items():
            gaps = [b["gap"] for b in blocks]
            gap_max = max(gaps)
            gap_min = min(gaps)

            for b in blocks:
                novelty = _novelty_from_gap(b["gap"], gap_max, gap_min)
                decay = (1.0 - novelty) * dr
                dl2 = b.get("delta_l2", 0.0)
                if dl2 is not None and dl2 < zero_delta_eps:
                    zero_novelties.append(novelty)
                    zero_decays.append(decay)
                else:
                    nonzero_novelties.append(novelty)

        zn_mean = sum(zero_novelties) / max(len(zero_novelties), 1)
        nz_mean = sum(nonzero_novelties) / max(len(nonzero_novelties), 1)
        zd_mean = sum(zero_decays) / max(len(zero_decays), 1)

        results.append({
            "decay_rate": dr,
            "known_novelty_mean": zn_mean,
            "novel_novelty_mean": nz_mean,
            "known_decay_mean": zd_mean,
            "spread": nz_mean - zn_mean,
        })

    results.sort(key=lambda r: r["spread"], reverse=True)
    return results


def print_decay_report(sweep_results: List[dict]):
    """Print decay_rate sweep results."""
    print(f"\n{'='*60}")
    print("Decay Rate Sweep (novelty-based attenuation + decay)")
    print(f"{'='*60}")
    print(f"{'decay_rate':>11} {'known_nov':>10} {'novel_nov':>10} "
          f"{'decay/step':>10} {'spread':>10}")
    print("-" * 60)
    for r in sweep_results[:12]:
        print(f"{r['decay_rate']:>11.4f} {r['known_novelty_mean']:>10.4f} "
              f"{r['novel_novelty_mean']:>10.4f} "
              f"{r['known_decay_mean']:>10.4f} {r['spread']:>10.4f}")
    print("-" * 60)
    print("known_nov    = mean novelty for zero-delta blocks (should be near 0)")
    print("novel_nov    = mean novelty for non-zero blocks (should be near 1)")
    print("decay/step   = mean per-step decay applied to known blocks")
    print("spread       = novel_nov - known_nov (higher = better separation)")

    if sweep_results:
        best = sweep_results[0]
        print(f"\nRecommended: decay_rate={best['decay_rate']:.4f}")
        print(f"  known blocks would get {best['known_decay_mean']:.4f} decay/step")
        print(f"  novelty spread: {best['spread']:.4f}")


def find_false_negative_blocks(
    block_records: List[dict],
    threshold: float,
    zero_delta_eps: float = 0.001,
) -> List[dict]:
    """Return blocks that would be incorrectly pruned (false negatives)."""
    fn_blocks = []
    by_step = defaultdict(list)
    by_layer = defaultdict(list)

    for br in block_records:
        if br["delta_l2"] >= zero_delta_eps and br["ratio"] >= threshold:
            fn_blocks.append(br)
            by_step[br["step"]].append(br["layer"])
            by_layer[br["layer"]].append(br["step"])

    return fn_blocks, dict(by_step), dict(by_layer)


def print_report(results: List[dict], block_records: List[dict]):
    """Print a formatted report of calibration results."""
    total_blocks = len(block_records)
    non_zero = sum(1 for br in block_records if br["delta_l2"] >= 0.001)
    zero = total_blocks - non_zero

    print(f"\n{'='*60}")
    print(f"Calibration Report — {total_blocks} blocks ({non_zero} non-zero, {zero} zero)")
    print(f"{'='*60}")

    print(f"\n{'Strategy':<25} {'Thresh':>8} {'F1':>6} {'FN':>5} {'Kept%':>6} {'Acc%':>6}")
    print("-" * 60)

    for r in results:
        fn_marker = " ***" if r["fn"] > 0 else ""
        print(f"{r['strategy']:<25} {r['threshold']:>8.3f} {r['f1']:>6.3f} {r['fn']:>5d} "
              f"{r['kept_fraction']*100:>5.1f}% {r['accuracy']*100:>5.1f}%{fn_marker}")

    print("-" * 60)
    print("FN = false negative (non-zero block incorrectly pruned = lost signal)")
    print("Kept% = fraction of blocks kept in VRAM (lower = better, as long as FN=0)")
    print("*** = lost signal, avoid this threshold")

    if results and results[0]["fn"] == 0:
        best = results[0]
        print(f"\nRecommended: {best['strategy']} (threshold={best['threshold']:.3f})")
        print(f"  F1={best['f1']:.3f}, keeps {best['kept_fraction']*100:.1f}% of blocks, "
              f"zero false negatives")

        # Show how many additional thresholds are viable
        viable = [r for r in results if r["fn"] == 0]
        if len(viable) > 1:
            best_vram = min(viable, key=lambda r: r["kept_fraction"])
            print(f"  Most aggressive w/o signal loss: {best_vram['strategy']} "
                  f"(keeps {best_vram['kept_fraction']*100:.1f}%)")
    else:
        print("\nNo threshold found with zero false negatives.")
        print("The diagnostic run may need more steps for the signal to emerge.")
        print("Try running with --max-steps 500 or increasing the zero_delta_eps threshold.")


def _plot(results: List[dict], block_records: List[dict], path: str):
    """Generate diagnostic plots (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping plots — matplotlib not installed.")
        print("  pip install matplotlib")
        return

    out_dir = os.path.dirname(path)
    out_name = os.path.splitext(os.path.basename(path))[0]

    # Plot 1: FN count vs threshold
    fixed_results = [r for r in results if not r["strategy"].startswith("max_ratio")]
    if fixed_results:
        fixed_results.sort(key=lambda r: r["threshold"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        thresholds = [r["threshold"] for r in fixed_results]
        fn_counts = [r["fn"] for r in fixed_results]
        kept_pcts = [r["kept_fraction"] * 100 for r in fixed_results]
        f1_scores = [r["f1"] for r in fixed_results]

        ax = axes[0]
        ax.bar(range(len(thresholds)), fn_counts, color=["green" if f == 0 else "red" for f in fn_counts])
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:.1f}" for t in thresholds], rotation=45)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("False Negatives (lost signal)")
        ax.set_title("Signal Loss vs Threshold")

        ax2 = axes[1]
        ax2.bar(range(len(thresholds)), kept_pcts, color="steelblue")
        ax2.set_xticks(range(len(thresholds)))
        ax2.set_xticklabels([f"{t:.1f}" for t in thresholds], rotation=45)
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Blocks Kept (%)")
        ax2.set_title("VRAM Usage vs Threshold")

        ax3 = axes[2]
        ax3.plot(thresholds, f1_scores, "o-", color="darkgreen", linewidth=2, markersize=8)
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("F1 Score")
        ax3.set_title("F1 Score vs Threshold")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{out_name}_threshold_sweep.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # Plot 2: Ratio histogram colored by delta_l2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    zero_ratios = [br["ratio"] for br in block_records if br["delta_l2"] < 0.001]
    nonzero_ratios = [br["ratio"] for br in block_records if br["delta_l2"] >= 0.001]

    if zero_ratios or nonzero_ratios:
        ax1 = axes[0]
        bins = 50
        ax1.hist(zero_ratios, bins=bins, alpha=0.6, label=f"Zero delta (n={len(zero_ratios)})",
                 color="steelblue", edgecolor="white")
        ax1.hist(nonzero_ratios, bins=bins, alpha=0.6, label=f"Non-zero delta (n={len(nonzero_ratios)})",
                 color="coral", edgecolor="white")
        ax1.set_xlabel("WeightDict Compression Ratio")
        ax1.set_ylabel("Count")
        ax1.set_title("Ratio Distribution by Delta Magnitude")
        ax1.legend()

        # Mark candidate thresholds
        if fixed_results:
            for r in fixed_results[:3]:
                ax1.axvline(r["threshold"], color="red", linestyle="--", alpha=0.5,
                           linewidth=1, label=f"t={r['threshold']:.1f}" if r == fixed_results[0] else None)

        ax2 = axes[1]
        ax2.scatter([br["ratio"] for br in block_records if br["delta_l2"] > 0],
                    [br["delta_l2"] for br in block_records if br["delta_l2"] > 0],
                    s=2, alpha=0.3, c="coral")
        ax2.axhline(0.001, color="gray", linestyle=":", alpha=0.5, label="zero-delta eps")
        ax2.set_xlabel("WeightDict Compression Ratio")
        ax2.set_ylabel("Delta L2 Norm")
        ax2.set_title("Ratio vs Delta Magnitude")
        ax2.set_yscale("log")

        if fixed_results:
            best = fixed_results[0]
            ax2.axvline(best["threshold"], color="green", linestyle="--", alpha=0.7,
                       linewidth=2, label=f"best t={best['threshold']:.1f}")
            ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{out_name}_ratio_distribution.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot saved: {plot_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ZPackR Calibration Tool — offline threshold boundary search"
    )
    parser.add_argument(
        "ratio_log", nargs="+",
        help="Path(s) to ratio_log.jsonl from diagnostic run(s)"
    )
    parser.add_argument(
        "--thresholds", type=str, default="1.1,1.2,1.3,1.5,2.0,3.0,5.0,10.0",
        help="Comma-separated fixed thresholds to test"
    )
    parser.add_argument(
        "--multipliers", type=str, default="0.01,0.1,0.2,0.3,0.5,0.8",
        help="Comma-separated gap multipliers to test (threshold = max_ratio * m)"
    )
    parser.add_argument(
        "--zero-delta-eps", type=float, default=0.001,
        help="Delta L2 norm below this is considered 'zero' (should be pruned)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate PNG plots (requires matplotlib)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to stdout"
    )
    parser.add_argument(
        "--step-breakdown", action="store_true",
        help="Show per-step confusion matrix for the best threshold"
    )
    parser.add_argument(
        "--sweep-decay", action="store_true",
        help="Sweep decay_rate values for novelty-based attenuation"
    )
    parser.add_argument(
        "--decay-rates", type=str, default="0.001,0.005,0.01,0.025,0.05,0.1,0.2,0.5",
        help="Comma-separated decay_rate values to sweep"
    )
    args = parser.parse_args()

    # Load all ratio logs
    all_entries = []
    for path_pattern in args.ratio_log:
        paths = glob.glob(path_pattern)
        if not paths:
            paths = [path_pattern]
        for p in paths:
            if not os.path.exists(p):
                print(f"File not found: {p}", file=sys.stderr)
                continue
            all_entries.extend(_load_ratio_log(p))

    if not all_entries:
        print("No data loaded. Provide a valid ratio_log.jsonl path.", file=sys.stderr)
        sys.exit(1)

    signals = _extract_block_signals(all_entries)
    block_records = signals["block_records"]

    if not block_records:
        print("No block records found in the ratio log.", file=sys.stderr)
        sys.exit(1)

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    multipliers = [float(m.strip()) for m in args.multipliers.split(",") if m.strip()]

    results = find_best_threshold(
        block_records,
        thresholds=thresholds,
        multipliers=multipliers,
        zero_delta_eps=args.zero_delta_eps,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_report(results, block_records)

    if args.step_breakdown and results:
        best = results[0]
        per_step = _compute_per_step_confusion(
            block_records, best["threshold"], args.zero_delta_eps
        )
        print(f"\nPer-step breakdown for threshold={best['threshold']:.3f}:")
        print(f"{'Step':>6} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'Kept%':>6}")
        for r in per_step:
            if r["total"] > 0:
                print(f"{r['step']:>6} {r['tp']:>5} {r['tn']:>5} {r['fp']:>5} "
                      f"{r['fn']:>5} {r['kept_fraction']*100:>5.1f}%")

    if args.sweep_decay:
        decay_rates = [float(r.strip()) for r in args.decay_rates.split(",") if r.strip()]
        sweep = sweep_decay_rates(
            block_records,
            decay_rate_options=decay_rates,
            zero_delta_eps=args.zero_delta_eps,
        )
        print_decay_report(sweep)

    if args.plot:
        _plot(results, block_records, args.ratio_log[0])


if __name__ == "__main__":
    main()
