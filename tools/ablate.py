"""ZPackR Ablation Runner — systematic parameter sweeps.

Usage:
    python tools/ablate.py --task sst2 --max-steps 500 --sweep config.json
    python tools/ablate.py --task sst2 --max-steps 500 \
        --param mode:packr,zpackr \
        --param velvet_enabled:true,false \
        --param zstd_salience_threshold:1.5,2.0,3.0,5.0
"""

import os
import sys
import json
import itertools
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.train_harness import TrainerConfig, ZPackRTrainer
from packr.config import PackRConfig


def _parse_val(s: str):
    """Parse a parameter value string into the appropriate type."""
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict, path: str, value):
    """Set a value at a dotted path in a nested dict."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested(d: dict, path: str):
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, None)
        else:
            return None
    return d


class AblationRunner:
    """Run a parameter sweep over TrainerConfig fields.

    Produces per-combination output directories under the base output dir.
    """

    def __init__(
        self,
        base_config: TrainerConfig = None,
        sweep: dict = None,
        parallel: bool = False,
        base_output_dir: str = "runs/ablation",
    ):
        if base_config is None:
            base_config = TrainerConfig()
        self.base_config = base_config
        self.sweep = sweep or {}
        self.parallel = parallel
        self.base_output_dir = base_output_dir
        self._combinations = []
        self._results = []

    def add_param(self, path: str, values: list):
        """Add a parameter to sweep over.

        Args:
            path: Dotted path to config field, e.g. 'packr_config.mode'
            values: List of values to try.
        """
        self.sweep[path] = values

    def _build_combinations(self):
        param_names = list(self.sweep.keys())
        param_values = [self.sweep[name] for name in param_names]

        for combo in itertools.product(*param_values):
            cfg = deepcopy(self.base_config)

            # Unpack PackRConfig first to handle nested updates
            pr_dict = cfg.packr_config.__dict__.copy() if hasattr(cfg.packr_config, '__dict__') else asdict(cfg.packr_config)

            for name, val in zip(param_names, combo):
                name = name.replace("_", "-").replace(".", "__sep__")
                name = name.replace("__sep__", ".")

                if name.startswith("packr_config."):
                    field = name.split(".", 1)[1]
                    if field in ["mode", "layer_scope", "scheme"]:
                        setattr(cfg.packr_config, field, val)
                    elif field in ["zstd_super_dict_path"]:
                        setattr(cfg.packr_config, field, val)
                    elif field in ["zstd_max_entries", "zstd_salience_threshold", "zstd_regrow_noise"]:
                        setattr(cfg.packr_config, field, val)
                    elif field in ["block_size", "learnable_lut", "gradient_checkpointing", "offload"]:
                        setattr(cfg.packr_config, field, val)
                elif hasattr(cfg, name):
                    setattr(cfg, name, val)
                elif name in cfg.__dict__:
                    cfg.__dict__[name] = val

            # Build a short label from the parameter combination
            label_parts = []
            for name, val in zip(param_names, combo):
                short_name = name.replace("packr_config.", "").replace("zstd_", "")
                if isinstance(val, bool):
                    label_parts.append(f"{short_name}={int(val)}")
                elif isinstance(val, float):
                    label_parts.append(f"{short_name}={val:.1f}")
                else:
                    label_parts.append(f"{short_name}={val}")
            label = "_".join(label_parts[:4])  # limit label length

            cfg.run_label = label
            self._combinations.append((label, cfg))

    def run(self):
        self._build_combinations()

        print(f"AblationRunner: {len(self._combinations)} combinations")
        print(f"  Output: {self.base_output_dir}")
        print()

        for i, (label, cfg) in enumerate(self._combinations):
            cfg.output_dir = self.base_output_dir
            print(f"[{i+1}/{len(self._combinations)}] {label}")
            try:
                trainer = ZPackRTrainer(cfg)
                result = trainer.run()
                self._results.append({
                    "label": label,
                    "config": {k: str(v) for k, v in cfg.__dict__.items()},
                    "summary": result,
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                self._results.append({
                    "label": label,
                    "error": str(e),
                })

        self._save_manifest()

    def _save_manifest(self):
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "base_output_dir": self.base_output_dir,
            "num_combinations": len(self._combinations),
            "results": self._results,
        }
        path = os.path.join(self.base_output_dir, "ablation_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"\nManifest saved: {path}")

    def summarize(self):
        """Print summary of all results."""
        print(f"\n{'Label':<40} {'Metric':<10} {'Status'}")
        print("-" * 60)
        for r in self._results:
            label = r.get("label", "?")
            if "error" in r:
                print(f"{label:<40} {'FAIL':<10} {r['error'][:40]}")
            else:
                metric = r.get("summary", {}).get("eval_metric", "?")
                print(f"{label:<40} {str(metric):<10} OK")


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZPackR Ablation Runner")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--mode", default="zpackr")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="runs/ablation")
    parser.add_argument("--param", action="append", default=[],
                        help="param:val1,val2,val3  (e.g. --param velvet_enabled:true,false)")
    parser.add_argument("--sweep", default=None,
                        help="JSON file with sweep dict")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    base = TrainerConfig(
        model_name=args.model,
        task_name=args.task,
        packr_config=PackRConfig(mode=args.mode),
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=min(250, args.max_steps // 2),
        output_dir=args.output_dir,
    )

    sweep = {}

    if args.sweep:
        with open(args.sweep) as f:
            sweep = json.load(f)

    for p in args.param:
        if ":" in p:
            path, vals_str = p.split(":", 1)
            vals = [_parse_val(v) for v in vals_str.split(",")]
            sweep[path] = vals

    runner = AblationRunner(
        base_config=base,
        sweep=sweep,
        parallel=args.parallel,
        base_output_dir=args.output_dir,
    )
    runner.run()
    runner.summarize()


if __name__ == "__main__":
    main()
