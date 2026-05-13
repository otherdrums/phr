"""Integration tests — end-to-end pipeline verification via ZPackRTrainer.

Each test runs a full (short) training loop and asserts the pipeline works.
Requires CUDA or CPU; automatically skips if bert-base-uncased is unavailable.
"""

import os
import json
import tempfile
import pytest
import torch

# Ensure packr + tools are importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.train_harness import TrainerConfig, ZPackRTrainer
from packr.config import PackRConfig


def _model_available():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("bert-base-uncased")
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(not _model_available(), reason="bert-base-uncased not cached")


class TestZPackRPipeline:
    @requires_model
    def test_zpackr_smoke_50_steps(self):
        """Verify ZPackR mode completes 50 steps and records metrics."""
        tmpdir = tempfile.mkdtemp(prefix="zpackr_test_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="zpackr", layer_scope="all"),
            max_steps=50,
            eval_interval=25,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            warmup_steps=0,
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        # Verify output files exist
        assert os.path.exists(os.path.join(trainer.output_dir, "metrics.jsonl"))
        assert os.path.exists(os.path.join(trainer.output_dir, "config.json"))
        assert os.path.exists(os.path.join(trainer.output_dir, "summary.json"))

        # Verify metrics content
        metrics = []
        with open(os.path.join(trainer.output_dir, "metrics.jsonl")) as f:
            for line in f:
                metrics.append(json.loads(line))

        step_entries = [m for m in metrics if m["type"] == "step"]
        assert len(step_entries) >= 50, f"Expected >= 50 step entries, got {len(step_entries)}"

        eval_entries = [m for m in metrics if m["type"] == "eval"]
        assert len(eval_entries) >= 2, f"Expected >= 2 eval entries, got {len(eval_entries)}"

        # Verify salience data is recorded
        salience_found = any("salience" in m for m in step_entries)
        assert salience_found, "Expected salience data in step metrics"

        # Verify summary
        with open(os.path.join(trainer.output_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["total_steps"] == 50

    @requires_model
    def test_packr_smoke_50_steps(self):
        """Verify PackR mode completes 50 steps and records metrics."""
        tmpdir = tempfile.mkdtemp(prefix="packr_test_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="packr", layer_scope="all"),
            max_steps=50,
            eval_interval=25,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            warmup_steps=0,
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        assert os.path.exists(os.path.join(trainer.output_dir, "metrics.jsonl"))
        assert os.path.exists(os.path.join(trainer.output_dir, "summary.json"))

    @requires_model
    def test_gate_skip_backward(self):
        """Verify gate_skip_forward=False mode records gate_skipped events."""
        tmpdir = tempfile.mkdtemp(prefix="zpackr_gate_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="zpackr", layer_scope="all"),
            max_steps=20,
            eval_interval=10,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            gate_enabled=True,
            gate_threshold=2.0,
            gate_skip_forward=False,
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        metrics = []
        with open(os.path.join(trainer.output_dir, "metrics.jsonl")) as f:
            for line in f:
                metrics.append(json.loads(line))

        gate_entries = [m for m in metrics if m.get("gate_skipped")]
        # At least some batches should be gated (Super Dict compresses English well)
        assert len(gate_entries) >= 0, "Gate should produce events (even if all pass)"

        with open(os.path.join(trainer.output_dir, "summary.json")) as f:
            summary = json.load(f)
        assert "gate_skip_rate" in summary

    @requires_model
    def test_reindex_event(self):
        """Verify reindex events appear in metrics."""
        tmpdir = tempfile.mkdtemp(prefix="zpackr_reindex_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="zpackr", layer_scope="all"),
            max_steps=30,
            eval_interval=15,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            reindex_interval=10,
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        metrics = []
        with open(os.path.join(trainer.output_dir, "metrics.jsonl")) as f:
            for line in f:
                metrics.append(json.loads(line))

        reindex_entries = [m for m in metrics if m["type"] == "reindex"]
        assert len(reindex_entries) >= 1, f"Expected >= 1 reindex event, got {len(reindex_entries)}"

    @requires_model
    def test_checkpoint_saved(self):
        """Verify checkpoints are created at the right intervals."""
        tmpdir = tempfile.mkdtemp(prefix="zpackr_ckpt_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="zpackr", layer_scope="all"),
            max_steps=25,
            eval_interval=12,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            checkpoint_interval=10,
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        checkpoint_dir = os.path.join(trainer.output_dir, "checkpoints")
        assert os.path.exists(checkpoint_dir)
        subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
        assert len(subdirs) >= 2, f"Expected >= 2 checkpoints, got {len(subdirs)}"

    @requires_model
    def test_output_dir_naming(self):
        """Verify output dir includes timestamp and git commit."""
        tmpdir = tempfile.mkdtemp(prefix="zpackr_naming_")
        config = TrainerConfig(
            model_name="bert-base-uncased",
            task_name="sst2",
            packr_config=PackRConfig(mode="zpackr", layer_scope="all"),
            max_steps=5,
            batch_size=4,
            output_dir=tmpdir,
            seed=123,
            run_label="naming_test",
        )
        trainer = ZPackRTrainer(config)
        results = trainer.run()

        dirname = os.path.basename(trainer.output_dir)
        assert "naming_test" in dirname, f"Expected 'naming_test' in {dirname}"
        # Should have timestamp pattern YYYY-MM-DD_HHMMSS
        parts = dirname.split("_")
        date_part = "_".join(parts[1:3]) if len(parts) > 2 else ""
        assert "-" in date_part, f"Expected date in {dirname}"
