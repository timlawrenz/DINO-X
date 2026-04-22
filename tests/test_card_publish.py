"""Tests for zoo/card.py and zoo/publish.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from zoo.card import generate_model_card, _scrub_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_config():
    return {
        "name": "vit-small",
        "dim": 384,
        "depth": 12,
        "heads": 6,
        "patch": 14,
        "mlp_ratio": 4.0,
        "img_size": 224,
        "scale_aware": True,
        "params_millions": 21.7,
    }


@pytest.fixture
def training_config():
    return {
        "model": {"name": "vit-small", "dim": 384, "depth": 12, "heads": 6},
        "batch_size": 8,
        "accumulation_steps": 4,
        "lr": 0.0002,
        "min_lr": 1e-6,
        "warmup_steps": 500,
        "max_steps": 5000,
        "ema": 0.996,
        "center_momentum": 0.9,
        "weight_decay": 0.04,
        "train_seed": 42,
        "gram_weight": 1.0,
        "koleo_weight": 0.1,
        "git_commit": "abc123",
        "data_manifest_hash": "deadbeef",
        "scale_aware": True,
        "img_size": 224,
        "index_csv": "/home/user/data/index.csv",
        "run_dir": "/tmp/runs/test",
    }


@pytest.fixture
def eval_results():
    return {
        "kind": "panorgan_evaluation",
        "step": 5000,
        "seed": 42,
        "val_slices": 3843,
        "checkpoint": "/home/user/runs/checkpoint.pth",
        "metrics": {
            "view_retrieval_per_dataset": {
                "lidc-idri": {
                    "n": 512, "top1": 0.027, "top5": 0.094,
                    "random_baseline": 0.002, "ratio_vs_random": 14.0,
                },
                "pancreas-ct": {
                    "n": 512, "top1": 0.010, "top5": 0.074,
                    "random_baseline": 0.002, "ratio_vs_random": 5.0,
                },
            },
            "dataset_discrimination_probe": {"accuracy": 1.0, "auc": 1.0},
            "spacing_counterfactual": {
                "cosine_distance_real_vs_2x": {"mean": 0.055},
                "cosine_distance_real_vs_half": {"mean": 0.107},
            },
            "spacing_prediction": {"r2": 0.876},
            "domain_clustering": {},
            "embedding_stats": {},
        },
    }


@pytest.fixture
def lineage():
    return {
        "datasets": [
            {
                "name": "LIDC-IDRI", "organ": "Lung", "slices": "24,159",
                "spacing_range": "0.46-0.98mm", "thickness_range": "0.625-5.0mm",
                "license": "CC-BY-NC-3.0",
            },
            {
                "name": "Pancreas-CT", "organ": "Abdomen", "slices": "17,764",
                "spacing_range": "0.6-0.98mm", "thickness_range": "1.0-3.0mm",
                "license": "CC-BY-3.0",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Card generation tests
# ---------------------------------------------------------------------------

class TestModelCard:
    def test_generates_valid_markdown(self, model_config):
        card = generate_model_card(model_config)
        assert card.startswith("---\n")
        assert "---" in card
        assert "# " in card

    def test_yaml_frontmatter_tags(self, model_config):
        card = generate_model_card(model_config)
        assert "library_name: dinox" in card
        assert "scale-aware" in card
        assert "medical-imaging" in card
        assert "pipeline_tag: feature-extraction" in card

    def test_architecture_section(self, model_config):
        card = generate_model_card(model_config)
        assert "## Architecture" in card
        assert "384" in card
        assert "12" in card
        assert "21.7M" in card
        assert "✅" in card  # scale_aware

    def test_no_scale_aware(self, model_config):
        model_config["scale_aware"] = False
        card = generate_model_card(model_config)
        assert "❌" in card
        assert "scale-aware" not in card.split("---")[1]  # not in tags

    def test_training_section(self, model_config, training_config):
        card = generate_model_card(model_config, training_config=training_config)
        assert "## Training" in card
        assert "DINOv3" in card
        assert "0.0002" in card
        assert "5000" in card
        assert "abc123" in card

    def test_dataset_section(self, model_config, lineage):
        card = generate_model_card(model_config, lineage=lineage)
        assert "## Training Data" in card
        assert "LIDC-IDRI" in card
        assert "Pancreas-CT" in card
        assert "CC-BY-NC-3.0" in card

    def test_eval_section(self, model_config, eval_results):
        card = generate_model_card(model_config, eval_results=eval_results)
        assert "## Evaluation" in card
        assert "14×" in card
        assert "5×" in card
        assert "1.000" in card  # AUC
        assert "0.876" in card  # R²
        assert "3843" in card  # val slices

    def test_usage_section_scale_aware(self, model_config):
        card = generate_model_card(model_config, model_name="test-model")
        assert "load_model" in card
        assert "encode" in card
        assert "spacing" in card
        assert "apply_lora" in card

    def test_disclaimer(self, model_config):
        card = generate_model_card(model_config)
        assert "Research use only" in card
        assert "not approved for clinical" in card

    def test_license_section(self, model_config):
        card = generate_model_card(model_config)
        assert "CC-BY-NC-3.0" in card

    def test_citation(self, model_config):
        card = generate_model_card(model_config)
        assert "@software{dinox2026" in card

    def test_full_card(self, model_config, training_config, eval_results, lineage):
        card = generate_model_card(
            model_config,
            training_config=training_config,
            eval_results=eval_results,
            lineage=lineage,
            model_name="dinox-ct-vit-small-v1",
        )
        # All major sections present
        assert "## Architecture" in card
        assert "## Training" in card
        assert "## Training Data" in card
        assert "## Evaluation" in card
        assert "## Usage" in card
        assert "## Citation" in card
        assert "## License" in card
        # Length reasonable
        assert 2000 < len(card) < 10000


class TestScrubPath:
    def test_scrubs_home_path(self):
        assert "home" not in _scrub_path("/home/user/data/processed/index.csv")

    def test_scrubs_tmp_path(self):
        assert "/tmp/" not in _scrub_path("/tmp/runs/test/config.json")

    def test_keeps_relative_path(self):
        result = _scrub_path("data/processed/index.csv")
        assert result == "data/processed/index.csv"

    def test_finds_data_anchor(self):
        result = _scrub_path("/home/user/project/data/processed/index.csv")
        assert result == "data/processed/index.csv"


# ---------------------------------------------------------------------------
# Publish tests
# ---------------------------------------------------------------------------

class TestPublish:
    def test_dry_run_creates_staging(self):
        """Dry run with a real checkpoint creates proper staging dir."""
        from zoo.arch import DinoStudentTeacher, PatchViT

        # Create a minimal checkpoint
        backbone = PatchViT(img_size=224, patch=14, dim=64, depth=2, heads=2)
        wrapper = DinoStudentTeacher(backbone, out_dim=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            torch.save({
                "step": 100,
                "student": wrapper.state_dict(),
                "config": {
                    "model": {"name": "test", "dim": 64, "depth": 2, "heads": 2, "patch": 14, "mlp_ratio": 4.0, "out_dim": 256},
                    "img_size": 224,
                    "scale_aware": False,
                    "batch_size": 8,
                    "accumulation_steps": 4,
                    "lr": 2e-4,
                    "min_lr": 1e-6,
                    "warmup_steps": 100,
                    "max_steps": 1000,
                    "ema": 0.996,
                    "center_momentum": 0.9,
                    "weight_decay": 0.04,
                    "train_seed": 42,
                    "gram_weight": 1.0,
                    "koleo_weight": 0.1,
                    "git_commit": "test123",
                    "data_manifest_hash": "abcd1234",
                },
            }, ckpt_path)

            # Create eval results
            eval_path = Path(tmpdir) / "eval.json"
            eval_path.write_text(json.dumps({
                "kind": "panorgan_evaluation",
                "step": 100,
                "seed": 42,
                "val_slices": 100,
                "checkpoint": "/home/test/checkpoint.pth",
                "metrics": {
                    "view_retrieval_per_dataset": {},
                    "dataset_discrimination_probe": {"accuracy": 0.5, "auc": 0.5},
                    "spacing_counterfactual": {
                        "cosine_distance_real_vs_2x": {"mean": 0.0},
                        "cosine_distance_real_vs_half": {"mean": 0.0},
                    },
                    "spacing_prediction": {"r2": 0.0},
                    "domain_clustering": {},
                    "embedding_stats": {},
                },
            }))

            from zoo.publish import publish_to_hub
            staging = publish_to_hub(
                training_checkpoint=ckpt_path,
                repo_id="test/test-model",
                eval_results_path=eval_path,
                dry_run=True,
            )

            staging_path = Path(staging)
            assert (staging_path / "README.md").exists()
            assert (staging_path / "config.json").exists()
            assert (staging_path / "training_config.json").exists()
            assert (staging_path / "eval_results.json").exists()
            # Either .pth or .safetensors should exist
            assert (staging_path / "backbone.pth").exists()

            # Verify README content
            readme = (staging_path / "README.md").read_text()
            assert "Research use only" in readme

            # Verify no local paths in training config
            with open(staging_path / "training_config.json") as f:
                tc = json.load(f)
            tc_str = json.dumps(tc)
            assert "/home/" not in tc_str

            # Verify scrubbed eval
            with open(staging_path / "eval_results.json") as f:
                ev = json.load(f)
            assert "/home/" not in ev.get("checkpoint", "")

    def test_missing_checkpoint_raises(self):
        from zoo.publish import publish_to_hub
        with pytest.raises(FileNotFoundError):
            publish_to_hub(
                training_checkpoint="/nonexistent/checkpoint.pth",
                repo_id="test/test",
                dry_run=True,
            )
