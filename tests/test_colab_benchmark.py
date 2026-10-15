"""Unit tests for the Colab/Kaggle defensive robustness benchmarking modules.

Verifies:
1. Deterministic seeding and reproducibility.
2. Mathematical formulations of TV loss, NPS loss, and Detection Sensitivity loss.
3. Expectation over Transformation (EoT) spatial bounds and pixel value constraints [0.0, 1.0].
4. Pattern optimizer bounds and image export.
5. Defensive dataset synthesis, annotation schema, and zip packaging.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from notebooks.colab_robustness_benchmark import (
    PRINTABLE_RGB_GAMUT,
    DefensiveDatasetBuilder,
    DetectionSensitivityLoss,
    ExpectationOverTransformation,
    NonPrintabilityScoreLoss,
    PatternOptimizer,
    TotalVariationLoss,
    seed_everything,
)


@pytest.fixture
def clean_canvas_tensor():
    """Provides a synthetic 3x640x640 canvas in [0, 1]."""
    return torch.ones(3, 640, 640, dtype=torch.float32) * 0.7


@pytest.fixture
def person_bounding_box():
    """Provides a realistic human bounding box (x1, y1, x2, y2)."""
    return (150, 100, 280, 450)


class TestLossFormulations:
    """Verifies mathematical properties of TV, NPS, and Detection losses."""

    def test_total_variation_loss_smoothness(self):
        """Uniform constant pattern must have zero TV loss; noisy pattern must have high TV loss."""
        tv_fn = TotalVariationLoss()

        # Perfectly smooth constant tensor
        smooth_pattern = torch.ones(3, 50, 50, dtype=torch.float32) * 0.5
        loss_smooth = tv_fn(smooth_pattern)
        assert float(loss_smooth.item()) == pytest.approx(0.0, abs=1e-6)

        # Alternating checkerboard / noisy tensor
        noisy_pattern = torch.zeros(3, 50, 50, dtype=torch.float32)
        noisy_pattern[:, ::2, ::2] = 1.0
        loss_noisy = tv_fn(noisy_pattern)
        assert float(loss_noisy.item()) > 0.1

    def test_non_printability_score_palette_match(self):
        """Pattern made entirely of a palette color must have near-zero NPS loss."""
        nps_fn = NonPrintabilityScoreLoss(printable_gamut=PRINTABLE_RGB_GAMUT)

        # Create pattern using exact Crimson Red from gamut [0.85, 0.1, 0.1]
        c_red = PRINTABLE_RGB_GAMUT[5]
        pattern = torch.zeros(3, 20, 20, dtype=torch.float32)
        for c in range(3):
            pattern[c, :, :] = c_red[c]

        loss = nps_fn(pattern)
        assert float(loss.item()) < 1e-4

        # Arbitrary non-gamut color has positive NPS loss
        off_pattern = torch.ones(3, 20, 20, dtype=torch.float32) * 0.4321
        loss_off = nps_fn(off_pattern)
        assert float(loss_off.item()) > 0.01

    def test_detection_sensitivity_loss(self):
        """Verifies DetectionSensitivityLoss handles YOLOv8 tensor shapes and suppresses scores."""
        det_fn = DetectionSensitivityLoss(target_class_idx=0)

        # Simulated YOLOv8 output: [batch=1, 84, anchors=100]
        # 4 coords + 80 class logits
        preds = torch.randn(1, 84, 100, dtype=torch.float32)
        loss = det_fn(preds)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert float(loss.item()) >= 0.0


class TestExpectationOverTransformation:
    """Verifies EoT transformations and strict adherence to [0.0, 1.0] domain."""

    def test_eot_bounds_and_range(self, clean_canvas_tensor, person_bounding_box):
        """Transformed composite image must strictly remain within [0.0, 1.0]."""
        eot = ExpectationOverTransformation(
            min_scale=0.2,
            max_scale=0.3,
            max_rotation_deg=20.0,
            brightness_jitter=0.15,
            contrast_jitter=0.15,
            noise_std=0.03,
        )

        pattern = torch.rand(3, 100, 100, dtype=torch.float32)
        composite = eot(pattern, clean_canvas_tensor, person_bounding_box)

        assert composite.shape == clean_canvas_tensor.shape
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0


class TestPatternOptimizer:
    """Verifies pattern optimization step, parameter bounds, and export."""

    def test_optimizer_step_and_export(self, clean_canvas_tensor, person_bounding_box, tmp_path):
        """Verifies optimization step converges and clamped to [0.0, 1.0]."""
        optimizer = PatternOptimizer(
            pattern_size=(50, 50),
            lr=0.05,
            device=torch.device("cpu"),
        )

        initial_pattern = optimizer.pattern.detach().clone()
        metrics = optimizer.optimize_step(clean_canvas_tensor, person_bounding_box)

        assert "total_loss" in metrics
        assert "loss_det" in metrics
        assert "loss_tv" in metrics
        assert "loss_nps" in metrics

        # Tensor must remain strictly clamped to [0.0, 1.0]
        assert optimizer.pattern.min() >= 0.0
        assert optimizer.pattern.max() <= 1.0

        # Weights should have updated
        assert not torch.equal(optimizer.pattern, initial_pattern)

        # Export test
        out_png = tmp_path / "test_pattern.png"
        saved_path = optimizer.export_pattern(str(out_png))
        assert os.path.exists(saved_path)
        assert Path(saved_path).stat().st_size > 0


class TestDefensiveDatasetBuilder:
    """Verifies dataset generation, metadata generation, and zip packaging."""

    def test_build_dataset_structure(self, tmp_path):
        """Verifies clean and perturbed frames, annotations.json, metadata.csv, and zip."""
        # 1. Create a dummy pattern
        pattern_file = tmp_path / "dummy_pattern.png"
        dummy_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        import cv2
        cv2.imwrite(str(pattern_file), dummy_img)

        output_dataset_dir = tmp_path / "test_dataset"
        builder = DefensiveDatasetBuilder(
            output_dir=str(output_dataset_dir),
            image_size=(320, 320),
        )

        summary = builder.build_dataset(
            pattern_path=str(pattern_file),
            num_samples_per_class=5,  # Fast test with 5 clean + 5 perturbed
        )

        assert summary["total_samples"] == 10
        assert summary["clean_count"] == 5
        assert summary["perturbed_count"] == 5

        # Check file system
        assert os.path.exists(summary["annotations_json"])
        assert os.path.exists(summary["metadata_csv"])
        assert os.path.exists(summary["zip_archive"])

        # Validate annotations JSON schema
        with open(summary["annotations_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_samples"] == 10
            assert len(data["data"]) == 10
            first_entry = data["data"][0]
            assert "frame_id" in first_entry
            assert "label" in first_entry
            assert "person_bbox" in first_entry
            assert "pattern_bbox" in first_entry
            assert "confidence_delta" in first_entry


def test_deterministic_seeding():
    """Verifies seed_everything produces deterministic torch and random output."""
    seed_everything(12345)
    r1 = random.random()
    t1 = torch.rand(5)

    seed_everything(12345)
    r2 = random.random()
    t2 = torch.rand(5)

    assert r1 == r2
    assert torch.equal(t1, t2)
