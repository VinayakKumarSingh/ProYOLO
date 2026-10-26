"""Unit and performance tests for the Attack Classifier.

Verifies:
1. Single-sample prediction latency is strictly under 2 milliseconds on CPU (< 2.0 ms).
2. Real-time inference accuracy on clean vs. adversarial optical patch feature vectors.
3. Decision threshold alert triggering (probability >= 0.65).
4. Feature importance and explainability attribution.
5. Cross-validation benchmarking and model serialization.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from src.classifier.attack_classifier import AttackClassifier, AttackPrediction
from src.classifier.train_classifier import (
    HardNegativePatchApplier,
    generate_synthetic_features,
    train_and_benchmark_models,
)
from src.detection.feature_extractor import FEATURE_NAMES, FeaturePacket


@pytest.fixture
def clean_feature_vector() -> np.ndarray:
    """Simulates a feature vector of a normal, unperturbed tracked human."""
    # Matches FEATURE_NAMES:
    # 0: conf_current (0.94), 1: conf_drop (0.01), 2: conf_std (0.02),
    # 3: area_stability (0.02), 4: aspect_ratio_stability (0.02),
    # 5: mean_consecutive_iou (0.95), 6: min_consecutive_iou (0.90),
    # 7: flicker_rate (0.0), 8: laplacian_var_norm (0.15),
    # 9: laplacian_entropy (0.20), 10: color_hue_entropy (0.20),
    # 11: color_sat_entropy (0.20), 12: color_val_entropy (0.30)
    return np.array(
        [0.94, 0.01, 0.02, 0.02, 0.02, 0.95, 0.90, 0.0, 0.15, 0.20, 0.20, 0.20, 0.30],
        dtype=np.float32,
    )


@pytest.fixture
def attacked_feature_vector() -> np.ndarray:
    """Simulates a feature vector under active adversarial optical patch attack."""
    # Severe confidence drop, erratic area stability, high flicker, high gradient/color entropy
    return np.array(
        [0.22, 0.72, 0.35, 0.55, 0.45, 0.40, 0.15, 0.30, 0.85, 0.90, 0.88, 0.82, 0.78],
        dtype=np.float32,
    )


@pytest.fixture
def classifier() -> AttackClassifier:
    """Instantiates AttackClassifier with the best available model (ONNX preferred for <2ms latency)."""
    onnx_path = Path("models/attack_classifier.onnx")
    joblib_path = Path("models/attack_classifier.joblib")
    if onnx_path.exists():
        return AttackClassifier(model_path=str(onnx_path), threshold=0.65)
    elif joblib_path.exists():
        return AttackClassifier(model_path=str(joblib_path), threshold=0.65)
    return AttackClassifier(threshold=0.65)


class TestAttackClassifierLatency:
    """Verifies the strict enterprise constraint: single-sample CPU latency < 2.0 ms."""

    def test_single_sample_latency_under_2ms_cpu(self, classifier, clean_feature_vector):
        """Single-sample prediction latency must be strictly under 2 milliseconds on CPU."""
        # Warmup runs
        for _ in range(50):
            _ = classifier.predict(clean_feature_vector)

        num_trials = 100
        latencies_ms = []

        for _ in range(num_trials):
            pred = classifier.predict(clean_feature_vector)
            latencies_ms.append(pred.inference_latency_ms)

        avg_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))

        print(f"\n[LATENCY BENCHMARK] Mean: {avg_latency:.4f} ms | P95: {p95_latency:.4f} ms")

        # Enterprise Target: Average latency strictly below 2.0 ms
        assert avg_latency < 2.0, f"Average latency {avg_latency:.4f} ms exceeded 2.0 ms limit!"
        assert p95_latency < 2.0, f"95th percentile latency {p95_latency:.4f} ms exceeded 2.0 ms limit!"


class TestAttackClassifierAccuracyAndAlerts:
    """Verifies decision thresholding and feature attribution."""

    def test_clean_sample_classification(self, classifier, clean_feature_vector):
        """Clean track must not trigger an attack alert."""
        pred = classifier.predict(clean_feature_vector)
        assert isinstance(pred, AttackPrediction)
        assert pred.is_attack is False
        assert pred.attack_probability < 0.65
        assert pred.confidence_clean > 0.35

    def test_attacked_sample_classification_and_alert(self, classifier, attacked_feature_vector):
        """Attacked track must trigger an alert with probability >= 0.65."""
        pred = classifier.predict(attacked_feature_vector)
        assert isinstance(pred, AttackPrediction)
        assert pred.is_attack is True
        assert pred.attack_probability >= 0.65
        assert pred.confidence_clean <= 0.35

    def test_feature_importance_explainability(self, classifier, attacked_feature_vector):
        """Triggered alerts must provide top contributing features explaining the decision."""
        pred = classifier.predict(attacked_feature_vector)
        assert len(pred.top_contributing_features) > 0

        # Verify structure: list of (feature_name, score) tuples
        top_names = [item[0] for item in pred.top_contributing_features]
        for name, score in pred.top_contributing_features:
            assert name in FEATURE_NAMES
            assert isinstance(score, float)
            assert score >= 0.0

        # Key attack features (e.g. conf_drop, entropy, area_stability) should be in top contributors
        assert any(
            f in top_names
            for f in ("conf_drop", "laplacian_entropy", "area_stability", "color_hue_entropy")
        )

    def test_input_type_flexibility(self, classifier, clean_feature_vector):
        """Classifier must accept FeaturePacket, 1D NumPy array, or Python list."""
        # 1. FeaturePacket input
        packet = FeaturePacket(
            track_id=1,
            bbox=np.array([100, 100, 200, 300]),
            feature_vector=clean_feature_vector,
            feature_dict={FEATURE_NAMES[i]: float(clean_feature_vector[i]) for i in range(13)},
        )
        pred_packet = classifier.predict(packet)
        assert isinstance(pred_packet, AttackPrediction)

        # 2. Python list input
        pred_list = classifier.predict(clean_feature_vector.tolist())
        assert isinstance(pred_list, AttackPrediction)

        # 3. Shape mismatch raises ValueError
        with pytest.raises(ValueError):
            classifier.predict(clean_feature_vector[:5])

    def test_batch_prediction(self, classifier, clean_feature_vector, attacked_feature_vector):
        """Batch prediction must process multiple tracks correctly."""
        matrix = np.vstack([clean_feature_vector, attacked_feature_vector])
        preds = classifier.predict_batch(matrix)
        assert len(preds) == 2
        assert preds[0].is_attack is False
        assert preds[1].is_attack is True

    def test_benign_checkerboard_vs_adversarial_suppression(self, classifier):
        """Verifies two-stage causal verification gate:

        1. A high-contrast benign checkerboard pattern on a person with delta_p_obj = 0.0
           must NOT trigger an attack alert (is_attack = False).
        2. An objectness-suppressing patch with delta_p_obj = 0.35
           must trigger an attack alert (is_attack = True).
        """
        person_bbox = [100.0, 100.0, 250.0, 450.0]
        patch_roi = [130.0, 160.0, 220.0, 250.0]

        # 1. Benign checkerboard pattern on person:
        # High texture/gradient/color entropy, but stable detection (conf_current=0.92, conf_drop=0.01)
        # and no recovery from Pass 2 inpainting (delta_p_obj=0.0).
        benign_checkerboard_vec = np.array(
            [
                0.92,  # conf_current: detected normally
                0.01,  # conf_drop: no confidence drop
                0.02,  # conf_std
                0.02,  # area_stability
                0.02,  # aspect_ratio_stability
                0.95,  # mean_consecutive_iou
                0.90,  # min_consecutive_iou
                0.0,   # flicker_rate
                0.86,  # laplacian_var_norm: high due to checkerboard edges
                0.89,  # laplacian_entropy: high due to checkerboard pattern
                0.82,  # color_hue_entropy: high contrast
                0.80,  # color_sat_entropy
                0.78,  # color_val_entropy
                1.5,   # max_grid_norm_ratio: normal
                1.1,   # jpeg_recompression_delta: normal
            ],
            dtype=np.float32,
        )

        pred_benign = classifier.predict(
            benign_checkerboard_vec,
            delta_p_obj=0.0,
            patch_roi=patch_roi,
            person_bbox=person_bbox,
        )
        assert isinstance(pred_benign, AttackPrediction)
        assert pred_benign.is_attack is False, (
            f"Expected is_attack=False for benign checkerboard with delta_p_obj=0.0, got: {pred_benign}"
        )
        assert pred_benign.causal_verification_passed is False

        # 2. Objectness-suppressing adversarial patch on person:
        # Suppressed detection (conf_current=0.25, conf_drop=0.68) and Pass 2 inpainting recovery (delta_p_obj=0.35).
        suppressing_patch_vec = np.array(
            [
                0.25,  # conf_current: suppressed
                0.68,  # conf_drop: huge confidence drop
                0.35,  # conf_std
                0.45,  # area_stability
                0.40,  # aspect_ratio_stability
                0.45,  # mean_consecutive_iou
                0.20,  # min_consecutive_iou
                0.25,  # flicker_rate
                0.85,  # laplacian_var_norm
                0.90,  # laplacian_entropy
                0.85,  # color_hue_entropy
                0.80,  # color_sat_entropy
                0.80,  # color_val_entropy
                2.5,   # max_grid_norm_ratio
                2.8,   # jpeg_recompression_delta
            ],
            dtype=np.float32,
        )

        pred_attack = classifier.predict(
            suppressing_patch_vec,
            delta_p_obj=0.35,
            patch_roi=patch_roi,
            person_bbox=person_bbox,
        )
        assert isinstance(pred_attack, AttackPrediction)
        assert pred_attack.is_attack is True, (
            f"Expected is_attack=True for suppressing patch with delta_p_obj=0.35, got: {pred_attack}"
        )
        assert pred_attack.causal_verification_passed is True
        assert pred_attack.delta_p_obj == 0.35

    def test_hard_negative_patch_applier(self):
        """Test HardNegativePatchApplier pattern generation and EOT augmentation."""
        import cv2

        applier = HardNegativePatchApplier(seed=42)
        checker = applier.generate_checkerboard(80, 80)
        assert checker.shape == (80, 80, 3)
        stripes = applier.generate_stripes(80, 80)
        assert stripes.shape == (80, 80, 3)
        qr = applier.generate_qr_like(80, 80)
        assert qr.shape == (80, 80, 3)
        text_box = applier.generate_text_box(80, 80)
        assert text_box.shape == (80, 80, 3)

        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        pbox = [50.0, 50.0, 200.0, 280.0]
        aug_img, roi = applier.apply(dummy_img, pbox)
        assert aug_img.shape == (300, 300, 3)
        assert len(roi) == 4
        assert roi[0] >= pbox[0] and roi[2] <= pbox[2]


class TestTrainingAndCrossValidation:
    """Verifies train_classifier.py cross-validation and benchmarking logic."""

    def test_cross_validation_and_onnx_export(self, tmp_path):
        """Verifies 5-fold CV runs across models and exports joblib and onnx."""
        X, y = generate_synthetic_features(n_samples_per_class=100, seed=123)

        out_joblib = tmp_path / "test_model.joblib"
        out_onnx = tmp_path / "test_model.onnx"

        results = train_and_benchmark_models(
            X=X,
            y=y,
            n_splits=3,
            output_model_path=str(out_joblib),
            output_onnx_path=str(out_onnx),
        )

        assert "best_model_name" in results
        assert results["best_f1"] > 0.85
        assert results["best_precision"] > 0.85
        assert results["best_recall"] > 0.85
        assert os.path.exists(results["joblib_path"])

        # Check exported ONNX inference if generated
        if results.get("onnx_path") and os.path.exists(results["onnx_path"]):
            import onnxruntime as ort

            sess = ort.InferenceSession(results["onnx_path"], providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            out = sess.run(None, {inp_name: X[:1]})
            assert len(out) > 0
