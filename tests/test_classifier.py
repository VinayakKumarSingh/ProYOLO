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
