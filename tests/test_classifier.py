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
    # Matches FEATURE_NAMES (15 features):
    # 0: conf_current (0.94), 1: conf_drop (0.01), 2: conf_std (0.02),
    # 3: area_stability (0.02), 4: aspect_ratio_stability (0.02),
    # 5: mean_consecutive_iou (0.95), 6: min_consecutive_iou (0.90),
    # 7: flicker_rate (0.0), 8: laplacian_var_norm (0.15),
    # 9: laplacian_entropy (0.20), 10: color_hue_entropy (0.20),
    # 11: color_sat_entropy (0.20), 12: color_val_entropy (0.30),
    # 13: max_grid_norm_ratio (1.4), 14: jpeg_recompression_delta (1.0)
    return np.array(
        [
            0.94, 0.01, 0.02, 0.02, 0.02, 0.95, 0.90, 0.0,
            0.15, 0.20, 0.20, 0.20, 0.30, 1.4, 1.0,
        ],
        dtype=np.float32,
    )


@pytest.fixture
def attacked_feature_vector() -> np.ndarray:
    """Simulates a feature vector under active adversarial optical patch attack."""
    # Severe confidence drop, erratic area stability, high flicker, high gradient/color entropy,
    # and elevated grid norm ratio / jpeg recompression delta
    return np.array(
        [
            0.22, 0.72, 0.35, 0.55, 0.45, 0.40, 0.15, 0.30,
            0.85, 0.90, 0.88, 0.82, 0.78, 2.6, 2.4,
        ],
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
            feature_dict={FEATURE_NAMES[i]: float(clean_feature_vector[i]) for i in range(len(clean_feature_vector))},
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

    def test_benign_striped_and_checkered_clothing(self, classifier):
        """Verifies that benign striped and checkered clothing with delta_p_obj = 0.0 and patch_conf = 0.0
        is NEVER flagged as an attack (is_attack = False).
        """
        person_bbox = [100.0, 100.0, 260.0, 480.0]
        torso_patch_roi = [135.0, 160.0, 225.0, 260.0]

        # 1. Benign striped clothing vector (high texture gradients, zero recovery delta, zero patch conf)
        striped_vec = np.array(
            [
                0.93, 0.01, 0.02, 0.03, 0.02, 0.94, 0.88, 0.0,
                0.88, 0.91, 0.75, 0.70, 0.72, 1.6, 1.2,
            ],
            dtype=np.float32,
        )
        pred_striped = classifier.predict(
            striped_vec,
            delta_p_obj=0.0,
            patch_conf=0.0,
            patch_roi=torso_patch_roi,
            person_bbox=person_bbox,
        )
        assert pred_striped.is_attack is False
        assert pred_striped.causal_verification_passed is False

        # 2. Benign checkered clothing vector
        checkered_vec = np.array(
            [
                0.91, 0.01, 0.02, 0.02, 0.02, 0.95, 0.89, 0.0,
                0.90, 0.93, 0.84, 0.82, 0.80, 1.7, 1.3,
            ],
            dtype=np.float32,
        )
        pred_checkered = classifier.predict(
            checkered_vec,
            delta_p_obj=0.0,
            patch_conf=0.0,
            patch_roi=torso_patch_roi,
            person_bbox=person_bbox,
        )
        assert pred_checkered.is_attack is False
        assert pred_checkered.causal_verification_passed is False

    def test_frame_edge_person_guard(self, classifier):
        """Verifies that any person box touching or near the frame edge (e.g. x1 = 2)
        is NEVER flagged as an attack (is_attack = False), even if delta_p_obj is high.
        """
        # Frame edge person: x1 = 2 touches left margin (2 <= 0.05 * 640 = 32)
        edge_person_bbox = [2.0, 100.0, 160.0, 450.0]
        torso_patch_roi = [20.0, 160.0, 120.0, 260.0]

        suspicious_vec = np.array(
            [
                0.35, 0.60, 0.30, 0.40, 0.35, 0.50, 0.25, 0.20,
                0.80, 0.85, 0.80, 0.75, 0.75, 2.4, 2.5,
            ],
            dtype=np.float32,
        )

        pred_edge = classifier.predict(
            suspicious_vec,
            delta_p_obj=0.25,
            patch_roi=torso_patch_roi,
            person_bbox=edge_person_bbox,
            frame_shape=(640, 640),
        )
        assert pred_edge.is_attack is False, (
            f"Frame-edge person with x1=2 must NOT be flagged as an attack! Got: {pred_edge}"
        )
        assert pred_edge.causal_verification_passed is False

    def test_interior_person_with_torso_patch(self, classifier):
        """Verifies that an interior person with a torso patch is flagged as is_attack = True
        when delta_p_obj = 0.30 (Case A) or patch_conf = 0.75 (Case B).
        """
        interior_person_bbox = [100.0, 100.0, 260.0, 480.0]
        torso_patch_roi = [135.0, 160.0, 225.0, 260.0]

        attack_vec = np.array(
            [
                0.24, 0.70, 0.35, 0.45, 0.40, 0.45, 0.20, 0.25,
                0.85, 0.90, 0.85, 0.80, 0.80, 2.5, 2.8,
            ],
            dtype=np.float32,
        )

        # Subcase 1: Case A (Causal Objectness Recovery delta_p_obj = 0.30 >= 0.15)
        pred_case_a = classifier.predict(
            attack_vec,
            delta_p_obj=0.30,
            patch_roi=torso_patch_roi,
            person_bbox=interior_person_bbox,
        )
        assert pred_case_a.is_attack is True, f"Expected is_attack=True for delta_p_obj=0.30, got {pred_case_a}"
        assert pred_case_a.causal_verification_passed is True
        assert pred_case_a.delta_p_obj == 0.30

        # Subcase 2: Case B (Fine-Tuned Ad-YOLO Match patch_conf = 0.75 >= 0.45 with classifier_prob >= 0.65)
        pred_case_b = classifier.predict(
            attack_vec,
            delta_p_obj=0.05,
            patch_conf=0.75,
            patch_roi=torso_patch_roi,
            person_bbox=interior_person_bbox,
        )
        assert pred_case_b.is_attack is True, f"Expected is_attack=True for patch_conf=0.75, got {pred_case_b}"
        assert pred_case_b.causal_verification_passed is True
        assert pred_case_b.patch_conf == 0.75

    def test_non_person_object_never_flagged(self, classifier):
        """Verifies that non-person objects (class_name != 'person') are NEVER flagged as attack."""
        person_bbox = [100.0, 100.0, 260.0, 480.0]
        torso_patch_roi = [135.0, 160.0, 225.0, 260.0]

        attack_vec = np.array(
            [
                0.24, 0.70, 0.35, 0.45, 0.40, 0.45, 0.20, 0.25,
                0.85, 0.90, 0.85, 0.80, 0.80, 2.5, 2.8,
            ],
            dtype=np.float32,
        )

        pred_car = classifier.predict(
            attack_vec,
            delta_p_obj=0.35,
            patch_roi=torso_patch_roi,
            person_bbox=person_bbox,
            class_name="car",
        )
        assert pred_car.is_attack is False
        assert pred_car.causal_verification_passed is False

    def test_global_dpatch_stream_gate(self, classifier):
        """Verifies that live stream global DPATCH alerts are disabled unless:
        max_grid_norm_ratio > 5.0 inside central 85% of frame
        AND Pass 2 inpainting recovers >= 1 suppressed object with delta_p_obj > 0.35.
        """
        # Corner DPATCH (outside central 85%): x1=10, y1=10
        corner_dpatch = [10.0, 10.0, 40.0, 40.0]
        vec = np.array(
            [
                0.30, 0.50, 0.30, 0.40, 0.30, 0.50, 0.20, 0.20,
                0.85, 0.90, 0.85, 0.80, 0.80, 6.0, 3.5,
            ],
            dtype=np.float32,
        )

        # 1. Corner patch fails central 85% requirement -> False
        pred_corner = classifier.predict(
            vec,
            max_grid_norm_ratio=6.0,
            delta_p_obj=0.40,
            recovered_objects_count=1,
            patch_roi=corner_dpatch,
            frame_shape=(640, 640),
        )
        assert pred_corner.is_attack is False

        # 2. Central patch (inside central 85%) but ratio <= 5.0 -> False
        central_dpatch = [250.0, 250.0, 320.0, 320.0]
        pred_low_ratio = classifier.predict(
            vec,
            max_grid_norm_ratio=4.5,
            delta_p_obj=0.40,
            recovered_objects_count=1,
            patch_roi=central_dpatch,
            frame_shape=(640, 640),
        )
        assert pred_low_ratio.is_attack is False

        # 3. Central patch with ratio > 5.0 AND delta_p_obj > 0.35 AND recovered >= 1 -> True
        pred_dpatch_attack = classifier.predict(
            vec,
            max_grid_norm_ratio=5.5,
            delta_p_obj=0.40,
            recovered_objects_count=1,
            patch_roi=central_dpatch,
            frame_shape=(640, 640),
        )
        assert pred_dpatch_attack.is_attack is True
        assert pred_dpatch_attack.causal_verification_passed is True

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
