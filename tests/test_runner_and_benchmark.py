"""Unit and integration tests for ProYOLO unified runner and physical benchmark suite.

Verifies:
1. src/main.py: CLI argument parsing, pipeline initialization, single-frame processing,
   alert dispatching, and graceful lifecycle shutdown.
2. src/evaluation/benchmark.py: Physical CCTV transforms (distance scaling, angle perspective,
   ambient/low-light/back-lit illumination), condition evaluation, and CSV/Markdown reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.detection.detector import Detection
from src.evaluation.benchmark import (
    ANGLES,
    DISTANCES,
    ILLUMINATIONS,
    BenchmarkReportData,
    ConditionResult,
    RobustnessBenchmark,
    apply_angle_transform,
    apply_cctv_conditions,
    apply_distance_transform,
    apply_illumination_transform,
    generate_synthetic_sample,
)
from src.main import (
    AlertDispatcher,
    ProYOLOPipeline,
    build_arg_parser,
    create_detector,
)


# ---------------------------------------------------------------------------
# Tests: src/main.py
# ---------------------------------------------------------------------------

class TestMainRunner:
    """Test suite for the unified pipeline runner in src/main.py."""

    def test_arg_parser_defaults(self):
        """Verify default CLI arguments."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.source == "0"
        assert args.model == "yolov8n.onnx"
        assert args.classifier == "models/attack_classifier.joblib"
        assert args.headless is False
        assert args.camera_id == "cam-01"
        assert args.threshold == 0.65
        assert args.target_fps == 15.0

    def test_arg_parser_custom_values(self):
        """Verify custom CLI argument parsing."""
        parser = build_arg_parser()
        cmd = [
            "--source", "rtsp://192.168.1.100:554/live",
            "--model", "models/custom.onnx",
            "--classifier", "models/custom_clf.joblib",
            "--headless",
            "--camera-id", "gate-camera",
            "--threshold", "0.75",
            "--target-fps", "25.0",
            "--max-frames", "50",
        ]
        args = parser.parse_args(cmd)
        assert args.source == "rtsp://192.168.1.100:554/live"
        assert args.model == "models/custom.onnx"
        assert args.classifier == "models/custom_clf.joblib"
        assert args.headless is True
        assert args.camera_id == "gate-camera"
        assert args.threshold == 0.75
        assert args.target_fps == 25.0
        assert args.max_frames == 50

    def test_alert_dispatcher_mock(self):
        """Verify AlertDispatcher formatting and payload delivery."""
        dispatcher = AlertDispatcher(server_url="http://testserver:8000")
        with patch.object(dispatcher.client, "post") as mock_post:
            mock_post.return_value.status_code = 201

            success = dispatcher.dispatch_alert(
                camera_id="cam-01",
                track_id=12,
                confidence=0.88,
                attack_probability=0.94,
                bbox=[10.0, 20.0, 100.0, 200.0],
                timestamp=1234567.89,
                thumbnail_b64="data:image/jpeg;base64,abc",
                top_features={"hsv_val_entropy": 0.42},
            )
            assert success is True
            mock_post.assert_called_once()
            called_url = mock_post.call_args[0][0]
            called_payload = mock_post.call_args[1]["json"]
            assert "/api/v1/alerts" in called_url
            assert called_payload["track_id"] == 12
            assert called_payload["attack_probability"] == 0.94

    def test_pipeline_process_synthetic_frame(self):
        """Verify single frame processing in headless mode."""
        pipeline = ProYOLOPipeline(
            source=0,
            model_path="yolov8n.onnx",
            classifier_path="models/attack_classifier.joblib",
            camera_id="test-cam",
            headless=True,
            server_url=None,  # No external HTTP calls during unit test
        )

        synthetic_frame = np.full((480, 640, 3), 70, dtype=np.uint8)
        # Mock detection to ensure consistent deterministic test
        mock_detection = Detection(
            bbox=np.array([150.0, 100.0, 300.0, 400.0], dtype=np.float32),
            confidence=0.92,
            class_id=0,
            class_name="person",
        )
        with patch.object(pipeline.detector, "detect", return_value=[mock_detection]):
            display_frame, telemetry = pipeline.process_frame(synthetic_frame, timestamp=1.0)
            assert display_frame.shape == (480, 640, 3)
            assert len(telemetry) == 1
            assert telemetry[0]["track_id"] > 0
            assert telemetry[0]["confidence"] == 0.92
            assert pipeline.total_frames_processed == 1

        pipeline.stop()


# ---------------------------------------------------------------------------
# Tests: src/evaluation/benchmark.py
# ---------------------------------------------------------------------------

class TestCCTVTransforms:
    """Test suite for physical CCTV condition transforms."""

    def test_distance_transform_preserves_shape(self):
        """Verify distance transforms scale resolution while keeping frame size."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for dist in [2, 5, 8]:
            out = apply_distance_transform(img, dist)
            assert out.shape == (480, 640, 3)
            assert out.dtype == np.uint8

    def test_angle_transform_preserves_shape(self):
        """Verify viewing angle projective transforms retain image dimensions."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for angle in [0, 30, 60]:
            out = apply_angle_transform(img, angle)
            assert out.shape == (480, 640, 3)
            assert out.dtype == np.uint8

    def test_illumination_transform(self):
        """Verify ambient, low-light, and back-lit transforms."""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        ambient = apply_illumination_transform(img, "ambient")
        assert np.array_equal(ambient, img)

        low_light = apply_illumination_transform(img, "low_light")
        assert low_light.shape == (480, 640, 3)
        assert np.mean(low_light) < np.mean(img)  # Must be darker

        back_lit = apply_illumination_transform(img, "back_lit")
        assert back_lit.shape == (480, 640, 3)

    def test_composite_conditions(self):
        """Verify composite application across all physical combinations."""
        img = np.full((240, 320, 3), 100, dtype=np.uint8)
        out = apply_cctv_conditions(img, distance_m=5, angle_deg=30, illumination="low_light")
        assert out.shape == (240, 320, 3)
        assert out.dtype == np.uint8


class TestRobustnessBenchmark:
    """Test suite for physical CCTV evaluation and report generation."""

    def test_synthetic_sample_generator(self):
        """Verify synthetic sample fallback generation."""
        sample_clean = generate_synthetic_sample(is_attack=False)
        assert sample_clean["is_attack"] is False
        assert sample_clean["image"].shape == (640, 640, 3)
        assert len(sample_clean["person_bbox"]) == 4

        sample_attack = generate_synthetic_sample(is_attack=True)
        assert sample_attack["is_attack"] is True
        assert sample_attack["confidence_delta"] > 0.0

    def test_evaluate_single_condition(self):
        """Verify condition evaluation generates valid metrics."""
        bench = RobustnessBenchmark(
            model_path="yolov8n.onnx",
            classifier_path="models/attack_classifier.joblib",
            threshold=0.65,
        )
        res: ConditionResult = bench.evaluate_condition(
            distance_m=2,
            angle_deg=0,
            illumination="ambient",
            num_samples_per_class=2,
        )
        assert res.num_samples == 4
        assert res.num_clean == 2
        assert res.num_attacked == 2
        assert 0.0 <= res.acc_with_defense <= 1.0
        assert 0.0 <= res.precision <= 1.0
        assert 0.0 <= res.recall <= 1.0
        assert res.total_latency_ms > 0.0
        assert res.sustained_fps > 0.0

    def test_report_export_csv_and_markdown(self, tmp_path):
        """Verify export of CSV and Markdown reports."""
        bench = RobustnessBenchmark(
            model_path="yolov8n.onnx",
            classifier_path="models/attack_classifier.joblib",
            threshold=0.65,
        )

        mock_condition = ConditionResult(
            distance=2,
            angle=0,
            illumination="ambient",
            num_samples=10,
            num_clean=5,
            num_attacked=5,
            acc_without_defense=0.50,
            acc_with_defense=1.00,
            tp=5,
            fp=0,
            tn=5,
            fn=0,
            precision=1.00,
            recall=1.00,
            fpr=0.00,
            f1_score=1.00,
            yolo_latency_ms=65.2,
            feature_latency_ms=1.8,
            classifier_latency_ms=0.05,
            total_latency_ms=67.05,
            sustained_fps=14.9,
        )

        report = BenchmarkReportData(
            condition_results=[mock_condition],
            overall_acc_without_defense=0.50,
            overall_acc_with_defense=1.00,
            overall_precision=1.00,
            overall_recall=1.00,
            overall_fpr=0.00,
            overall_f1=1.00,
            avg_yolo_ms=65.2,
            avg_feature_ms=1.8,
            avg_classifier_ms=0.05,
            avg_total_ms=67.05,
            avg_sustained_fps=14.9,
        )

        csv_file = tmp_path / "test_report.csv"
        md_file = tmp_path / "test_report.md"

        bench.export_csv(report, csv_file)
        bench.export_markdown(report, md_file)

        assert csv_file.exists()
        assert md_file.exists()

        csv_content = csv_file.read_text(encoding="utf-8")
        assert "distance" in csv_content
        assert "acc_with_defense" in csv_content

        md_content = md_file.read_text(encoding="utf-8")
        assert "ProYOLO Physical CCTV Robustness" in md_content
        assert "Without Defense (Vanilla YOLO)" in md_content
        assert "With ProYOLO Defense" in md_content
        assert "14.9 FPS" in md_content
