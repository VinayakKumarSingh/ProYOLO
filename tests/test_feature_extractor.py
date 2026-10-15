"""Unit tests for the CPU-optimized detection, tracking, and feature extraction pipeline.

Verifies:
1. YOLOv8ONNXDetector letterboxing, coordinate rescaling, and inference.
2. ByteTracker Kalman filtering, track association, and sliding-window history (N=10).
3. DualFeatureExtractor temporal anomaly calculation and visual patch signature entropy.
4. Robustness against boundary edge cases (zero-area, out-of-frame, missing detections).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.detector import Detection, YOLOv8ONNXDetector
from src.detection.feature_extractor import (
    FEATURE_NAMES,
    DualFeatureExtractor,
    FeaturePacket,
)
from src.detection.tracker import ByteTracker, Track, TrackState, bbox_iou


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def clean_frame():
    """Generates a synthetic 640x640 surveillance frame with smooth background."""
    frame = np.ones((640, 640, 3), dtype=np.uint8) * 160
    # Simulate smooth clothing on person torso [100:300, 200:320]
    frame[100:300, 200:320] = [60, 70, 90]
    return frame


@pytest.fixture
def perturbed_patch_frame():
    """Generates a synthetic frame containing a high-contrast optical perturbation pattern."""
    frame = np.ones((640, 640, 3), dtype=np.uint8) * 160
    # Person base
    frame[100:300, 200:320] = [60, 70, 90]
    # Insert high-frequency, multi-colored optical pattern on chest [130:190, 220:300]
    np.random.seed(42)
    patch = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
    frame[130:190, 220:300] = patch
    return frame


@pytest.fixture
def feature_extractor():
    """Instantiates a DualFeatureExtractor."""
    return DualFeatureExtractor()


# ============================================================================
# 1. Detector Tests
# ============================================================================


class TestYOLOv8ONNXDetector:
    """Verifies detector letterboxing, coordinate scaling, and inference."""

    def test_letterbox_aspect_ratio_preservation(self):
        """Letterboxing must preserve aspect ratio and pad symmetrically."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, (pad_x, pad_y) = YOLOv8ONNXDetector.letterbox(
            img, target_size=(480, 480)
        )

        assert padded.shape == (480, 480, 3)
        assert scale == pytest.approx(480.0 / 640.0, abs=1e-3)
        assert pad_x == 0.0
        assert pad_y > 0.0  # Vertical padding

    def test_detector_inference_on_synthetic_frame(self, clean_frame):
        """Verify detector runs forward inference and returns Detection objects."""
        # Check if yolov8n.onnx is present in workspace
        import os
        model_path = "yolov8n.onnx"
        if not os.path.exists(model_path):
            pytest.skip("yolov8n.onnx not found in workspace")

        detector = YOLOv8ONNXDetector(
            model_path=model_path,
            input_size=(480, 480),
            conf_threshold=0.01,  # Low threshold to capture synthetic structures
            target_classes=None,  # All classes
        )
        dets = detector.detect(clean_frame)
        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)
            assert len(d.bbox) == 4
            assert 0.0 <= d.confidence <= 1.0


# ============================================================================
# 2. ByteTracker Tests
# ============================================================================


class TestByteTracker:
    """Verifies tracking continuity, Kalman predictions, and history buffer cap."""

    def test_track_history_sliding_window_cap(self):
        """Track history must never exceed max_history (default N=10)."""
        tracker = ByteTracker(max_history=10)
        det_box = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)

        # Feed 25 consecutive detections with small motion
        for frame_idx in range(25):
            jittered_box = det_box + frame_idx * 1.5
            det = Detection(bbox=jittered_box, confidence=0.90, class_id=0, class_name="person")
            active_tracks = tracker.update([det], frame_idx=frame_idx)

        assert len(active_tracks) == 1
        track = active_tracks[0]
        # History must be strictly capped at 10
        assert len(track.history) == 10
        assert track.history[-1].frame_idx == 24
        assert track.history[0].frame_idx == 15

    def test_low_confidence_association_bytetrack(self):
        """ByteTrack must maintain track when confidence drops to low tier."""
        tracker = ByteTracker(track_thresh=0.5, match_thresh=0.7)
        box = np.array([50.0, 50.0, 150.0, 250.0], dtype=np.float32)

        # Frame 0: High confidence initialization
        det_high = Detection(bbox=box, confidence=0.85, class_id=0, class_name="person")
        tracks_0 = tracker.update([det_high], frame_idx=0)
        assert len(tracks_0) == 1
        original_id = tracks_0[0].track_id

        # Frame 1: Sudden confidence drop to 0.20 (simulating patch suppression)
        det_low = Detection(bbox=box + 2.0, confidence=0.20, class_id=0, class_name="person")
        tracks_1 = tracker.update([det_low], frame_idx=1)

        # Track should be preserved via second-stage association
        assert len(tracks_1) == 1
        assert tracks_1[0].track_id == original_id

    def test_bbox_iou_computation(self):
        """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
        b1 = np.array([0, 0, 10, 10])
        b2 = np.array([0, 0, 10, 10])
        assert bbox_iou(b1, b2) == pytest.approx(1.0)

        b3 = np.array([5, 0, 15, 10])  # 50% overlap
        assert bbox_iou(b1, b3) == pytest.approx(1.0 / 3.0)

        b4 = np.array([20, 20, 30, 30])  # Disjoint
        assert bbox_iou(b1, b4) == pytest.approx(0.0)


# ============================================================================
# 3. Feature Extractor Tests
# ============================================================================


class TestDualFeatureExtractor:
    """Verifies temporal anomaly and visual entropy feature extraction."""

    def test_temporal_features_clean_vs_perturbed(self, feature_extractor):
        """Perturbed tracks with confidence drops and area fluctuation must yield high anomaly scores."""
        # 1. Clean track sequence: steady confidence, stable area
        clean_track = Track(
            Detection(bbox=np.array([100, 100, 200, 300]), confidence=0.95, class_id=0, class_name="person"),
            frame_idx=0,
            max_history=10,
        )
        for i in range(1, 6):
            clean_track.update(
                Detection(bbox=np.array([100 + i, 100, 200 + i, 300]), confidence=0.94, class_id=0, class_name="person"),
                frame_idx=i,
            )

        clean_feats = feature_extractor.compute_temporal_features(clean_track)
        assert clean_feats["conf_drop"] == pytest.approx(0.0, abs=0.05)
        assert clean_feats["flicker_rate"] == 0.0
        assert clean_feats["area_stability"] < 0.05
        assert clean_feats["mean_consecutive_iou"] > 0.85

        # 2. Perturbed track sequence: confidence collapse, area fluctuations, missed frame
        pert_track = Track(
            Detection(bbox=np.array([100, 100, 200, 300]), confidence=0.92, class_id=0, class_name="person"),
            frame_idx=0,
            max_history=10,
        )
        pert_track.update(
            Detection(bbox=np.array([100, 100, 200, 300]), confidence=0.90, class_id=0, class_name="person"),
            frame_idx=1,
        )
        # Frame 2 is skipped (flicker drop), frame 3 has collapsed area and confidence drop
        pert_track.update(
            Detection(bbox=np.array([110, 150, 160, 250]), confidence=0.25, class_id=0, class_name="person"),
            frame_idx=3,
        )

        pert_feats = feature_extractor.compute_temporal_features(pert_track)
        # Confidence drop must be substantial
        assert pert_feats["conf_drop"] > 0.50
        # Flicker rate must be positive due to missed frame 2
        assert pert_feats["flicker_rate"] > 0.0
        # Area stability metric must detect area change
        assert pert_feats["area_stability"] > 0.15

    def test_visual_features_smooth_vs_patch(
        self, feature_extractor, clean_frame, perturbed_patch_frame
    ):
        """Adversarial optical patch must produce higher Laplacian entropy and color dispersion."""
        person_box = np.array([200, 100, 320, 300])

        roi_clean = feature_extractor.crop_upper_body(clean_frame, person_box)
        roi_patch = feature_extractor.crop_upper_body(perturbed_patch_frame, person_box)

        vis_clean = feature_extractor.compute_visual_features(roi_clean)
        vis_patch = feature_extractor.compute_visual_features(roi_patch)

        # Gradient entropy and variance should be noticeably higher on the optical patch
        assert vis_patch["laplacian_var_norm"] > vis_clean["laplacian_var_norm"]
        assert vis_patch["laplacian_entropy"] > vis_clean["laplacian_entropy"]

        # Color entropy in Hue and Saturation should also be higher due to multi-colored patch
        assert vis_patch["color_hue_entropy"] > vis_clean["color_hue_entropy"]

    def test_feature_vector_dimensions_and_finite_bounds(self, feature_extractor, clean_frame):
        """Extracted 1D feature vector must strictly match FEATURE_NAMES length and be finite."""
        track = Track(
            Detection(bbox=np.array([150, 80, 280, 350]), confidence=0.90, class_id=0, class_name="person"),
            frame_idx=1,
            max_history=10,
        )
        packet = feature_extractor.extract_single(clean_frame, track)

        assert isinstance(packet, FeaturePacket)
        assert packet.track_id == track.track_id
        assert isinstance(packet.feature_vector, np.ndarray)
        assert packet.feature_vector.shape == (len(FEATURE_NAMES),)
        assert packet.feature_vector.dtype == np.float32

        # Assert no NaNs or Infs
        assert not np.isnan(packet.feature_vector).any()
        assert not np.isinf(packet.feature_vector).any()

        # Batch extraction matrix shape
        matrix = feature_extractor.extract_feature_matrix(clean_frame, [track, track])
        assert matrix.shape == (2, len(FEATURE_NAMES))

    def test_empty_roi_boundary_handling(self, feature_extractor, clean_frame):
        """Extractor must handle empty, inverted, or out-of-frame boxes gracefully without crashing."""
        # Box completely outside frame
        out_box = np.array([-100, -100, -50, -50])
        track = Track(
            Detection(bbox=out_box, confidence=0.5, class_id=0, class_name="person"),
            frame_idx=1,
        )
        packet = feature_extractor.extract_single(clean_frame, track)
        assert packet.feature_vector.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(packet.feature_vector).any()
