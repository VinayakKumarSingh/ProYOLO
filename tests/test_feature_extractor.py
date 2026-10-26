"""Unit tests for the CPU-optimized detection, tracking, and feature extraction pipeline.

Verifies:
1. YOLOv8ONNXDetector letterboxing, coordinate rescaling, and inference.
2. ByteTracker Kalman filtering, track association, and sliding-window history (N=10).
3. DualFeatureExtractor temporal anomaly calculation and visual patch signature entropy.
4. Robustness against boundary edge cases (zero-area, out-of-frame, missing detections).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.detection.detector import (
    Detection,
    YOLOv8ONNXDetector,
    neutralize_patch_regions,
)
from src.detection.feature_extractor import (
    FEATURE_NAMES,
    DualFeatureExtractor,
    FeaturePacket,
    SpatialFeatureNormClipper,
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

    def test_neutralize_patch_regions_inpainting(self, perturbed_patch_frame):
        """neutralize_patch_regions must inpaint the patch area with dilated margin."""
        patch_box = [220, 130, 300, 190]
        inpainted = neutralize_patch_regions(perturbed_patch_frame, [patch_box], dilate_ratio=0.10)

        assert inpainted.shape == perturbed_patch_frame.shape
        # Inpainted patch pixels must differ from original high-contrast noise
        original_patch = perturbed_patch_frame[130:190, 220:300]
        inpainted_patch = inpainted[130:190, 220:300]
        assert not np.array_equal(original_patch, inpainted_patch)

        # Inpainted patch should have lower gradient variance (smoother blending)
        orig_gray = cv2.cvtColor(original_patch, cv2.COLOR_BGR2GRAY)
        inp_gray = cv2.cvtColor(inpainted_patch, cv2.COLOR_BGR2GRAY)
        assert float(np.var(cv2.Laplacian(inp_gray, cv2.CV_32F))) < float(
            np.var(cv2.Laplacian(orig_gray, cv2.CV_32F))
        )

        # Empty patch list returns unchanged frame
        unchanged = neutralize_patch_regions(perturbed_patch_frame, [])
        assert np.array_equal(unchanged, perturbed_patch_frame)

    def test_detector_clipper_hook_registration(self):
        """YOLOv8ONNXDetector must register SpatialFeatureNormClipper onto provided YOLO model."""
        model = DummyYOLOBackbone()
        detector = YOLOv8ONNXDetector(model=model, session="mock_session")

        assert detector.clipper is not None
        assert detector.clipper.hook_handle is not None

    def test_detect_with_recovery_clean_pass(self):
        """On clean images with no patch ROIs, detect_with_recovery returns Pass 1 detections."""
        detector = YOLOv8ONNXDetector(session="mock_session")
        det = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 300.0]),
            confidence=0.88,
            class_id=0,
            class_name="person",
        )

        # Mock detect returning high-confidence person
        detector.detect = lambda frame, conf_threshold=None: [det]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_with_recovery(frame, conf_thresh=0.40, low_conf_thresh=0.12)

        assert len(results) == 1
        assert results[0].confidence == 0.88
        assert results[0].delta_p_obj == 0.0
        assert results[0].recovered_by_inpainting is False

    def test_detect_with_recovery_suppressed_person_recovered(self, perturbed_patch_frame):
        """Active Two-Pass pipeline recovers suppressed person, computing positive delta_p_obj."""
        detector = YOLOv8ONNXDetector(session="mock_session")

        person_box = np.array([100.0, 80.0, 250.0, 350.0], dtype=np.float32)
        patch_roi = [150.0, 120.0, 200.0, 180.0]

        # Pass 1 yields suppressed person (conf=0.20 < conf_thresh=0.40)
        p1_det = Detection(bbox=person_box, confidence=0.20, class_id=0, class_name="person")
        # Pass 2 (after inpainting) yields recovered person with conf=0.85
        p2_det = Detection(bbox=person_box, confidence=0.85, class_id=0, class_name="person")

        call_count = 0

        def mock_detect(frame, conf_threshold=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [p1_det]
            else:
                return [p2_det]

        detector.detect = mock_detect

        results = detector.detect_with_recovery(
            perturbed_patch_frame,
            conf_thresh=0.40,
            low_conf_thresh=0.12,
            patch_rois=[patch_roi],
        )

        assert call_count == 2, "Expected two-pass detection to trigger"
        assert len(results) == 1
        recovered_person = results[0]
        assert recovered_person.confidence == 0.85
        assert recovered_person.delta_p_obj == pytest.approx(0.85 - 0.20, abs=1e-3)
        assert recovered_person.recovered_by_inpainting is True

    def test_detect_with_recovery_completely_missed_person(self, perturbed_patch_frame):
        """When person was completely missed in Pass 1, delta_p_obj equals conf_pass2."""
        detector = YOLOv8ONNXDetector(session="mock_session")

        person_box = np.array([100.0, 80.0, 250.0, 350.0], dtype=np.float32)
        patch_roi = [150.0, 120.0, 200.0, 180.0]

        # Pass 1 yields nothing (complete evasion)
        p2_det = Detection(bbox=person_box, confidence=0.82, class_id=0, class_name="person")

        call_count = 0

        def mock_detect(frame, conf_threshold=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            else:
                return [p2_det]

        detector.detect = mock_detect

        results = detector.detect_with_recovery(
            perturbed_patch_frame,
            conf_thresh=0.40,
            low_conf_thresh=0.12,
            patch_rois=[patch_roi],
        )

        assert call_count == 2
        assert len(results) == 1
        recovered = results[0]
        assert recovered.confidence == 0.82
        assert recovered.delta_p_obj == pytest.approx(0.82 - 0.0, abs=1e-3)
        assert recovered.recovered_by_inpainting is True


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

    def test_suppressed_confidence_to_018_keeps_tracklet_alive_with_kalman_smoothing(self):
        """ByteTrack associates low-confidence detection (conf=0.18) with Kalman prediction."""
        tracker = ByteTracker(track_thresh=0.40, low_thresh=0.12, match_thresh=0.7)
        box = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)

        # Frame 0: High confidence detection
        det_high = Detection(bbox=box, confidence=0.85, class_id=0, class_name="person")
        tracks_0 = tracker.update([det_high], frame_idx=0)
        assert len(tracks_0) == 1
        original_id = tracks_0[0].track_id

        # Frame 1: Patch suppresses confidence to 0.18 and slightly jitters box
        det_low = Detection(bbox=box + 5.0, confidence=0.18, class_id=0, class_name="person")
        tracks_1 = tracker.update([det_low], frame_idx=1)

        assert len(tracks_1) == 1
        assert tracks_1[0].track_id == original_id
        # Smoothed box from Kalman filter is recorded in history
        smoothed_box = tracks_1[0].history[-1].bbox
        assert smoothed_box.shape == (4,)
        # Check that smoothed box is close to the expected position
        assert np.all(np.abs(smoothed_box - box) < 10.0)

    def test_temporal_alert_persistence_k3_of_n5(self):
        """Track alert confirms only if is_attack == True in at least K=3 out of N=5 frames."""
        det = Detection(
            bbox=np.array([50.0, 50.0, 150.0, 250.0]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
        track = Track(detection=det, frame_idx=0)

        # Frame 1: 1 attack -> 1/1 -> False
        assert track.record_attack_prediction(True) is False
        assert track.confirmed_attack_alert is False

        # Frame 2: 0 attack -> 1/2 -> False
        assert track.record_attack_prediction(False) is False
        assert track.confirmed_attack_alert is False

        # Frame 3: 1 attack -> 2/3 -> False
        assert track.record_attack_prediction(True) is False
        assert track.confirmed_attack_alert is False

        # Frame 4: 1 attack -> 3/4 -> True (K=3 reached!)
        assert track.record_attack_prediction(True) is True
        assert track.confirmed_attack_alert is True

        # Frame 5: 0 attack -> 3/5 -> True (still 3 of last 5)
        assert track.record_attack_prediction(False) is True
        assert track.confirmed_attack_alert is True

        # Frame 6: 0 attack -> 2/5 -> False (drops below 3 of last 5)
        assert track.record_attack_prediction(False) is False
        assert track.confirmed_attack_alert is False

    def test_dpatch_global_persistence_deque(self):
        """Tracker records global DPATCH candidate alerts requiring K=3 of N=5 consecutive frames."""
        tracker = ByteTracker()
        assert tracker.confirmed_attack_alert is False

        # Frame 1: Attack detected
        assert tracker.record_dpatch_attack(True) is False
        assert tracker.confirmed_attack_alert is False

        # Frame 2: Attack detected
        assert tracker.record_dpatch_attack(True) is False
        assert tracker.confirmed_attack_alert is False

        # Frame 3: Attack detected -> K=3 / 3 -> True!
        assert tracker.record_dpatch_attack(True) is True
        assert tracker.confirmed_attack_alert is True

        # Frame 4 & 5: Clean frames -> sum is 3 of 5 -> True
        assert tracker.record_dpatch_attack(False) is True
        assert tracker.record_dpatch_attack(False) is True
        assert tracker.confirmed_attack_alert is True

        # Frame 6: Clean frame -> sum is 2 of 5 -> False
        assert tracker.record_dpatch_attack(False) is False
        assert tracker.confirmed_attack_alert is False

    def test_tracker_recovered_by_inpainting_metadata(self):
        """Tracker preserves recovered_by_inpainting and delta_p_obj across updates."""
        tracker = ByteTracker()
        box = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)

        det_rec = Detection(
            bbox=box,
            confidence=0.75,
            class_id=0,
            class_name="person",
            recovered_by_inpainting=True,
            delta_p_obj=0.38,
        )
        tracks = tracker.update([det_rec], frame_idx=0)
        assert len(tracks) == 1
        assert tracks[0].recovered_by_inpainting is True
        assert pytest.approx(tracks[0].delta_p_obj, abs=1e-3) == 0.38


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

    def test_new_features_appended_and_dpatch_discrimination(
        self, feature_extractor, clean_frame, perturbed_patch_frame
    ):
        """Verify max_grid_norm_ratio and jpeg_recompression_delta are in FEATURE_NAMES and discriminate patches."""
        assert len(FEATURE_NAMES) == 15
        assert FEATURE_NAMES[-2] == "max_grid_norm_ratio"
        assert FEATURE_NAMES[-1] == "jpeg_recompression_delta"

        person_box = np.array([200, 100, 320, 300])
        track_clean = Track(
            Detection(bbox=person_box, confidence=0.90, class_id=0, class_name="person"),
            frame_idx=0,
        )
        track_pert = Track(
            Detection(bbox=person_box, confidence=0.88, class_id=0, class_name="person"),
            frame_idx=0,
        )

        packet_clean = feature_extractor.extract_single(clean_frame, track_clean)
        packet_pert = feature_extractor.extract_single(perturbed_patch_frame, track_pert)

        assert "max_grid_norm_ratio" in packet_clean.feature_dict
        assert "jpeg_recompression_delta" in packet_clean.feature_dict
        assert packet_clean.feature_vector.shape == (15,)

        # Perturbed patch frame must yield higher jpeg_recompression_delta due to high-frequency artifacts
        assert (
            packet_pert.feature_dict["jpeg_recompression_delta"]
            > packet_clean.feature_dict["jpeg_recompression_delta"]
        )


# ============================================================================
# 4. Spatial Feature Norm Clipper Tests (DPATCH Counter-Measures)
# ============================================================================


class DummySPPF(nn.Module):
    """Dummy SPPF block for testing YOLO backbone hook targeting."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyYOLOBackbone(nn.Module):
    """Synthetic multi-stage YOLO backbone with SPPF layer."""

    def __init__(self, in_channels: int = 32, num_stages: int = 9) -> None:
        super().__init__()
        layers = []
        c = in_channels
        for i in range(num_stages - 1):
            layers.append(nn.Conv2d(c, c, kernel_size=3, padding=1))
        layers.append(DummySPPF())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TestSpatialFeatureNormClipper:
    """Verifies SpatialFeatureNormClipper activation outlier capping and ROI localization."""

    def test_outlier_spike_clamping_dummy_tensor(self):
        """Verify clipper caps synthetic outlier spikes in a dummy (1, C, H, W) tensor."""
        b, c, h, w = 1, 64, 16, 16
        # Generate baseline feature map with uniform background activation
        torch.manual_seed(42)
        f_dummy = torch.ones((b, c, h, w), dtype=torch.float32)

        # Inject extreme synthetic outlier spike at cell (8, 8) (simulating DPATCH monopolization)
        spike_val = 60.0
        f_dummy[0, :, 8, 8] = spike_val

        # Pre-clipping channel norm at spike location
        pre_spike_norm = float(torch.linalg.vector_norm(f_dummy[0, :, 8, 8], ord=2).item())
        assert pre_spike_norm == pytest.approx(spike_val * np.sqrt(c), rel=1e-3)

        clipper = SpatialFeatureNormClipper(alpha=2.0, image_size=(640, 640))
        f_clipped = clipper(f_dummy)

        # Output shape must be strictly preserved
        assert f_clipped.shape == (b, c, h, w)

        # Post-clipping channel norm at spike location
        post_spike_norm = float(torch.linalg.vector_norm(f_clipped[0, :, 8, 8], ord=2).item())

        # Expected threshold tau = mu_E + 2.0 * sigma_E
        tau = clipper.last_tau
        assert tau > 0.0

        # Outlier spike must be clamped to tau
        assert post_spike_norm <= tau + 1e-4
        assert post_spike_norm < pre_spike_norm

        # Unaffected cells (0, 0) must remain identical
        normal_cell_pre = f_dummy[0, :, 0, 0]
        normal_cell_post = f_clipped[0, :, 0, 0]
        assert torch.allclose(normal_cell_post, normal_cell_pre, atol=1e-5)

        # Verify max_grid_norm_ratio captures DPATCH monopolization
        assert clipper.max_grid_norm_ratio > 3.0

    def test_patch_rois_localization_dummy_tensor(self):
        """Verify clipper returns valid candidate patch_rois bounding boxes in image coordinates."""
        b, c, h, w = 1, 32, 16, 16
        f_dummy = torch.ones((b, c, h, w), dtype=torch.float32)

        # Spike at spatial location (row=6, col=10)
        f_dummy[0, :, 6, 10] = 50.0

        clipper = SpatialFeatureNormClipper(alpha=2.0, image_size=(640, 640))
        _ = clipper(f_dummy)

        rois = clipper.patch_rois
        assert isinstance(rois, list)
        assert len(rois) > 0, "Expected at least one candidate patch ROI around the spike"

        # Check candidate bounding box coordinates
        best_roi = rois[0]
        assert len(best_roi) == 4
        x1, y1, x2, y2 = best_roi

        # Coordinates must be valid image coordinates
        assert 0.0 <= x1 < x2 <= 640.0
        assert 0.0 <= y1 < y2 <= 640.0

        # Spike was at grid cell (row=6, col=10), which corresponds to (x=400, y=240) in 640x640
        expected_x = 10.0 * (640.0 / w)
        expected_y = 6.0 * (640.0 / h)

        # Candidate ROI must cover the spike location
        assert x1 <= expected_x <= x2
        assert y1 <= expected_y <= y2

    def test_clean_tensor_no_clamping_and_empty_rois(self):
        """Uniform/normal feature map without outlier spikes must not trigger clamping or ROIs."""
        b, c, h, w = 1, 32, 8, 8
        f_clean = torch.ones((b, c, h, w), dtype=torch.float32) * 2.0

        clipper = SpatialFeatureNormClipper(alpha=2.0, image_size=(640, 640))
        f_clipped = clipper(f_clean)

        # Clean tensor should be identical after clipping
        assert torch.allclose(f_clipped, f_clean, atol=1e-5)
        # No ROIs should be generated
        assert len(clipper.patch_rois) == 0
        assert clipper.max_grid_norm_ratio == pytest.approx(1.0, rel=0.1)

    def test_pytorch_forward_hook_attachment_and_removal(self):
        """Verify clipper attaches as forward hook to YOLO backbone stage before SPPF."""
        model = DummyYOLOBackbone(in_channels=32, num_stages=9)
        clipper = SpatialFeatureNormClipper(alpha=2.0)

        # Attach to the layer preceding SPPF
        clipper.attach(model)
        assert clipper.hook_handle is not None

        # Create dummy input tensor (B=1, C=32, H=16, W=16)
        x_in = torch.ones((1, 32, 16, 16), dtype=torch.float32)
        # Inject spike
        x_in[0, :, 4, 4] = 40.0

        # Forward pass triggers hook automatically
        out = model(x_in)
        assert out is not None
        assert clipper.last_tau > 0.0

        # Detach hook
        clipper.detach()
        assert clipper.hook_handle is None

    def test_jpeg_recompression_difference_integration(self, perturbed_patch_frame):
        """Verify merging spatial anomaly heatmap with JPEG recompression difference map."""
        b, c, h, w = 1, 32, 16, 16
        f_dummy = torch.ones((b, c, h, w), dtype=torch.float32)
        # Corresponds roughly to chest region in 640x640: row ~ 3..5, col ~ 5..8
        f_dummy[0, :, 4, 6] = 50.0

        clipper = SpatialFeatureNormClipper(alpha=2.0)
        clipper.set_image(perturbed_patch_frame)
        _ = clipper(f_dummy)

        assert clipper.jpeg_diff_map is not None
        assert clipper.jpeg_diff_map.shape == (640, 640)
        assert clipper.merged_heatmap is not None
        assert clipper.merged_heatmap.shape == (640, 640)
        assert len(clipper.patch_rois) > 0

        # Candidate ROI must yield elevated jpeg_recompression_delta
        assert clipper.jpeg_recompression_delta > 1.5
