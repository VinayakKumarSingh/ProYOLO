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
    is_inside_torso,
    is_near_frame_edge,
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

    def test_is_near_frame_edge_guard(self):
        """is_near_frame_edge returns True if box touches or is within 5% frame margin."""
        frame_shape = (480, 640)
        # Margin = 0.05 * 640 = 32px horizontally, 0.05 * 480 = 24px vertically

        # Left edge touch (x1 <= 32)
        assert is_near_frame_edge([10, 100, 100, 200], frame_shape, margin=0.05) is True
        # Top edge touch (y1 <= 24)
        assert is_near_frame_edge([100, 15, 200, 200], frame_shape, margin=0.05) is True
        # Right edge touch (x2 >= 640 - 32 = 608)
        assert is_near_frame_edge([500, 100, 615, 200], frame_shape, margin=0.05) is True
        # Bottom edge touch (y2 >= 480 - 24 = 456)
        assert is_near_frame_edge([100, 300, 200, 465], frame_shape, margin=0.05) is True
        # Interior box (well away from all edges)
        assert is_near_frame_edge([100, 100, 250, 300], frame_shape, margin=0.05) is False

    def test_is_inside_torso_guard(self):
        """is_inside_torso returns True only if patch center is in torso and area ratio is 4%..45%."""
        # Person box: width = 100, height = 200, area = 20000
        person_box = [100.0, 100.0, 200.0, 300.0]
        # Torso center bounds: x in [115, 185], y in [130, 260]

        # Valid chest/torso patch: [130, 160, 170, 220] -> cx=150, cy=190, area=2400 (12%)
        valid_patch = [130.0, 160.0, 170.0, 220.0]
        assert is_inside_torso(valid_patch, person_box) is True

        # Patch too high (head/face region): cy = 115 < 130
        head_patch = [135.0, 105.0, 165.0, 125.0]
        assert is_inside_torso(head_patch, person_box) is False

        # Patch too low (feet/legs region): cy = 285 > 260
        feet_patch = [135.0, 275.0, 165.0, 295.0]
        assert is_inside_torso(feet_patch, person_box) is False

        # Patch too small (< 4% person area): area = 10 * 10 = 100 (0.5%)
        tiny_patch = [145.0, 185.0, 155.0, 195.0]
        assert is_inside_torso(tiny_patch, person_box) is False

        # Patch too large (> 45% person area): area = 80 * 150 = 12000 (60%)
        huge_patch = [110.0, 120.0, 190.0, 270.0]
        assert is_inside_torso(huge_patch, person_box) is False

    def test_detect_with_recovery_preserves_80_coco_classes(self):
        """detect_with_recovery retains non-person COCO classes untouched with is_attack=False."""
        detector = YOLOv8ONNXDetector(session="mock_session")

        p_det = Detection(bbox=np.array([100.0, 100.0, 200.0, 300.0]), confidence=0.88, class_id=0, class_name="person")
        laptop_det = Detection(bbox=np.array([250.0, 200.0, 350.0, 300.0]), confidence=0.78, class_id=63, class_name="laptop")
        phone_det = Detection(bbox=np.array([180.0, 150.0, 210.0, 190.0]), confidence=0.82, class_id=67, class_name="cell phone")

        detector.detect = lambda frame, conf_threshold=None: [p_det, laptop_det, phone_det]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_with_recovery(frame, conf_thresh=0.40, low_conf_thresh=0.15)

        assert len(results) == 3
        class_names = [d.class_name for d in results]
        assert "person" in class_names
        assert "laptop" in class_names
        assert "cell phone" in class_names

        for d in results:
            if d.class_name != "person":
                assert d.is_attack is False
                assert d.delta_p_obj == 0.0
                assert d.recovered_by_inpainting is False
                # Verify dict interface
                d_dict = d.to_dict()
                assert d_dict["is_attack"] is False
                assert d["is_attack"] is False
                assert d["class_name"] == d.class_name

    def test_detect_with_recovery_edge_guard_rejects_border_patch(self):
        """Patch ROIs near frame edges must be filtered out and never trigger Pass 2."""
        detector = YOLOv8ONNXDetector(session="mock_session")
        p_det = Detection(bbox=np.array([200.0, 150.0, 350.0, 400.0]), confidence=0.85, class_id=0, class_name="person")

        call_count = 0

        def mock_detect(frame, conf_threshold=None):
            nonlocal call_count
            call_count += 1
            return [p_det]

        detector.detect = mock_detect
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Border patch at x1=5 (touches margin 5%)
        border_patch = [5.0, 50.0, 40.0, 100.0]
        results = detector.detect_with_recovery(
            frame,
            conf_thresh=0.40,
            low_conf_thresh=0.15,
            patch_rois=[border_patch],
        )

        # Edge guard must filter out border patch; Pass 2 should NOT trigger
        assert call_count == 1
        assert len(results) == 1
        assert results[0].class_name == "person"
        assert results[0].recovered_by_inpainting is False

    def test_detect_with_recovery_torso_guard_triggers_pass2_and_attaches_metadata(self):
        """Valid interior patch on torso triggers Pass 2, computes delta_p_obj, and attaches patch_roi."""
        detector = YOLOv8ONNXDetector(session="mock_session")

        p_suppressed = Detection(
            bbox=np.array([100.0, 80.0, 250.0, 350.0]),
            confidence=0.18,
            class_id=0,
            class_name="person",
        )
        chair_det = Detection(
            bbox=np.array([400.0, 300.0, 500.0, 450.0]),
            confidence=0.80,
            class_id=56,
            class_name="chair",
        )
        p_recovered = Detection(
            bbox=np.array([100.0, 80.0, 250.0, 350.0]),
            confidence=0.86,
            class_id=0,
            class_name="person",
        )

        call_count = 0

        def mock_detect(frame, conf_threshold=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [p_suppressed, chair_det]
            else:
                return [p_recovered]

        detector.detect = mock_detect
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Interior patch inside person's torso region
        torso_patch = [150.0, 140.0, 200.0, 200.0]
        results = detector.detect_with_recovery(
            frame,
            conf_thresh=0.40,
            low_conf_thresh=0.15,
            patch_rois=[torso_patch],
        )

        assert call_count == 2, "Pass 2 should trigger for valid torso patch"
        assert len(results) == 2

        person_res = [d for d in results if d.class_name == "person"][0]
        chair_res = [d for d in results if d.class_name == "chair"][0]

        assert person_res.confidence == 0.86
        assert person_res.recovered_by_inpainting is True
        assert person_res.delta_p_obj == pytest.approx(0.86 - 0.18, abs=1e-3)
        assert person_res.is_attack is True
        assert person_res.patch_roi == torso_patch

        # Dict access checks
        assert person_res["delta_p_obj"] == person_res.delta_p_obj
        assert person_res["patch_roi"] == torso_patch
        assert person_res.to_dict()["recovered_by_inpainting"] is True

        assert chair_res.confidence == 0.80
        assert chair_res.class_name == "chair"
        assert chair_res.is_attack is False
        assert chair_res.delta_p_obj == 0.0

    def test_optional_patch_model_detection(self):
        """Detector queries secondary patch_model to predict tight adversarial_patch boxes."""
        class MockBox:
            xyxy = torch.tensor([[120.0, 130.0, 180.0, 190.0]])

        class MockResult:
            boxes = MockBox()

        class MockPatchModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        detector = YOLOv8ONNXDetector(session="mock_session", patch_model=MockPatchModel())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rois = detector.predict_patch_rois(frame)

        assert len(rois) == 1
        assert rois[0] == [120.0, 130.0, 180.0, 190.0]


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

    def test_temporal_alert_persistence_k4_of_n5(self):
        """Track alert confirms ONLY when person tracklet is NOT near frame edge and is_attack in >= 4/5 frames."""
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

        # Frame 2: 1 attack -> 2/2 -> False
        assert track.record_attack_prediction(True) is False
        assert track.confirmed_attack_alert is False

        # Frame 3: 1 attack -> 3/3 -> False
        assert track.record_attack_prediction(True) is False
        assert track.confirmed_attack_alert is False

        # Frame 4: 1 attack -> 4/4 -> True (K=4 reached!)
        assert track.record_attack_prediction(True) is True
        assert track.confirmed_attack_alert is True

        # Frame 5: 0 attack -> 4/5 -> True (still 4 of last 5)
        assert track.record_attack_prediction(False) is True
        assert track.confirmed_attack_alert is True

        # Frame 6: 0 attack -> 3/5 -> False (drops below 4 of last 5)
        assert track.record_attack_prediction(False) is False
        assert track.confirmed_attack_alert is False

        # Frame Edge Guard: Person touching border (x1=5 < 32px margin) must NEVER confirm attack alert
        edge_det = Detection(
            bbox=np.array([5.0, 50.0, 120.0, 250.0]),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
        edge_track = Track(detection=edge_det, frame_idx=0)
        for _ in range(5):
            edge_track.record_attack_prediction(True)
        assert edge_track.confirmed_attack_alert is False

        # Non-Person Guard: Non-person object (e.g. car, chair) must NEVER confirm attack alert
        car_det = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
        car_track = Track(detection=car_det, frame_idx=0)
        for _ in range(5):
            car_track.record_attack_prediction(True)
        assert car_track.confirmed_attack_alert is False

    def test_dpatch_global_persistence_deque(self):
        """Tracker records global DPATCH candidate alerts requiring K=4 of N=5 consecutive frames."""
        tracker = ByteTracker()
        assert tracker.confirmed_attack_alert is False

        # Frame 1: Attack detected (1/1)
        assert tracker.record_dpatch_attack(True) is False
        assert tracker.confirmed_attack_alert is False

        # Frame 2: Attack detected (2/2)
        assert tracker.record_dpatch_attack(True) is False
        assert tracker.confirmed_attack_alert is False

        # Frame 3: Attack detected (3/3)
        assert tracker.record_dpatch_attack(True) is False
        assert tracker.confirmed_attack_alert is False

        # Frame 4: Attack detected -> K=4 / 4 -> True!
        assert tracker.record_dpatch_attack(True) is True
        assert tracker.confirmed_attack_alert is True

        # Frame 5: Clean frame -> sum is 4 of 5 -> True
        assert tracker.record_dpatch_attack(False) is True
        assert tracker.confirmed_attack_alert is True

        # Frame 6: Clean frame -> sum is 3 of 5 -> False
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

    def test_bytetrack_preserves_80_coco_classes_and_prevents_cross_class_association(self):
        """ByteTracker tracks multi-class detections and prevents cross-class tracklet stealing."""
        tracker = ByteTracker(track_thresh=0.40, low_thresh=0.15, match_thresh=0.7)

        # Frame 0: Multiple distinct COCO classes detected with high confidence
        p_det = Detection(bbox=np.array([100.0, 100.0, 200.0, 300.0]), confidence=0.88, class_id=0, class_name="person")
        chair_det = Detection(bbox=np.array([300.0, 200.0, 400.0, 300.0]), confidence=0.78, class_id=56, class_name="chair")
        phone_det = Detection(bbox=np.array([150.0, 150.0, 190.0, 190.0]), confidence=0.82, class_id=67, class_name="cell phone")
        bottle_det = Detection(bbox=np.array([500.0, 100.0, 530.0, 200.0]), confidence=0.85, class_id=39, class_name="bottle")

        tracks_0 = tracker.update([p_det, chair_det, phone_det, bottle_det], frame_idx=0)
        assert len(tracks_0) == 4

        track_map = {t.class_name: t for t in tracks_0}
        assert "person" in track_map and track_map["person"].class_id == 0
        assert "chair" in track_map and track_map["chair"].class_id == 56
        assert "cell phone" in track_map and track_map["cell phone"].class_id == 67
        assert "bottle" in track_map and track_map["bottle"].class_id == 39

        person_tid = track_map["person"].track_id
        chair_tid = track_map["chair"].track_id

        # Frame 1: Low-confidence detection for person (conf=0.22 in range 0.15 <= conf < 0.40)
        # and slightly moved chair (conf=0.75)
        p_low = Detection(bbox=np.array([102.0, 101.0, 202.0, 301.0]), confidence=0.22, class_id=0, class_name="person")
        chair_next = Detection(bbox=np.array([302.0, 201.0, 402.0, 301.0]), confidence=0.75, class_id=56, class_name="chair")

        tracks_1 = tracker.update([p_low, chair_next], frame_idx=1)
        assert len(tracks_1) == 2
        t1_map = {t.class_name: t for t in tracks_1}

        # Second tier association associates low confidence person to same track_id
        assert t1_map["person"].track_id == person_tid
        assert t1_map["chair"].track_id == chair_tid
        assert t1_map["person"].class_name == "person"
        assert t1_map["chair"].class_name == "chair"

        # Frame 2: Cross-class collision test. A chair appears at exact previous location of lost phone.
        # It must NOT associate with the cell phone tracklet because class IDs differ.
        imposter_chair = Detection(bbox=np.array([150.0, 150.0, 190.0, 190.0]), confidence=0.80, class_id=56, class_name="chair")
        tracks_2 = tracker.update([imposter_chair], frame_idx=2)
        assert len(tracks_2) == 1
        assert tracks_2[0].class_name == "chair"
        assert tracks_2[0].class_id == 56


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

    def test_border_activation_spikes_ignored_and_never_returned_in_rois(self):
        """Synthetic activation spikes on the outer frame border (e.g., x=0 or y=0) are ignored and never returned in patch_rois."""
        b, c, h, w = 1, 32, 16, 16
        torch.manual_seed(42)
        f_dummy = torch.ones((b, c, h, w), dtype=torch.float32) * 2.0

        # Inject extreme synthetic activation spikes on the outer frame border (x=0, y=0, bottom, right)
        f_dummy[0, :, 0, 5] = 90.0   # top edge (y=0)
        f_dummy[0, :, 7, 0] = 85.0   # left edge (x=0)
        f_dummy[0, :, 15, 10] = 95.0 # bottom edge (y=H-1)
        f_dummy[0, :, 10, 15] = 80.0 # right edge (x=W-1)

        clipper = SpatialFeatureNormClipper(alpha=2.0, image_size=(640, 640))
        f_clipped = clipper(f_dummy)

        # Output shape must be strictly preserved
        assert f_clipped.shape == (b, c, h, w)

        # Border spikes must be completely ignored (unclamped)
        assert torch.allclose(f_clipped[0, :, 0, 5], f_dummy[0, :, 0, 5], atol=1e-5)
        assert torch.allclose(f_clipped[0, :, 7, 0], f_dummy[0, :, 7, 0], atol=1e-5)
        assert torch.allclose(f_clipped[0, :, 15, 10], f_dummy[0, :, 15, 10], atol=1e-5)
        assert torch.allclose(f_clipped[0, :, 10, 15], f_dummy[0, :, 10, 15], atol=1e-5)

        # Border spikes must NEVER be returned in patch_rois
        assert len(clipper.patch_rois) == 0

        # max_grid_norm_ratio must ignore border spikes and remain ~1.0 for uniform interior
        assert clipper.max_grid_norm_ratio == pytest.approx(1.0, rel=0.1)

    def test_border_spikes_ignored_while_interior_spikes_clamped_and_localized(self):
        """Synthetic border spikes are ignored while interior spikes are clamped and localized."""
        b, c, h, w = 1, 32, 16, 16
        torch.manual_seed(42)
        f_dummy = torch.ones((b, c, h, w), dtype=torch.float32) * 2.0

        # 1. Border spikes: (y=0, x=5) and (y=8, x=0)
        f_dummy[0, :, 0, 5] = 100.0
        f_dummy[0, :, 8, 0] = 100.0

        # 2. Interior spike: (y=8, x=8)
        interior_spike_val = 60.0
        f_dummy[0, :, 8, 8] = interior_spike_val

        pre_interior_norm = float(torch.linalg.vector_norm(f_dummy[0, :, 8, 8], ord=2).item())

        clipper = SpatialFeatureNormClipper(alpha=2.0, image_size=(640, 640))
        f_clipped = clipper(f_dummy)

        # Border spikes are ignored and unclamped
        assert torch.allclose(f_clipped[0, :, 0, 5], f_dummy[0, :, 0, 5], atol=1e-5)
        assert torch.allclose(f_clipped[0, :, 8, 0], f_dummy[0, :, 8, 0], atol=1e-5)

        # Interior spike IS clamped
        post_interior_norm = float(torch.linalg.vector_norm(f_clipped[0, :, 8, 8], ord=2).item())
        tau = clipper.last_tau
        assert post_interior_norm <= tau + 1e-4
        assert post_interior_norm < pre_interior_norm

        # Candidate patch_rois must be returned and localize the interior spike
        assert len(clipper.patch_rois) > 0
        best_roi = clipper.patch_rois[0]

        # In 640x640: cell (8, 8) in 16x16 grid maps to (320, 320)
        expected_x = 8.0 * (640.0 / w)
        expected_y = 8.0 * (640.0 / h)
        assert best_roi[0] <= expected_x <= best_roi[2]
        assert best_roi[1] <= expected_y <= best_roi[3]

        # Verify that candidate ROIs NEVER touch frame edges or have centers within 6% of frame edges
        for roi in clipper.patch_rois:
            x1, y1, x2, y2 = roi
            # Not touching edges within 5px
            assert x1 > 5.0 and y1 > 5.0
            assert x2 < (640.0 - 5.0) and y2 < (640.0 - 5.0)

            # Center not within 6% of frame edge
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            assert 0.06 * 640.0 <= cx <= 0.94 * 640.0
            assert 0.06 * 640.0 <= cy <= 0.94 * 640.0

    def test_candidate_roi_border_rejection_rules(self):
        """_extract_candidate_rois masks out outer 6% and rejects ROIs near edges or with edge centers."""
        clipper = SpatialFeatureNormClipper(image_size=(640, 640))
        heatmap = np.zeros((640, 640), dtype=np.float32)

        # 1. Anomaly on outer 6% border (top-left corner, x=0..30, y=0..30)
        heatmap[0:30, 0:30] = 1.0

        # 2. Anomaly touching right edge (x2 >= 635)
        heatmap[200:260, 630:640] = 1.0

        # 3. Anomaly with center within 6% border (x=10..36, y=200..260 -> cx = 23 < 38.4)
        heatmap[200:260, 10:36] = 1.0

        # 4. Valid interior anomaly (x=280..360, y=200..260 -> cx = 320, cy = 230)
        heatmap[200:260, 280:360] = 1.0

        rois = clipper._extract_candidate_rois(heatmap, min_area=16.0)

        # Only the valid interior anomaly should survive
        assert len(rois) == 1
        x1, y1, x2, y2 = rois[0]
        assert x1 >= 270 and x2 <= 370
        assert y1 >= 190 and y2 <= 270

