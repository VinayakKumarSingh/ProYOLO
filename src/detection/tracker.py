"""Lightweight Multi-Object ByteTrack Tracker for ProYOLO.

Pure-NumPy and SciPy implementation of ByteTrack with Kalman filtering and
consecutive frame track history maintenance (last N=10 frames) for temporal
anomaly and flicker analysis.
"""

from __future__ import annotations

import enum
import time
from collections import deque
from typing import Deque, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.detector import Detection


class TrackState(enum.Enum):
    """Lifecycle state of an individual tracked target."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackRecord(NamedTuple):
    """Historical snapshot of a track state at a specific frame index."""

    frame_idx: int
    bbox: np.ndarray  # Format: [x1, y1, x2, y2]
    confidence: float
    area: float
    aspect_ratio: float
    timestamp: float


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    inter_x1 = max(float(box1[0]), float(box2[0]))
    inter_y1 = max(float(box1[1]), float(box2[1]))
    inter_x2 = min(float(box1[2]), float(box2[2]))
    inter_y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    if union <= 1e-6:
        return 0.0
    return inter_area / union


def iou_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU cost matrix (1.0 - IoU) for Hungarian matching."""
    num_a = len(boxes_a)
    num_b = len(boxes_b)
    if num_a == 0 or num_b == 0:
        return np.zeros((num_a, num_b), dtype=np.float32)

    cost_mat = np.ones((num_a, num_b), dtype=np.float32)
    for i in range(num_a):
        for j in range(num_b):
            cost_mat[i, j] = 1.0 - bbox_iou(boxes_a[i], boxes_b[j])
    return cost_mat


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self, bbox: np.ndarray) -> None:
        # State vector: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]
        self.dim_x = 8
        self.dim_z = 4

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # State transition matrix F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        # Process and measurement covariances
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty for initial velocity
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 1.0
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(self.dim_z, dtype=np.float32) * 1.0

        # Initialize state from bbox
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self._set_state_from_bbox(bbox)

    def _set_state_from_bbox(self, bbox: np.ndarray) -> None:
        w = max(1e-4, float(bbox[2] - bbox[0]))
        h = max(1e-4, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0
        ar = w / h
        self.x[:4, 0] = [cx, cy, ar, h]

    def predict(self) -> np.ndarray:
        """Advance state estimation by one time step."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_bbox()

    def update(self, bbox: np.ndarray) -> None:
        """Correct state estimate with new measurement."""
        w = max(1e-4, float(bbox[2] - bbox[0]))
        h = max(1e-4, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0
        ar = w / h
        z = np.array([[cx], [cy], [ar], [h]], dtype=np.float32)

        # Innovation
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_bbox(self) -> np.ndarray:
        """Convert current state back to [x1, y1, x2, y2]."""
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        ar = max(1e-4, float(self.x[2, 0]))
        h = max(1.0, float(self.x[3, 0]))
        w = max(1.0, ar * h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class Track:
    """Represents a continuous track trajectory with history tracking."""

    _count = 0

    def __init__(
        self,
        detection: Detection,
        frame_idx: int,
        max_history: int = 10,
    ) -> None:
        Track._count += 1
        self.track_id = Track._count
        self.kf = KalmanBoxTracker(detection.bbox)
        self.max_history = max_history

        self.state = TrackState.New
        self.is_activated = False
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

        self.delta_p_obj: float = float(getattr(detection, "delta_p_obj", 0.0))
        self.recovered_by_inpainting: bool = bool(getattr(detection, "recovered_by_inpainting", False))

        # Rolling deque (maxlen=5) for attack predictions & alert confirmation (K=3/N=5)
        self.attack_history: Deque[bool] = deque(maxlen=5)
        self._confirmed_attack_alert: Optional[bool] = None

        # Track history deque: last N frames
        self.history: Deque[TrackRecord] = deque(maxlen=max_history)
        self._record_history(detection.bbox, detection.confidence, frame_idx)

    def _record_history(self, bbox: np.ndarray, confidence: float, frame_idx: int) -> None:
        w = max(0.0, float(bbox[2] - bbox[0]))
        h = max(0.0, float(bbox[3] - bbox[1]))
        ar = w / max(1e-4, h)
        area = w * h
        record = TrackRecord(
            frame_idx=frame_idx,
            bbox=bbox.copy(),
            confidence=float(confidence),
            area=area,
            aspect_ratio=ar,
            timestamp=time.time(),
        )
        self.history.append(record)

    def predict(self) -> np.ndarray:
        """Predict bounding box for next time step."""
        predicted_box = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return predicted_box

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Update track state with a matched detection and record Kalman-smoothed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.delta_p_obj = float(getattr(detection, "delta_p_obj", 0.0))
        self.recovered_by_inpainting = bool(getattr(detection, "recovered_by_inpainting", False))
        self.kf.update(detection.bbox)
        # Use Kalman filter state estimate to produce smoothed bounding box resistant to patch suppression
        smoothed_box = self.kf.get_bbox()
        self._record_history(smoothed_box, detection.confidence, frame_idx)

    def record_attack_prediction(self, is_attack: bool) -> bool:
        """Record an attack prediction in rolling history (maxlen=5) and evaluate K=3 persistence.

        Only sets confirmed_attack_alert = True when is_attack == True in at least
        K=3 out of N=5 consecutive frames.
        """
        self.attack_history.append(bool(is_attack))
        self._confirmed_attack_alert = bool(sum(self.attack_history) >= 3)
        return self._confirmed_attack_alert

    @property
    def confirmed_attack_alert(self) -> bool:
        """True when the track registers is_attack in at least K=3 out of N=5 consecutive frames."""
        if self._confirmed_attack_alert is not None:
            return self._confirmed_attack_alert
        return bool(sum(self.attack_history) >= 3)

    @confirmed_attack_alert.setter
    def confirmed_attack_alert(self, value: bool) -> None:
        self._confirmed_attack_alert = bool(value)

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark track as permanently terminated."""
        self.state = TrackState.Removed

    # =========================================================================
    # Temporal Metrics & History Properties
    # =========================================================================

    @property
    def latest_bbox(self) -> np.ndarray:
        """Most recent bounding box [x1, y1, x2, y2]."""
        if self.history:
            return self.history[-1].bbox
        return self.kf.get_bbox()

    @property
    def latest_confidence(self) -> float:
        """Confidence score from the most recent detection."""
        if self.history:
            return self.history[-1].confidence
        return 0.0

    @property
    def mean_confidence(self) -> float:
        """Average confidence score across the history window."""
        if not self.history:
            return 0.0
        return float(np.mean([r.confidence for r in self.history]))

    @property
    def confidence_drop(self) -> float:
        """Difference between prior mean confidence and current frame confidence."""
        if len(self.history) < 2:
            return 0.0
        prior_mean = float(np.mean([r.confidence for r in list(self.history)[:-1]]))
        current = self.history[-1].confidence
        return max(0.0, prior_mean - current)

    @property
    def consecutive_ious(self) -> List[float]:
        """Pairwise IoU values between consecutive frames in history."""
        if len(self.history) < 2:
            return [1.0]
        records = list(self.history)
        ious = [bbox_iou(records[i - 1].bbox, records[i].bbox) for i in range(1, len(records))]
        return ious

    @property
    def area_variance(self) -> float:
        """Relative variance of bounding box area across history window."""
        if len(self.history) < 2:
            return 0.0
        areas = np.array([r.area for r in self.history], dtype=np.float32)
        mean_area = np.mean(areas)
        if mean_area <= 1e-4:
            return 0.0
        return float(np.std(areas) / mean_area)

    @property
    def aspect_ratio_variance(self) -> float:
        """Standard deviation of aspect ratio across history window."""
        if len(self.history) < 2:
            return 0.0
        ars = [r.aspect_ratio for r in self.history]
        return float(np.std(ars))

    @property
    def frame_drop_count(self) -> int:
        """Number of frames dropped/skipped within the history span (flicker indicator)."""
        if len(self.history) < 2:
            return 0
        records = list(self.history)
        expected_span = records[-1].frame_idx - records[0].frame_idx + 1
        actual_count = len(records)
        return max(0, expected_span - actual_count)


class ByteTracker:
    """CPU-friendly ByteTrack Multi-Object Tracker.

    Implements dual-threshold association to maintain track continuity
    during physical patch occlusion and objectness suppression attacks:
    - Tier 1: Associates high-confidence person detections (conf >= 0.40, including Pass 2 recovered boxes).
    - Tier 2: Associates unmatched tracklets with low-confidence detections (0.12 <= conf < 0.40)
      using Kalman filter state predictions, keeping partially suppressed persons alive with smoothed boxes.
    - Temporal persistence: Enforces K=3 out of N=5 consecutive frame alert confirmation.
    """

    def __init__(
        self,
        track_thresh: float = 0.40,
        low_thresh: float = 0.12,
        match_thresh: float = 0.7,
        track_buffer: int = 30,
        max_history: int = 10,
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.max_history = max_history

        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.frame_idx = 0

        # Rolling deques (maxlen=5) for attack predictions & alert confirmation (K=3/N=5)
        self.attack_history: Deque[bool] = deque(maxlen=5)
        self.dpatch_attack_history: Deque[bool] = deque(maxlen=5)
        self._confirmed_dpatch_alert: Optional[bool] = None

    def record_dpatch_attack(self, is_attack: bool) -> bool:
        """Record a global background DPATCH candidate alert in rolling deque (maxlen=5)."""
        self.dpatch_attack_history.append(bool(is_attack))
        self.attack_history.append(bool(is_attack))
        self._confirmed_dpatch_alert = bool(sum(self.dpatch_attack_history) >= 3)
        return self._confirmed_dpatch_alert

    @property
    def confirmed_attack_alert(self) -> bool:
        """True when any active track or the global DPATCH deque has K=3/5 confirmed alerts."""
        if self._confirmed_dpatch_alert:
            return True
        return bool(
            sum(self.dpatch_attack_history) >= 3
            or any(t.confirmed_attack_alert for t in self.tracked_tracks)
        )

    def update(
        self,
        detections: List[Detection],
        frame_idx: Optional[int] = None,
    ) -> List[Track]:
        """Update tracker state with new frame detections.

        Args:
            detections: List of Detection instances from the detector.
            frame_idx: Optional explicit frame index counter.

        Returns:
            List of active Track instances in the current frame.
        """
        if frame_idx is not None:
            self.frame_idx = frame_idx
        else:
            self.frame_idx += 1

        # 1. Dual-threshold partitioning:
        # High confidence (conf >= 0.40) and low confidence (0.12 <= conf < 0.40)
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []
        for det in detections:
            if det.confidence >= self.track_thresh:
                high_dets.append(det)
            elif det.confidence >= self.low_thresh:
                low_dets.append(det)

        # 2. Advance Kalman Filter state predictions for all existing tracks
        for trk in self.tracked_tracks:
            trk.predict()
        for trk in self.lost_tracks:
            trk.predict()

        # 3. First Association: Match active tracks with HIGH confidence detections
        # (Includes Pass 2 actively recovered person boxes)
        pool_tracks = self.tracked_tracks + self.lost_tracks
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            pool_tracks, high_dets, self.match_thresh
        )

        for trk, det in matched_tracks_1:
            trk.update(det, self.frame_idx)
            if trk in self.lost_tracks:
                self.lost_tracks.remove(trk)
            if trk not in self.tracked_tracks:
                self.tracked_tracks.append(trk)

        # 4. Second Association: Match remaining tracks with LOW confidence detections (ByteTrack)
        # using Kalman filter state predictions to keep partially suppressed persons (0.12 <= conf < 0.40)
        # alive during adversarial patch suppression or pose changes, outputting smoothed bounding boxes.
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            unmatched_tracks_1, low_dets, self.match_thresh
        )

        for trk, det in matched_tracks_2:
            trk.update(det, self.frame_idx)
            if trk in self.lost_tracks:
                self.lost_tracks.remove(trk)
            if trk not in self.tracked_tracks:
                self.tracked_tracks.append(trk)

        # 5. Handle remaining unmatched tracks
        for trk in unmatched_tracks_2:
            if trk in self.tracked_tracks:
                self.tracked_tracks.remove(trk)
                self.lost_tracks.append(trk)
                trk.mark_missed()

        # 6. Initialize new tracks from unmatched high confidence detections
        for det in unmatched_dets_1:
            new_trk = Track(det, self.frame_idx, max_history=self.max_history)
            new_trk.state = TrackState.Tracked
            new_trk.is_activated = True
            self.tracked_tracks.append(new_trk)

        # 7. Remove stale lost tracks that exceed track_buffer
        active_lost: List[Track] = []
        for trk in self.lost_tracks:
            if self.frame_idx - trk.history[-1].frame_idx <= self.track_buffer:
                active_lost.append(trk)
            else:
                trk.mark_removed()
        self.lost_tracks = active_lost

        return [t for t in self.tracked_tracks if t.is_activated and t.state == TrackState.Tracked]

    @staticmethod
    def _associate(
        tracks: List[Track],
        detections: List[Detection],
        threshold: float,
    ) -> Tuple[List[Tuple[Track, Detection]], List[Track], List[Detection]]:
        """Perform bipartite matching using Hungarian algorithm with IoU distance on Kalman state predictions."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks.copy(), detections.copy()

        # Use Kalman filter state predictions for robust spatial association
        track_boxes = np.array([t.kf.get_bbox() for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)

        cost_mat = iou_distance_matrix(track_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matched: List[Tuple[Track, Detection]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] <= threshold:
                matched.append((tracks[r], detections[c]))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        remaining_tracks = [tracks[i] for i in unmatched_tracks]
        remaining_dets = [detections[j] for j in unmatched_dets]
        return matched, remaining_tracks, remaining_dets

    def reset(self) -> None:
        """Reset all tracker state."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_idx = 0
