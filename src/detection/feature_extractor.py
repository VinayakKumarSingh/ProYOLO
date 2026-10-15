"""Dual Visual and Temporal Feature Extraction Engine for ProYOLO.

Extracts temporal anomaly indicators (confidence drop, bounding box area/aspect ratio
stability, consecutive IoU consistency, flicker rate) and visual perturbation signatures
(Laplacian gradient variance/entropy, HSV color dispersion) for every tracked person.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from src.detection.tracker import Track

logger = logging.getLogger(__name__)

# Complete schema of extracted feature names (13-dimensional vector)
FEATURE_NAMES: List[str] = [
    "conf_current",
    "conf_drop",
    "conf_std",
    "area_stability",
    "aspect_ratio_stability",
    "mean_consecutive_iou",
    "min_consecutive_iou",
    "flicker_rate",
    "laplacian_var_norm",
    "laplacian_entropy",
    "color_hue_entropy",
    "color_sat_entropy",
    "color_val_entropy",
]


class FeaturePacket(NamedTuple):
    """Encapsulates the extracted 1D feature vector, metadata, and dictionary breakdown."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    feature_vector: np.ndarray  # 1D float32 array of shape (13,)
    feature_dict: Dict[str, float]
    upper_body_roi: Optional[np.ndarray] = None


class DualFeatureExtractor:
    """Computes fused temporal and visual features for physical patch attack detection.

    Attributes:
        torso_vertical_ratio: Tuple (top_ratio, bottom_ratio) for upper body cropping.
        torso_horizontal_inset: Percentage inset from left and right edges.
    """

    def __init__(
        self,
        torso_vertical_ratio: Tuple[float, float] = (0.15, 0.65),
        torso_horizontal_inset: float = 0.10,
    ) -> None:
        self.torso_top_ratio = torso_vertical_ratio[0]
        self.torso_bottom_ratio = torso_vertical_ratio[1]
        self.torso_horizontal_inset = torso_horizontal_inset

    def crop_upper_body(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop the upper body / chest region of a detected person silhouette."""
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))

        # Upper body region: 15% to 65% down from head, 10% inset from left/right
        ty1 = max(0, int(round(y1 + self.torso_top_ratio * bh)))
        ty2 = min(h_frame, int(round(y1 + self.torso_bottom_ratio * bh)))
        tx1 = max(0, int(round(x1 + self.torso_horizontal_inset * bw)))
        tx2 = min(w_frame, int(round(x2 - self.torso_horizontal_inset * bw)))

        if ty2 > ty1 and tx2 > tx1:
            return frame[ty1:ty2, tx1:tx2]

        # Fallback to bounded full box if torso crop is collapsed
        by1 = max(0, min(h_frame - 1, int(y1)))
        by2 = max(by1 + 1, min(h_frame, int(y2)))
        bx1 = max(0, min(w_frame - 1, int(x1)))
        bx2 = max(bx1 + 1, min(w_frame, int(x2)))
        return frame[by1:by2, bx1:bx2]

    # =========================================================================
    # Temporal Anomaly Feature Calculations
    # =========================================================================

    @staticmethod
    def compute_temporal_features(track: Track) -> Dict[str, float]:
        """Compute sliding-window temporal stability metrics from track history."""
        history = list(track.history)
        conf_current = track.latest_confidence

        if len(history) < 2:
            return {
                "conf_current": float(np.clip(conf_current, 0.0, 1.0)),
                "conf_drop": 0.0,
                "conf_std": 0.0,
                "area_stability": 0.0,
                "aspect_ratio_stability": 0.0,
                "mean_consecutive_iou": 1.0,
                "min_consecutive_iou": 1.0,
                "flicker_rate": 0.0,
            }

        # 1. Confidence metrics
        confs = [r.confidence for r in history]
        prior_mean_conf = float(np.mean(confs[:-1]))
        conf_drop = float(max(0.0, prior_mean_conf - conf_current))
        conf_std = float(np.std(confs))

        # 2. Area stability: relative standard deviation (coefficient of variation)
        areas = np.array([r.area for r in history], dtype=np.float32)
        mean_area = float(np.mean(areas))
        area_stability = float(np.std(areas) / (mean_area + 1e-4)) if mean_area > 0 else 0.0
        area_stability = float(np.clip(area_stability, 0.0, 2.0))

        # 3. Aspect ratio stability
        ars = [r.aspect_ratio for r in history]
        aspect_ratio_stability = float(np.clip(np.std(ars), 0.0, 1.0))

        # 4. Consecutive IoU consistency
        ious = track.consecutive_ious
        mean_iou = float(np.clip(np.mean(ious), 0.0, 1.0)) if ious else 1.0
        min_iou = float(np.clip(np.min(ious), 0.0, 1.0)) if ious else 1.0

        # 5. Flicker rate (missed detection frame drops in sliding window)
        expected_span = max(1, history[-1].frame_idx - history[0].frame_idx + 1)
        actual_count = len(history)
        frame_drops = max(0, expected_span - actual_count)
        flicker_rate = float(np.clip(frame_drops / expected_span, 0.0, 1.0))

        return {
            "conf_current": float(np.clip(conf_current, 0.0, 1.0)),
            "conf_drop": float(np.clip(conf_drop, 0.0, 1.0)),
            "conf_std": float(np.clip(conf_std, 0.0, 1.0)),
            "area_stability": area_stability,
            "aspect_ratio_stability": aspect_ratio_stability,
            "mean_consecutive_iou": mean_iou,
            "min_consecutive_iou": min_iou,
            "flicker_rate": flicker_rate,
        }

    # =========================================================================
    # Visual Feature Calculations (Spatial Gradient & Color Dispersion)
    # =========================================================================

    @staticmethod
    def compute_visual_features(roi: np.ndarray) -> Dict[str, float]:
        """Compute Laplacian gradient entropy and HSV color dispersion on upper body ROI."""
        if roi is None or roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return {
                "laplacian_var_norm": 0.0,
                "laplacian_entropy": 0.0,
                "color_hue_entropy": 0.0,
                "color_sat_entropy": 0.0,
                "color_val_entropy": 0.0,
            }

        # Downscale large ROI to max 128x128 for sub-millisecond CPU execution
        h_roi, w_roi = roi.shape[:2]
        if h_roi > 128 or w_roi > 128:
            scale = 128.0 / max(h_roi, w_roi)
            roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 1. Laplacian gradient variance and entropy
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(np.var(laplacian))

        # Log-normalization of variance into [0.0, 1.0] (typical var ranges 0..10000)
        lap_var_norm = float(np.clip(np.log10(1.0 + lap_var) / np.log10(10001.0), 0.0, 1.0))

        # Laplacian gradient magnitude entropy
        abs_lap = np.abs(laplacian)
        lap_hist, _ = np.histogram(abs_lap, bins=32, range=(0.0, 256.0))
        lap_p = lap_hist.astype(np.float32) / float(max(1.0, np.sum(lap_hist)))
        lap_entropy = float(
            -np.sum(lap_p * np.log2(lap_p + 1e-12)) / np.log2(32.0)
        )
        lap_entropy = float(np.clip(lap_entropy, 0.0, 1.0))

        # 2. Color dispersion in HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Hue distribution (16 bins, range 0..180 in OpenCV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        h_p = h_hist / float(max(1.0, np.sum(h_hist)))
        h_entropy = float(-np.sum(h_p * np.log2(h_p + 1e-12)) / np.log2(16.0))

        # Saturation distribution (8 bins, range 0..256)
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        s_p = s_hist / float(max(1.0, np.sum(s_hist)))
        s_entropy = float(-np.sum(s_p * np.log2(s_p + 1e-12)) / np.log2(8.0))

        # Value distribution (8 bins, range 0..256)
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        v_p = v_hist / float(max(1.0, np.sum(v_hist)))
        v_entropy = float(-np.sum(v_p * np.log2(v_p + 1e-12)) / np.log2(8.0))

        return {
            "laplacian_var_norm": lap_var_norm,
            "laplacian_entropy": lap_entropy,
            "color_hue_entropy": float(np.clip(h_entropy, 0.0, 1.0)),
            "color_sat_entropy": float(np.clip(s_entropy, 0.0, 1.0)),
            "color_val_entropy": float(np.clip(v_entropy, 0.0, 1.0)),
        }

    # =========================================================================
    # Extraction Pipeline
    # =========================================================================

    def extract_single(self, frame: np.ndarray, track: Track) -> FeaturePacket:
        """Extract fused 1D feature vector for a single tracked person."""
        bbox = track.latest_bbox
        roi = self.crop_upper_body(frame, bbox)

        temporal_feats = self.compute_temporal_features(track)
        visual_feats = self.compute_visual_features(roi)

        fused_dict: Dict[str, float] = {}
        fused_dict.update(temporal_feats)
        fused_dict.update(visual_feats)

        # Assemble fixed-order 1D float32 vector matching FEATURE_NAMES
        vector = np.array(
            [fused_dict.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float32,
        )

        # Sanitize against any NaN or Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)

        return FeaturePacket(
            track_id=track.track_id,
            bbox=bbox,
            feature_vector=vector,
            feature_dict=fused_dict,
            upper_body_roi=roi,
        )

    def extract_batch(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> List[FeaturePacket]:
        """Extract features for all active tracks in the current frame."""
        return [self.extract_single(frame, trk) for trk in tracks]

    def extract_feature_matrix(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> np.ndarray:
        """Extract 2D feature matrix of shape (num_tracks, 13) for fast tabular model inference."""
        packets = self.extract_batch(frame, tracks)
        if not packets:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.vstack([p.feature_vector for p in packets])
