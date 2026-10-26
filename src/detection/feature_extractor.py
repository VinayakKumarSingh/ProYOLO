"""Dual Visual and Temporal Feature Extraction Engine for ProYOLO.

Extracts temporal anomaly indicators (confidence drop, bounding box area/aspect ratio
stability, consecutive IoU consistency, flicker rate), visual perturbation signatures
(Laplacian gradient variance/entropy, HSV color dispersion), and spatial feature norm
clipping anomaly metrics (DPATCH grid monopolization ratio, JPEG recompression delta)
for tracked persons.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.detection.tracker import Track

logger = logging.getLogger(__name__)

# Complete schema of extracted feature names (15-dimensional vector)
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
    "max_grid_norm_ratio",
    "jpeg_recompression_delta",
]


class FeaturePacket(NamedTuple):
    """Encapsulates the extracted 1D feature vector, metadata, and dictionary breakdown."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    feature_vector: np.ndarray  # 1D float32 array of shape (15,)
    feature_dict: Dict[str, float]
    upper_body_roi: Optional[np.ndarray] = None
    candidate_patch_rois: Optional[List[List[float]]] = None
    delta_p_obj: float = 0.0


class SpatialFeatureNormClipper:
    """PyTorch forward hook to counter DPATCH RoI hijacking and localize physical adversarial patches.

    Attaches to the final backbone stages of a YOLO model (before the neck/SPPF).
    - Computes channel-wise L2 norm at each spatial grid cell: E_{i,j} = ||F_{:, i, j}||_2.
    - Calculates spatial mean (\\mu_E) and standard deviation (\\sigma_E) across H x W grid.
    - Clamps outlier activation spikes where E_{i,j} > \\tau (\\tau = \\mu_E + alpha * \\sigma_E, default alpha=2.0)
      by scaling the feature vector: F'_{:, i, j} = F_{:, i, j} * min(1.0, \\tau / (E_{i,j} + 1e-6)).
    - Stores the pre-clipped spatial anomaly heatmap (where E_{i,j} > \\tau) merged with a JPEG
      recompression difference map ||X - JPEG(X, quality=75)||_2^2 to output candidate `patch_rois`
      (list of [x1, y1, x2, y2] bounding boxes in image coordinates).
    """

    def __init__(
        self,
        alpha: float = 2.0,
        quality: int = 75,
        target_layer: Optional[Union[nn.Module, int, str]] = None,
        image_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.alpha = float(alpha)
        self.quality = int(quality)
        self.target_layer = target_layer
        self.image_size = image_size
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.current_image: Optional[np.ndarray] = None

        # State outputs
        self.patch_rois: List[List[float]] = []
        self.spatial_anomaly_heatmap: Optional[np.ndarray] = None
        self.jpeg_diff_map: Optional[np.ndarray] = None
        self.merged_heatmap: Optional[np.ndarray] = None
        self.max_grid_norm_ratio: float = 0.0
        self.jpeg_recompression_delta: float = 1.0
        self.last_tau: float = 0.0
        self.last_mu_E: float = 0.0
        self.last_sigma_E: float = 0.0

    def set_image(self, image: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """Cache the current image frame for JPEG recompression difference calculation."""
        if image is None:
            self.current_image = None
            return

        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
            self.current_image = arr
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    self.current_image = (image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    self.current_image = image.clip(0, 255).astype(np.uint8)
            else:
                self.current_image = image
        else:
            self.current_image = None

    @staticmethod
    def compute_jpeg_diff_map(image: np.ndarray, quality: int = 75) -> np.ndarray:
        """Compute normalized squared L2 difference map ||X - JPEG(X, quality)||_2^2."""
        if image is None or image.size == 0:
            return np.zeros((640, 640), dtype=np.float32)

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_u8 = image

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", img_u8, encode_param)
        if not success:
            return np.zeros(img_u8.shape[:2], dtype=np.float32)

        jpeg_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if jpeg_img is None:
            return np.zeros(img_u8.shape[:2], dtype=np.float32)

        diff = (img_u8.astype(np.float32) - jpeg_img.astype(np.float32)) ** 2
        diff_map = np.sum(diff, axis=-1)  # (H, W)
        max_d = float(np.max(diff_map))
        if max_d > 0:
            return (diff_map / max_d).astype(np.float32)
        return diff_map.astype(np.float32)

    @staticmethod
    def compute_jpeg_recompression_delta(
        image: np.ndarray,
        candidate_roi: Optional[Sequence[float]] = None,
        quality: int = 75,
    ) -> float:
        """Compute ratio of high-frequency energy lost under Q=75 JPEG compression inside candidate ROI vs background."""
        if image is None or image.size == 0:
            return 1.0

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_u8 = image

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", img_u8, encode_param)
        if not success:
            return 1.0

        jpeg_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if jpeg_img is None:
            return 1.0

        diff = np.sum((img_u8.astype(np.float32) - jpeg_img.astype(np.float32)) ** 2, axis=-1)
        h, w = diff.shape[:2]

        if candidate_roi is None or len(candidate_roi) < 4:
            return 1.0

        x1 = max(0, min(w - 1, int(round(candidate_roi[0]))))
        y1 = max(0, min(h - 1, int(round(candidate_roi[1]))))
        x2 = max(x1 + 1, min(w, int(round(candidate_roi[2]))))
        y2 = max(y1 + 1, min(h, int(round(candidate_roi[3]))))

        roi_diff = diff[y1:y2, x1:x2]
        if roi_diff.size == 0:
            return 1.0

        roi_energy = float(np.mean(roi_diff))

        bg_mask = np.ones((h, w), dtype=bool)
        bg_mask[y1:y2, x1:x2] = False
        bg_diff = diff[bg_mask]

        if bg_diff.size == 0:
            return 1.0

        bg_energy = float(np.mean(bg_diff))
        delta = roi_energy / (bg_energy + 1e-6)
        return float(np.clip(delta, 0.0, 100000.0))

    def _extract_candidate_rois(
        self,
        merged_heatmap: np.ndarray,
        min_area: float = 16.0,
    ) -> List[List[float]]:
        """Extract candidate patch ROIs [x1, y1, x2, y2] from merged heatmap."""
        if merged_heatmap is None or merged_heatmap.size == 0:
            return []

        max_val = float(np.max(merged_heatmap))
        if max_val <= 1e-4:
            return []

        norm_map = merged_heatmap / max_val
        thresh = (norm_map >= 0.25).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois: List[List[float]] = []

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                rois.append([float(x), float(y), float(x + w), float(y + h)])

        if not rois:
            ys, xs = np.where(norm_map >= 0.25)
            if len(xs) > 0:
                rois.append([
                    float(np.min(xs)),
                    float(np.min(ys)),
                    float(np.max(xs) + 1),
                    float(np.max(ys) + 1),
                ])

        rois.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        return rois

    def clip_features(
        self,
        F: torch.Tensor,
        image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply spatial feature norm clipping to feature map F of shape (B, C, H, W)."""
        if image is not None:
            self.set_image(image)

        is_3d = False
        if F.ndim == 3:
            is_3d = True
            F = F.unsqueeze(0)

        # 1. Channel-wise L2 norm at each spatial grid cell: E_{i,j} = ||F_{:, i, j}||_2
        E = torch.linalg.vector_norm(F, ord=2, dim=1)  # shape (B, H, W)

        # 2. Spatial mean (\mu_E) and standard deviation (\sigma_E) across H x W grid
        mu_E = torch.mean(E, dim=(-2, -1), keepdim=True)  # shape (B, 1, 1)
        sigma_E = torch.std(E, dim=(-2, -1), keepdim=True)  # shape (B, 1, 1)

        # 3. Threshold \tau = \mu_E + alpha * \sigma_E
        tau = mu_E + self.alpha * sigma_E  # shape (B, 1, 1)

        # Cache metrics
        E_det = E.detach()
        mu_det = mu_E.detach()
        sigma_det = sigma_E.detach()
        tau_det = tau.detach()

        self.last_mu_E = float(mu_det[0, 0, 0].item())
        self.last_sigma_E = float(sigma_det[0, 0, 0].item())
        self.last_tau = float(tau_det[0, 0, 0].item())

        # max_grid_norm_ratio: max(E_{i,j}) / (\mu_E + 1e-6) (captures DPATCH grid monopolization)
        self.max_grid_norm_ratio = float((torch.max(E_det) / (mu_det[0, 0, 0] + 1e-6)).item())

        # 4. Clamp outlier activation spikes where E_{i,j} > \tau by scaling feature vector:
        # F'_{:, i, j} = F_{:, i, j} * min(1.0, \tau / (E_{i,j} + 1e-6))
        scale = torch.clamp(tau / (E + 1e-6), max=1.0)  # shape (B, H, W)
        F_clipped = F * scale.unsqueeze(1)  # shape (B, C, H, W)

        # 5. Store pre-clipped spatial anomaly heatmap (where E_{i,j} > \tau)
        anomaly = torch.clamp(E_det[0] - tau_det[0, 0, 0], min=0.0).cpu().numpy()
        a_max = float(np.max(anomaly))
        if a_max > 0:
            self.spatial_anomaly_heatmap = (anomaly / a_max).astype(np.float32)
        else:
            self.spatial_anomaly_heatmap = np.zeros_like(anomaly, dtype=np.float32)

        # 6. Merge with JPEG recompression difference map ||X - JPEG(X, quality=75)||_2^2
        img = self.current_image
        if img is not None:
            img_h, img_w = img.shape[:2]
            jpeg_diff = self.compute_jpeg_diff_map(img, quality=self.quality)
        else:
            img_h, img_w = self.image_size
            jpeg_diff = np.ones((img_h, img_w), dtype=np.float32)

        self.jpeg_diff_map = jpeg_diff

        # Upsample spatial anomaly heatmap to image coordinates
        a_up = cv2.resize(self.spatial_anomaly_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        merged = a_up * (0.5 + 0.5 * jpeg_diff)
        m_max = float(np.max(merged))
        if m_max > 0:
            self.merged_heatmap = (merged / m_max).astype(np.float32)
        else:
            self.merged_heatmap = np.zeros_like(merged, dtype=np.float32)

        # 7. Output candidate patch_rois (list of [x1, y1, x2, y2] bounding boxes in image coordinates)
        if a_max > 0:
            self.patch_rois = self._extract_candidate_rois(self.merged_heatmap)
        else:
            self.patch_rois = []

        # 8. Compute jpeg_recompression_delta
        if img is not None and self.patch_rois:
            self.jpeg_recompression_delta = self.compute_jpeg_recompression_delta(
                img, candidate_roi=self.patch_rois[0], quality=self.quality
            )
        else:
            self.jpeg_recompression_delta = 1.0

        if is_3d:
            return F_clipped.squeeze(0)
        return F_clipped

    def forward_hook(self, module: nn.Module, inp: Any, output: Any) -> Any:
        """PyTorch forward hook execution callback."""
        if isinstance(output, torch.Tensor):
            return self.clip_features(output, image=self.current_image)
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            clipped = self.clip_features(output[0], image=self.current_image)
            return type(output)((clipped, *output[1:]))
        return output

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Support PyTorch forward hook or direct feature tensor call."""
        if len(args) == 3 and isinstance(args[0], nn.Module):
            return self.forward_hook(args[0], args[1], args[2])
        elif len(args) >= 1 and isinstance(args[0], torch.Tensor):
            img = kwargs.get("image", args[1] if len(args) > 1 else None)
            return self.clip_features(args[0], image=img)
        raise TypeError(f"Invalid arguments for SpatialFeatureNormClipper: {args}")

    @staticmethod
    def _find_backbone_stage(model: Any) -> Optional[nn.Module]:
        """Locate final backbone stage before SPPF in YOLO or PyTorch model."""
        m = model
        if hasattr(m, "model"):
            m = m.model
        if hasattr(m, "model"):
            m = m.model

        children = list(m.named_children())
        for idx, (name, child) in enumerate(children):
            if "sppf" in type(child).__name__.lower() or "sppf" in name.lower():
                if idx > 0:
                    return children[idx - 1][1]

        if hasattr(m, "__getitem__") and len(m) >= 9:
            return m[8]

        if isinstance(model, nn.Module):
            return model
        return None

    def attach(self, model: Any, layer: Optional[Any] = None) -> SpatialFeatureNormClipper:
        """Attach forward hook to the final backbone stage (before neck/SPPF)."""
        target = layer
        if target is None:
            target = self._find_backbone_stage(model)
        elif isinstance(target, int):
            m = model
            if hasattr(m, "model"):
                m = m.model
            if hasattr(m, "model"):
                m = m.model
            target = m[target]

        if target is None or not isinstance(target, nn.Module):
            raise ValueError(f"Could not resolve target layer for attaching hook: {target}")

        self.detach()
        self.hook_handle = target.register_forward_hook(self)
        self.target_layer = target
        logger.info("SpatialFeatureNormClipper attached to %s", type(target).__name__)
        return self

    def detach(self) -> None:
        """Remove registered forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def remove(self) -> None:
        """Alias for detach."""
        self.detach()

    def __enter__(self) -> SpatialFeatureNormClipper:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.detach()


class DualFeatureExtractor:
    """Computes fused temporal and visual features for physical patch attack detection.

    Attributes:
        torso_vertical_ratio: Tuple (top_ratio, bottom_ratio) for upper body cropping.
        torso_horizontal_inset: Percentage inset from left and right edges.
        clipper: Optional attached SpatialFeatureNormClipper instance.
    """

    def __init__(
        self,
        torso_vertical_ratio: Tuple[float, float] = (0.15, 0.65),
        torso_horizontal_inset: float = 0.10,
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> None:
        self.torso_top_ratio = torso_vertical_ratio[0]
        self.torso_bottom_ratio = torso_vertical_ratio[1]
        self.torso_horizontal_inset = torso_horizontal_inset
        self.clipper = clipper

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
    # DPATCH & JPEG Feature Calculations
    # =========================================================================

    @staticmethod
    def compute_grid_norm_ratio(roi: np.ndarray, grid_size: Tuple[int, int] = (8, 8)) -> float:
        """Compute max spatial grid cell energy to mean grid cell energy ratio over ROI."""
        if roi is None or roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return 1.0

        h, w = roi.shape[:2]
        gh, gw = grid_size
        cell_h = h // gh
        cell_w = w // gw
        if cell_h == 0 or cell_w == 0:
            return 1.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        cell_norms = []
        for i in range(gh):
            for j in range(gw):
                cell = lap[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                cell_norms.append(float(np.linalg.norm(cell)))

        norms = np.array(cell_norms, dtype=np.float32)
        mean_norm = float(np.mean(norms))
        max_norm = float(np.max(norms))
        return float(max_norm / (mean_norm + 1e-6))

    @staticmethod
    def compute_jpeg_recompression_delta(
        frame: np.ndarray,
        candidate_roi: Optional[Sequence[float]] = None,
        quality: int = 75,
    ) -> float:
        """Compute ratio of high-frequency energy lost under Q=75 JPEG compression inside candidate ROI vs background."""
        return SpatialFeatureNormClipper.compute_jpeg_recompression_delta(
            frame, candidate_roi=candidate_roi, quality=quality
        )

    # =========================================================================
    # Extraction Pipeline
    # =========================================================================

    def extract_single(
        self,
        frame: np.ndarray,
        track: Track,
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> FeaturePacket:
        """Extract fused 1D feature vector for a single tracked person."""
        bbox = track.latest_bbox
        roi = self.crop_upper_body(frame, bbox)

        temporal_feats = self.compute_temporal_features(track)
        visual_feats = self.compute_visual_features(roi)

        active_clipper = clipper or self.clipper

        candidate_patch_rois: List[List[float]] = []
        if active_clipper is not None and active_clipper.max_grid_norm_ratio > 0:
            max_grid_norm_ratio = active_clipper.max_grid_norm_ratio
            candidate_patch_rois = list(active_clipper.patch_rois)
            roi_for_jpeg = candidate_patch_rois[0] if candidate_patch_rois else bbox
            jpeg_recompression_delta = self.compute_jpeg_recompression_delta(
                frame, candidate_roi=roi_for_jpeg
            )
        else:
            max_grid_norm_ratio = self.compute_grid_norm_ratio(roi)
            jpeg_recompression_delta = self.compute_jpeg_recompression_delta(
                frame, candidate_roi=bbox
            )

        dpatch_feats: Dict[str, float] = {
            "max_grid_norm_ratio": float(max_grid_norm_ratio),
            "jpeg_recompression_delta": float(jpeg_recompression_delta),
        }

        fused_dict: Dict[str, float] = {}
        fused_dict.update(temporal_feats)
        fused_dict.update(visual_feats)
        fused_dict.update(dpatch_feats)

        # Assemble fixed-order 1D float32 vector matching FEATURE_NAMES
        vector = np.array(
            [fused_dict.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float32,
        )

        # Sanitize against any NaN or Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)

        delta_p_obj = float(getattr(track, "delta_p_obj", 0.0))
        return FeaturePacket(
            track_id=track.track_id,
            bbox=bbox,
            feature_vector=vector,
            feature_dict=fused_dict,
            upper_body_roi=roi,
            candidate_patch_rois=candidate_patch_rois,
            delta_p_obj=delta_p_obj,
        )

    def extract_batch(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> List[FeaturePacket]:
        """Extract features for all active tracks in the current frame."""
        return [self.extract_single(frame, trk, clipper=clipper) for trk in tracks]

    def extract_feature_matrix(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> np.ndarray:
        """Extract 2D feature matrix of shape (num_tracks, 15) for fast tabular model inference."""
        packets = self.extract_batch(frame, tracks, clipper=clipper)
        if not packets:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.vstack([p.feature_vector for p in packets])
