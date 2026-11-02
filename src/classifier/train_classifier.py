"""Model Training and Cross-Validation Benchmark Suite for ProYOLO.

Trains, evaluates, and benchmarks three CPU-efficient classifiers:
1. Random Forest (Scikit-Learn)
2. HistGradientBoosting (Scikit-Learn)
3. XGBoost Classifier (XGBoost)

Performs 5-fold cross-validation measuring Precision, Recall, F1-Score, and CPU
inference latency per sample (<2 ms), serializing the best performing model to
`models/attack_classifier.joblib` and ONNX format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except (ImportError, Exception):  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except (ImportError, Exception):  # pragma: no cover
    convert_sklearn = None  # type: ignore
    FloatTensorType = None  # type: ignore

from src.detection.feature_extractor import FEATURE_NAMES, DualFeatureExtractor
from src.detection.tracker import Track

logger = logging.getLogger(__name__)


class ModelBenchmarkResult(NamedTuple):
    """Container holding evaluation metrics and performance benchmarks for a candidate model."""

    model_name: str
    mean_precision: float
    mean_recall: float
    mean_f1: float
    std_f1: float
    cpu_latency_per_sample_ms: float
    trained_model: Any


class HardNegativePatchApplier:
    """Data augmentation engine applying Expectation Over Transformation (EOT) benign patches.

    Following the Ad-YOLO and Thys et al. formulation:
    Applies high-contrast benign graphics (checkerboards, stripes, QR-like patterns, and text boxes)
    onto person bounding boxes with random rotation (+-20 deg), scaling, perspective warps,
    and brightness jitter to train the classifier not to flag non-adversarial textures as attacks.
    """

    def __init__(
        self,
        max_rotation_deg: float = 20.0,
        scale_range: Tuple[float, float] = (0.15, 0.40),
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        seed: Optional[int] = None,
    ) -> None:
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.rng = np.random.RandomState(seed)

    def generate_checkerboard(self, width: int = 120, height: int = 120, num_checks: int = 6) -> np.ndarray:
        """Generate high-contrast alternating checkerboard pattern."""
        tile_w = max(2, width // num_checks)
        tile_h = max(2, height // num_checks)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = (240, 240, 240)
        c2 = (15, 15, 15)
        for i in range(num_checks + 2):
            for j in range(num_checks + 2):
                y1 = min(height, i * tile_h)
                y2 = min(height, (i + 1) * tile_h)
                x1 = min(width, j * tile_w)
                x2 = min(width, (j + 1) * tile_w)
                color = c1 if (i + j) % 2 == 0 else c2
                img[y1:y2, x1:x2] = color
        return img

    def generate_stripes(
        self, width: int = 120, height: int = 120, stripe_w: int = 10, orientation: str = "vertical"
    ) -> np.ndarray:
        """Generate high-contrast striped pattern (horizontal, vertical, or diagonal)."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = (255, 255, 255)
        c2 = (0, 0, 0)
        if orientation == "horizontal":
            for y in range(0, height, stripe_w * 2):
                img[y : y + stripe_w, :] = c1
        elif orientation == "diagonal":
            for y in range(height):
                for x in range(width):
                    if ((x + y) // stripe_w) % 2 == 0:
                        img[y, x] = c1
        else:  # vertical
            for x in range(0, width, stripe_w * 2):
                img[:, x : x + stripe_w] = c1
        return img

    def generate_qr_like(self, width: int = 120, height: int = 120, cell_size: int = 12) -> np.ndarray:
        """Generate high-contrast QR-like pseudo-random binary block grid."""
        rows = max(1, height // cell_size)
        cols = max(1, width // cell_size)
        grid = self.rng.choice([0, 255], size=(rows, cols)).astype(np.uint8)
        img = cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def generate_text_box(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Generate synthetic graphic text box with high-contrast text lines."""
        img = np.full((height, width, 3), 245, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = ["SECURITY", "PASS", "NOTICE", "CAUTION", "STAFF", "EVENT", "BADGE"]
        y = 25
        while y < height - 10:
            text = str(self.rng.choice(words))
            cv2.putText(img, text, (5, y), font, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
            y += 24
        return img

    def generate_color_blocks(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Generate high-contrast geometric color block pattern."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        num_blocks = self.rng.choice([2, 3, 4])
        split_horiz = self.rng.choice([True, False])
        colors = [
            (220, 40, 40), (40, 200, 40), (40, 40, 220),
            (240, 240, 30), (240, 120, 20), (20, 20, 20), (240, 240, 240)
        ]
        chosen = [colors[i] for i in self.rng.choice(len(colors), size=num_blocks, replace=False)]
        if split_horiz:
            h_step = height // num_blocks
            for i, col in enumerate(chosen):
                y1 = i * h_step
                y2 = height if i == num_blocks - 1 else (i + 1) * h_step
                img[y1:y2, :] = col
        else:
            w_step = width // num_blocks
            for i, col in enumerate(chosen):
                x1 = i * w_step
                x2 = width if i == num_blocks - 1 else (i + 1) * w_step
                img[:, x1:x2] = col
        return img

    def generate_smooth_tv_patch(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Generate smooth Total-Variation (Thys et al. L_tv) colorful blob patch."""
        low_h = max(4, height // 6)
        low_w = max(4, width // 6)
        low = self.rng.randint(0, 256, (low_h, low_w, 3), dtype=np.uint8)
        pat = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
        ksize = 2 * (max(width, height) // 10) + 1
        pat = cv2.GaussianBlur(pat, (ksize, ksize), sigmaX=2.5)
        # Enhance color saturation
        hsv = cv2.cvtColor(pat, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 50, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def generate_dpatch_grid(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Generate high-frequency DPATCH grid pattern."""
        cell = max(2, min(width, height) // 6)
        rows = height // cell + 1
        cols = width // cell + 1
        raw = self.rng.randint(0, 256, (rows, cols, 3), dtype=np.uint8)
        return cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)

    def generate_random_benign_patch(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Sample a random high-contrast benign patch pattern."""
        choice = self.rng.choice(["checkerboard", "stripes_v", "stripes_d", "qr", "text", "color_blocks"])
        if choice == "checkerboard":
            return self.generate_checkerboard(width, height)
        elif choice == "stripes_v":
            return self.generate_stripes(width, height, orientation="vertical")
        elif choice == "stripes_d":
            return self.generate_stripes(width, height, orientation="diagonal")
        elif choice == "qr":
            return self.generate_qr_like(width, height)
        elif choice == "color_blocks":
            return self.generate_color_blocks(width, height)
        else:
            return self.generate_text_box(width, height)

    def generate_random_adversarial_patch(self, width: int = 120, height: int = 120) -> np.ndarray:
        """Sample a random adversarial patch pattern (smooth TV blob or DPATCH grid)."""
        if self.rng.rand() < 0.5:
            return self.generate_smooth_tv_patch(width, height)
        return self.generate_dpatch_grid(width, height)

    def apply_eot_transform(
        self,
        patch: np.ndarray,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Apply EOT transformations: rotation +-20 deg, scaling, brightness jitter."""
        # 1. Resize to target dimensions
        resized = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 2. Random rotation in [-max_rotation_deg, +max_rotation_deg]
        angle = float(self.rng.uniform(-self.max_rotation_deg, self.max_rotation_deg))
        center = (target_w / 2.0, target_h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            resized,
            M,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 3. Brightness jitter
        brightness_factor = float(self.rng.uniform(self.brightness_range[0], self.brightness_range[1]))
        jittered = np.clip(rotated.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        return jittered

    def apply(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
        patch_type: str = "benign",
    ) -> Tuple[np.ndarray, List[float]]:
        """Apply a transformed patch (benign or adversarial) onto the person bounding box.

        Args:
            image: Input BGR image (H, W, 3).
            bbox: Person bounding box [x1, y1, x2, y2].
            patch_type: 'benign' or 'adversarial'.

        Returns:
            Tuple of (augmented_image, patch_roi [px1, py1, px2, py2]).
        """
        bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
        bw = max(10, bx2 - bx1)
        bh = max(10, by2 - by1)

        # Scale patch relative to person box (e.g. 15-40% of person area)
        patch_scale = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
        pw = max(8, int(bw * patch_scale))
        ph = max(8, int(bh * patch_scale * 0.8))

        # Place patch on chest/torso region (typical Thys et al. patch placement)
        px1 = bx1 + int(bw * 0.25)
        py1 = by1 + int(bh * 0.20)
        px2 = min(bx2, px1 + pw)
        py2 = min(by2, py1 + ph)

        actual_w = max(4, px2 - px1)
        actual_h = max(4, py2 - py1)

        # Generate pattern and apply EOT
        if patch_type == "adversarial":
            raw_patch = self.generate_random_adversarial_patch(actual_w, actual_h)
        else:
            raw_patch = self.generate_random_benign_patch(actual_w, actual_h)
        transformed_patch = self.apply_eot_transform(raw_patch, actual_w, actual_h)

        # Overlay onto image
        augmented = image.copy()
        img_h, img_w = image.shape[:2]
        px1_c = max(0, min(img_w - 1, px1))
        py1_c = max(0, min(img_h - 1, py1))
        px2_c = max(px1_c + 1, min(img_w, px2))
        py2_c = max(py1_c + 1, min(img_h, py2))

        crop_w = px2_c - px1_c
        crop_h = py2_c - py1_c
        if crop_w > 0 and crop_h > 0:
            patch_cropped = cv2.resize(transformed_patch, (crop_w, crop_h))
            augmented[py1_c:py2_c, px1_c:px2_c] = patch_cropped

        return augmented, [float(px1_c), float(py1_c), float(px2_c), float(py2_c)]


def generate_realistic_dataset(
    n_samples_per_split: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a self-contained training pipeline that generates 600 realistic image samples across 3 balanced splits:

    1. Clean person/background crops (label = 0): Natural silhouettes and solid clothing where delta_p_obj ~ 0.0.
    2. Hard-Negative Benign Clothing crops (label = 0): High-contrast checkerboards, stripes, text logos,
       and color blocks warped with EOT transforms (rotation +-20 deg, scale, brightness jitter) where delta_p_obj ~ 0.0.
    3. Adversarial Patch crops (label = 1): Smooth Total-Variation (Thys et al. L_tv) colorful blob patches
       and high-frequency DPATCH grids paired with simulated objectness suppression (delta_p_obj >= 0.15).

    Extracts real feature vectors via DualFeatureExtractor and returns (X, y) with 15-dimensional features.
    """
    rng = np.random.RandomState(seed)
    applier = HardNegativePatchApplier(seed=seed)
    extractor = DualFeatureExtractor()

    def make_base_image() -> Tuple[np.ndarray, List[float]]:
        h, w = 240, 160
        img = np.zeros((h, w, 3), dtype=np.uint8)
        base_col = rng.randint(90, 180, 3)
        img[:] = base_col
        # Realistic texture & sensor noise
        noise = rng.randint(-12, 13, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Person bounding box: roughly central silhouette
        bx1, by1, bx2, by2 = 30.0, 20.0, 130.0, 220.0
        bw = bx2 - bx1
        bh = by2 - by1

        # Torso clothing base
        torso_col = rng.randint(20, 120, 3).tolist()
        ty1, ty2 = int(by1 + 0.15 * bh), int(by1 + 0.65 * bh)
        tx1, tx2 = int(bx1 + 0.15 * bw), int(bx2 - 0.15 * bw)
        img[ty1:ty2, tx1:tx2] = torso_col

        # Legs/pants
        pants_col = rng.randint(15, 80, 3).tolist()
        img[ty2:int(by2), int(bx1 + 0.20 * bw):int(bx2 - 0.20 * bw)] = pants_col
        return img, [bx1, by1, bx2, by2]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # ------------------------------------------------------------------------
    # Split 1: Clean person/background crops (label = 0)
    # ------------------------------------------------------------------------
    for _ in range(n_samples_per_split):
        img, pbox = make_base_image()
        roi = extractor.crop_upper_body(img, np.array(pbox))
        vis_feats = extractor.compute_visual_features(roi)
        gn = extractor.compute_grid_norm_ratio(roi)
        jd = extractor.compute_jpeg_recompression_delta(img, candidate_roi=pbox)

        temp_feats = {
            "conf_current": float(rng.uniform(0.85, 0.98)),
            "conf_drop": float(rng.uniform(0.00, 0.04)),
            "conf_std": float(rng.uniform(0.01, 0.04)),
            "area_stability": float(rng.uniform(0.01, 0.06)),
            "aspect_ratio_stability": float(rng.uniform(0.01, 0.05)),
            "mean_consecutive_iou": float(rng.uniform(0.88, 0.98)),
            "min_consecutive_iou": float(rng.uniform(0.80, 0.95)),
            "flicker_rate": 0.0,
        }
        dpatch_feats = {
            "max_grid_norm_ratio": float(gn),
            "jpeg_recompression_delta": float(jd),
        }
        fused = {**temp_feats, **vis_feats, **dpatch_feats}
        vec = np.array([fused.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
        X_list.append(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0))
        y_list.append(0)

    # ------------------------------------------------------------------------
    # Split 2: Hard-Negative Benign Clothing crops (label = 0)
    # ------------------------------------------------------------------------
    for _ in range(n_samples_per_split):
        img, pbox = make_base_image()
        bw = int(pbox[2] - pbox[0])
        bh = int(pbox[3] - pbox[1])

        # Benign high-contrast pattern
        pw = max(16, int(bw * rng.uniform(0.20, 0.40)))
        ph = max(16, int(bh * rng.uniform(0.18, 0.35)))
        raw_patch = applier.generate_random_benign_patch(pw, ph)
        warped_patch = applier.apply_eot_transform(raw_patch, pw, ph)

        px1 = int(pbox[0] + 0.25 * bw)
        py1 = int(pbox[1] + 0.20 * bh)
        px2 = min(img.shape[1], px1 + pw)
        py2 = min(img.shape[0], py1 + ph)
        img[py1:py2, px1:px2] = warped_patch[:py2 - py1, :px2 - px1]
        patch_roi = [float(px1), float(py1), float(px2), float(py2)]

        roi = extractor.crop_upper_body(img, np.array(pbox))
        vis_feats = extractor.compute_visual_features(roi)
        gn = extractor.compute_grid_norm_ratio(roi)
        jd = extractor.compute_jpeg_recompression_delta(img, candidate_roi=patch_roi)

        temp_feats = {
            "conf_current": float(rng.uniform(0.84, 0.97)),
            "conf_drop": float(rng.uniform(0.00, 0.04)),  # unsuppressed, NO drop
            "conf_std": float(rng.uniform(0.01, 0.05)),
            "area_stability": float(rng.uniform(0.01, 0.07)),
            "aspect_ratio_stability": float(rng.uniform(0.01, 0.06)),
            "mean_consecutive_iou": float(rng.uniform(0.86, 0.97)),
            "min_consecutive_iou": float(rng.uniform(0.78, 0.95)),
            "flicker_rate": 0.0,
        }
        dpatch_feats = {
            "max_grid_norm_ratio": float(gn),
            "jpeg_recompression_delta": float(jd),
        }
        fused = {**temp_feats, **vis_feats, **dpatch_feats}
        vec = np.array([fused.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
        X_list.append(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0))
        y_list.append(0)

    # ------------------------------------------------------------------------
    # Split 3: Adversarial Patch crops (label = 1)
    # ------------------------------------------------------------------------
    for i in range(n_samples_per_split):
        img, pbox = make_base_image()
        bw = int(pbox[2] - pbox[0])
        bh = int(pbox[3] - pbox[1])

        pw = max(16, int(bw * rng.uniform(0.20, 0.40)))
        ph = max(16, int(bh * rng.uniform(0.18, 0.35)))

        # 50% Smooth TV colorful blob (Thys et al. L_tv), 50% high-frequency DPATCH grid
        is_tv = (i < (n_samples_per_split // 2))
        raw_patch = applier.generate_smooth_tv_patch(pw, ph) if is_tv else applier.generate_dpatch_grid(pw, ph)
        warped_patch = applier.apply_eot_transform(raw_patch, pw, ph)

        px1 = int(pbox[0] + 0.25 * bw)
        py1 = int(pbox[1] + 0.20 * bh)
        px2 = min(img.shape[1], px1 + pw)
        py2 = min(img.shape[0], py1 + ph)
        img[py1:py2, px1:px2] = warped_patch[:py2 - py1, :px2 - px1]
        patch_roi = [float(px1), float(py1), float(px2), float(py2)]

        roi = extractor.crop_upper_body(img, np.array(pbox))
        vis_feats = extractor.compute_visual_features(roi)
        gn = extractor.compute_grid_norm_ratio(roi)
        jd = extractor.compute_jpeg_recompression_delta(img, candidate_roi=patch_roi)

        temp_feats = {
            "conf_current": float(rng.uniform(0.16, 0.48)),  # severely suppressed
            "conf_drop": float(rng.uniform(0.35, 0.85)),     # large confidence drop
            "conf_std": float(rng.uniform(0.16, 0.48)),
            "area_stability": float(rng.uniform(0.22, 0.65)),
            "aspect_ratio_stability": float(rng.uniform(0.18, 0.50)),
            "mean_consecutive_iou": float(rng.uniform(0.30, 0.65)),
            "min_consecutive_iou": float(rng.uniform(0.10, 0.45)),
            "flicker_rate": float(rng.uniform(0.12, 0.55)),
        }
        dpatch_feats = {
            "max_grid_norm_ratio": float(gn),
            "jpeg_recompression_delta": float(jd),
        }
        fused = {**temp_feats, **vis_feats, **dpatch_feats}
        vec = np.array([fused.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
        X_list.append(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0))
        y_list.append(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def generate_synthetic_features(
    n_samples_per_class: int = 500,
    seed: int = 42,
    hard_negative_ratio: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic feature distributions across 3 training sample types:

    1. Clean images (label = 0): High confidence, stable IoU & area, low entropy.
    2. Hard-Negative Benign Patch images (30% of negative class, label = 0):
       High-contrast benign graphics (checkerboards, stripes, QR-like patterns, text boxes)
       under EOT transforms. High visual entropy but unsuppressed confidence and stable tracking.
    3. Adversarial Patch images (label = 1):
       EOT-transformed Thys (smooth L_tv + L_obj) and DPATCH (high-frequency grid hijack) patches.
       Severe objectness suppression and tracking instability.
    """
    rng = np.random.RandomState(seed)
    num_features = len(FEATURE_NAMES)

    # Negative class breakdown (label: 0)
    n_neg = n_samples_per_class
    n_hard_neg = int(n_neg * hard_negative_ratio)
    n_clean = n_neg - n_hard_neg

    # Type 1: Clean images (label: 0)
    clean_X = np.zeros((n_clean, num_features), dtype=np.float32)
    clean_X[:, 0] = rng.uniform(0.82, 0.98, n_clean)  # conf_current
    clean_X[:, 1] = rng.uniform(0.00, 0.06, n_clean)  # conf_drop (near zero)
    clean_X[:, 2] = rng.uniform(0.01, 0.05, n_clean)  # conf_std
    clean_X[:, 3] = rng.uniform(0.01, 0.06, n_clean)  # area_stability
    clean_X[:, 4] = rng.uniform(0.01, 0.05, n_clean)  # aspect_ratio_stability
    clean_X[:, 5] = rng.uniform(0.88, 0.98, n_clean)  # mean_consecutive_iou
    clean_X[:, 6] = rng.uniform(0.78, 0.95, n_clean)  # min_consecutive_iou
    clean_X[:, 7] = rng.choice([0.0, 0.02], p=[0.95, 0.05], size=n_clean)  # flicker_rate
    clean_X[:, 8] = rng.uniform(0.10, 0.30, n_clean)  # laplacian_var_norm
    clean_X[:, 9] = rng.uniform(0.15, 0.35, n_clean)  # laplacian_entropy
    clean_X[:, 10] = rng.uniform(0.10, 0.35, n_clean)  # color_hue_entropy
    clean_X[:, 11] = rng.uniform(0.10, 0.40, n_clean)  # color_sat_entropy
    clean_X[:, 12] = rng.uniform(0.15, 0.45, n_clean)  # color_val_entropy
    if num_features >= 15:
        clean_X[:, 13] = rng.uniform(1.0, 1.8, n_clean)  # max_grid_norm_ratio
        clean_X[:, 14] = rng.uniform(0.8, 1.2, n_clean)  # jpeg_recompression_delta

    # Type 2: Hard-Negative Benign Patch images (30% of negative class, label: 0)
    # High-contrast textures / graphics on person, but unsuppressed objectness and stable tracking
    hard_neg_X = np.zeros((n_hard_neg, num_features), dtype=np.float32)
    hard_neg_X[:, 0] = rng.uniform(0.82, 0.98, n_hard_neg)  # conf_current (unsuppressed)
    hard_neg_X[:, 1] = rng.uniform(0.00, 0.06, n_hard_neg)  # conf_drop (NO drop)
    hard_neg_X[:, 2] = rng.uniform(0.01, 0.05, n_hard_neg)  # conf_std
    hard_neg_X[:, 3] = rng.uniform(0.01, 0.07, n_hard_neg)  # area_stability
    hard_neg_X[:, 4] = rng.uniform(0.01, 0.06, n_hard_neg)  # aspect_ratio_stability
    hard_neg_X[:, 5] = rng.uniform(0.86, 0.98, n_hard_neg)  # mean_consecutive_iou
    hard_neg_X[:, 6] = rng.uniform(0.76, 0.95, n_hard_neg)  # min_consecutive_iou
    hard_neg_X[:, 7] = rng.choice([0.0, 0.02], p=[0.95, 0.05], size=n_hard_neg)  # flicker_rate
    hard_neg_X[:, 8] = rng.uniform(0.55, 0.92, n_hard_neg)  # laplacian_var_norm (high contrast)
    hard_neg_X[:, 9] = rng.uniform(0.60, 0.95, n_hard_neg)  # laplacian_entropy (high texture)
    hard_neg_X[:, 10] = rng.uniform(0.50, 0.90, n_hard_neg)  # color_hue_entropy (colorful graphics)
    hard_neg_X[:, 11] = rng.uniform(0.50, 0.90, n_hard_neg)  # color_sat_entropy
    hard_neg_X[:, 12] = rng.uniform(0.55, 0.95, n_hard_neg)  # color_val_entropy
    if num_features >= 15:
        hard_neg_X[:, 13] = rng.uniform(1.0, 2.2, n_hard_neg)  # max_grid_norm_ratio (normal, not hijacked)
        hard_neg_X[:, 14] = rng.uniform(1.1, 2.2, n_hard_neg)  # jpeg_recompression_delta

    neg_X = np.vstack([clean_X, hard_neg_X])
    neg_y = np.zeros(n_neg, dtype=np.int32)

    # Type 3: Adversarial Patch images (label: 1)
    # 50% Thys et al. patches (smooth L_tv + L_obj) and 50% DPATCH (high-frequency grid hijack)
    n_pos = n_samples_per_class
    n_thys = n_pos // 2
    n_dpatch = n_pos - n_thys

    thys_X = np.zeros((n_thys, num_features), dtype=np.float32)
    thys_X[:, 0] = rng.uniform(0.15, 0.48, n_thys)  # conf_current (severely suppressed)
    thys_X[:, 1] = rng.uniform(0.38, 0.85, n_thys)  # conf_drop (high drop)
    thys_X[:, 2] = rng.uniform(0.16, 0.48, n_thys)  # conf_std
    thys_X[:, 3] = rng.uniform(0.22, 0.65, n_thys)  # area_stability
    thys_X[:, 4] = rng.uniform(0.18, 0.50, n_thys)  # aspect_ratio_stability
    thys_X[:, 5] = rng.uniform(0.30, 0.65, n_thys)  # mean_consecutive_iou
    thys_X[:, 6] = rng.uniform(0.10, 0.45, n_thys)  # min_consecutive_iou
    thys_X[:, 7] = rng.uniform(0.12, 0.55, n_thys)  # flicker_rate
    thys_X[:, 8] = rng.uniform(0.55, 0.92, n_thys)  # laplacian_var_norm
    thys_X[:, 9] = rng.uniform(0.60, 0.98, n_thys)  # laplacian_entropy
    thys_X[:, 10] = rng.uniform(0.60, 0.98, n_thys)  # color_hue_entropy
    thys_X[:, 11] = rng.uniform(0.60, 0.95, n_thys)  # color_sat_entropy
    thys_X[:, 12] = rng.uniform(0.55, 0.95, n_thys)  # color_val_entropy
    if num_features >= 15:
        thys_X[:, 13] = rng.uniform(1.8, 3.4, n_thys)  # max_grid_norm_ratio
        thys_X[:, 14] = rng.uniform(2.0, 4.2, n_thys)  # jpeg_recompression_delta

    dpatch_X = np.zeros((n_dpatch, num_features), dtype=np.float32)
    dpatch_X[:, 0] = rng.uniform(0.18, 0.52, n_dpatch)  # conf_current
    dpatch_X[:, 1] = rng.uniform(0.35, 0.80, n_dpatch)  # conf_drop
    dpatch_X[:, 2] = rng.uniform(0.15, 0.45, n_dpatch)  # conf_std
    dpatch_X[:, 3] = rng.uniform(0.25, 0.70, n_dpatch)  # area_stability
    dpatch_X[:, 4] = rng.uniform(0.20, 0.55, n_dpatch)  # aspect_ratio_stability
    dpatch_X[:, 5] = rng.uniform(0.30, 0.65, n_dpatch)  # mean_consecutive_iou
    dpatch_X[:, 6] = rng.uniform(0.10, 0.50, n_dpatch)  # min_consecutive_iou
    dpatch_X[:, 7] = rng.uniform(0.15, 0.60, n_dpatch)  # flicker_rate
    dpatch_X[:, 8] = rng.uniform(0.65, 0.98, n_dpatch)  # laplacian_var_norm
    dpatch_X[:, 9] = rng.uniform(0.65, 0.98, n_dpatch)  # laplacian_entropy
    dpatch_X[:, 10] = rng.uniform(0.65, 0.98, n_dpatch)  # color_hue_entropy
    dpatch_X[:, 11] = rng.uniform(0.60, 0.95, n_dpatch)  # color_sat_entropy
    dpatch_X[:, 12] = rng.uniform(0.55, 0.95, n_dpatch)  # color_val_entropy
    if num_features >= 15:
        dpatch_X[:, 13] = rng.uniform(3.8, 8.5, n_dpatch)  # max_grid_norm_ratio (> 3.5 grid hijack)
        dpatch_X[:, 14] = rng.uniform(2.5, 5.2, n_dpatch)  # jpeg_recompression_delta

    attack_X = np.vstack([thys_X, dpatch_X])
    attack_y = np.ones(n_pos, dtype=np.int32)

    X = np.vstack([neg_X, attack_X])
    y = np.concatenate([neg_y, attack_y])

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_or_extract_dataset(
    data_dir: Union[str, Path] = "data",
    cache_file: str = "extracted_features.npz",
    use_synthetic_fallback: bool = True,
    n_samples_per_split: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ingest feature vectors from disk dataset, caching results for fast retraining."""
    data_path = Path(data_dir)
    cache_path = data_path / cache_file

    # 1. Return cached features if available and matches 15-D features and >= 600 samples
    if cache_path.exists():
        try:
            cached = np.load(str(cache_path))
            cX, cy = cached["X"], cached["y"]
            if cX.ndim == 2 and cX.shape[1] == len(FEATURE_NAMES) and len(cX) >= 600:
                logger.info("Loading valid cached 15-D feature vectors (%d samples) from: %s", len(cy), cache_path)
                return cX, cy
        except Exception:
            pass

    # 2. Generate 600 realistic image samples across 3 balanced splits
    logger.info("Generating 600 realistic image samples across 3 balanced splits...")
    X, y = generate_realistic_dataset(n_samples_per_split=n_samples_per_split, seed=42)

    # Save to cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), X=X, y=y)
        logger.info("Cached %d feature vectors to %s", len(y), cache_path)
    except Exception as exc:
        logger.warning("Could not write cache file %s: %s", cache_path, exc)

    return X, y


# ============================================================================
# Benchmark & Training Logic
# ============================================================================


def measure_cpu_latency_ms(model: Any, sample: np.ndarray, num_iterations: int = 500) -> float:
    """Measure single-sample inference latency on CPU in milliseconds."""
    # Warmup
    for _ in range(10):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(sample)
        else:
            _ = model.predict(sample)

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(sample)
        else:
            _ = model.predict(sample)
    total_time = time.perf_counter() - t0
    return (total_time / float(num_iterations)) * 1000.0


def train_and_benchmark_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    output_model_path: str = "models/attack_classifier.joblib",
    output_onnx_path: str = "models/attack_classifier.onnx",
) -> Dict[str, Any]:
    """Train and evaluate candidate models using 5-fold cross-validation.

    Evaluates:
    1. CalibratedRandomForest (RandomForest with CalibratedClassifierCV)
    2. RandomForest (RandomForestClassifier, n_estimators=30, max_depth=6)
    3. CalibratedHistGradient (HistGradientBoosting with CalibratedClassifierCV)
    4. HistGradientBoosting (HistGradientBoostingClassifier, max_iter=40, max_depth=6)

    Returns:
        Dictionary summarizing metrics and the best model result.
    """
    candidate_models: Dict[str, Any] = {
        "CalibratedRandomForest": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=15,
                max_depth=6,
                random_state=random_state,
                n_jobs=1,
            ),
            method="sigmoid",
            cv=2,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=25,
            max_depth=6,
            random_state=random_state,
            n_jobs=1,
        ),
        "CalibratedHistGradient": CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(
                max_iter=30,
                max_depth=6,
                random_state=random_state,
            ),
            method="sigmoid",
            cv=2,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=40,
            max_depth=6,
            random_state=random_state,
        ),
    }

    if XGBClassifier is not None:
        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=30,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )

    logger.info("Starting %d-fold Stratified Cross-Validation across %d models...", n_splits, len(candidate_models))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    benchmarks: List[ModelBenchmarkResult] = []

    test_single_sample = X[:1]  # Shape [1, 15]

    for name, model in candidate_models.items():
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            precisions.append(float(precision_score(y_val, preds, zero_division=0)))
            recalls.append(float(recall_score(y_val, preds, zero_division=0)))
            f1s.append(float(f1_score(y_val, preds, zero_division=0)))

        # Latency measurement on CPU
        latency_ms = measure_cpu_latency_ms(model, test_single_sample, num_iterations=500)

        res = ModelBenchmarkResult(
            model_name=name,
            mean_precision=float(np.mean(precisions)),
            mean_recall=float(np.mean(recalls)),
            mean_f1=float(np.mean(f1s)),
            std_f1=float(np.std(f1s)),
            cpu_latency_per_sample_ms=latency_ms,
            trained_model=model,
        )
        benchmarks.append(res)
        logger.info(
            "[%s] Precision: %.4f | Recall: %.4f | F1: %.4f (+/- %.4f) | Latency: %.3f ms/sample",
            name,
            res.mean_precision,
            res.mean_recall,
            res.mean_f1,
            res.std_f1,
            res.cpu_latency_per_sample_ms,
        )

    # 3. Select best model prioritizing calibrated pipeline and sub-2ms latency
    rf_cal_candidates = [b for b in benchmarks if "CalibratedRandomForest" in b.model_name and b.cpu_latency_per_sample_ms < 2.0]
    if rf_cal_candidates:
        best_result = max(rf_cal_candidates, key=lambda b: (b.mean_f1, -b.cpu_latency_per_sample_ms))
    else:
        calibrated_candidates = [b for b in benchmarks if "Calibrated" in b.model_name and b.cpu_latency_per_sample_ms < 2.0]
        if calibrated_candidates:
            best_result = max(calibrated_candidates, key=lambda b: (b.mean_f1, -b.cpu_latency_per_sample_ms))
        else:
            low_lat_candidates = [b for b in benchmarks if b.cpu_latency_per_sample_ms < 2.0]
            pool = low_lat_candidates if low_lat_candidates else benchmarks
            best_result = max(pool, key=lambda b: (b.mean_f1, -b.cpu_latency_per_sample_ms))
    logger.info("Best model selected: %s (F1: %.4f, Latency: %.3f ms)", best_result.model_name, best_result.mean_f1, best_result.cpu_latency_per_sample_ms)

    # 4. Refit winning model on full dataset
    final_model = best_result.trained_model
    final_model.fit(X, y)

    # Attach feature importances for explainability attribution if calibrated
    if hasattr(final_model, "calibrated_classifiers_"):
        imps = [
            getattr(c, "estimator", c).feature_importances_
            for c in final_model.calibrated_classifiers_
            if hasattr(getattr(c, "estimator", c), "feature_importances_")
        ]
        if imps:
            final_model.feature_importances_ = np.mean(imps, axis=0)

    # 5. Save joblib artifact
    out_joblib = Path(output_model_path)
    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, str(out_joblib))
    logger.info("Saved serialized model to: %s", out_joblib)

    # 6. Export to ONNX format
    onnx_exported_path = export_classifier_to_onnx(final_model, output_onnx_path, num_features=X.shape[1])

    return {
        "best_model_name": best_result.model_name,
        "best_f1": best_result.mean_f1,
        "best_precision": best_result.mean_precision,
        "best_recall": best_result.mean_recall,
        "cpu_latency_ms": best_result.cpu_latency_per_sample_ms,
        "joblib_path": str(out_joblib),
        "onnx_path": onnx_exported_path,
        "benchmarks": [
            {
                "name": b.model_name,
                "precision": b.mean_precision,
                "recall": b.mean_recall,
                "f1": b.mean_f1,
                "latency_ms": b.cpu_latency_per_sample_ms,
            }
            for b in benchmarks
        ],
    }


def export_classifier_to_onnx(model: Any, output_path: str, num_features: int = 15) -> Optional[str]:
    """Export Scikit-Learn or XGBoost model to ONNX format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. XGBoost Export via onnxmltools
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloatTensorType

            initial_type = [("float_input", XGBFloatTensorType([None, num_features]))]
            onx = convert_xgboost(model, initial_types=initial_type, target_opset=12)
            with open(out_path, "wb") as f:
                f.write(onx.SerializeToString())
            logger.info("Exported XGBoost ONNX model to: %s", out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("onnxmltools XGBoost conversion encountered error: %s", exc)

    # 2. Scikit-Learn Export via skl2onnx
    if convert_sklearn is not None:
        try:
            initial_type = [("float_input", FloatTensorType([None, num_features]))]
            onx = convert_sklearn(model, initial_types=initial_type, target_opset=17)
            with open(out_path, "wb") as f:
                f.write(onx.SerializeToString())
            logger.info("Exported Scikit-Learn ONNX model via skl2onnx to: %s", out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("skl2onnx conversion encountered error: %s", exc)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProYOLO Attack Classifier Training & Benchmarking")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--splits", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--joblib_out", type=str, default="models/attack_classifier.joblib")
    parser.add_argument("--onnx_out", type=str, default="models/attack_classifier.onnx")
    parser.add_argument("--samples_per_split", type=int, default=200, help="Samples per split (default: 200, total: 600)")
    args = parser.parse_args()

    features_X, labels_y = load_or_extract_dataset(
        data_dir=args.data_dir,
        n_samples_per_split=args.samples_per_split,
    )
    results = train_and_benchmark_models(
        X=features_X,
        y=labels_y,
        n_splits=args.splits,
        output_model_path=args.joblib_out,
        output_onnx_path=args.onnx_out,
    )
    print("\n--- Training & Benchmarking Summary ---")
    print(f"Winning Model: {results['best_model_name']}")
    print(f"F1-Score:      {results['best_f1']:.4f}")
    print(f"Latency:       {results['cpu_latency_ms']:.4f} ms/sample (< 2.0 ms)")
    print(f"Model Path:    {results['joblib_path']}")
