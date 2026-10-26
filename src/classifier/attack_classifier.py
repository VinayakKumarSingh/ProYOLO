
"""Real-Time Attack Classification Engine for ProYOLO.

Evaluates fused 1D feature vectors from FeatureExtractor in <2 ms on CPU,
predicts attack probability against calibrated thresholds, and provides feature
importance attribution explaining triggered alerts.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import joblib
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

from src.detection.feature_extractor import FEATURE_NAMES, FeaturePacket

logger = logging.getLogger(__name__)

# Empirical normal/clean feature baselines for attribution scaling
CLEAN_FEATURE_BASELINES: Dict[str, Tuple[float, float]] = {
    # name: (mean, std)
    "conf_current": (0.90, 0.08),
    "conf_drop": (0.02, 0.03),
    "conf_std": (0.03, 0.02),
    "area_stability": (0.04, 0.03),
    "aspect_ratio_stability": (0.03, 0.02),
    "mean_consecutive_iou": (0.92, 0.05),
    "min_consecutive_iou": (0.85, 0.08),
    "flicker_rate": (0.01, 0.02),
    "laplacian_var_norm": (0.22, 0.08),
    "laplacian_entropy": (0.28, 0.08),
    "color_hue_entropy": (0.25, 0.09),
    "color_sat_entropy": (0.25, 0.09),
    "color_val_entropy": (0.35, 0.10),
    "max_grid_norm_ratio": (1.8, 0.4),
    "jpeg_recompression_delta": (1.0, 0.2),
}


class AttackPrediction(NamedTuple):
    """Result of real-time attack classifier inference."""

    is_attack: bool
    attack_probability: float
    confidence_clean: float
    feature_importances: Dict[str, float]
    top_contributing_features: List[Tuple[str, float]]
    inference_latency_ms: float
    causal_verification_passed: bool = False
    delta_p_obj: float = 0.0


class AttackClassifier:
    """Lightweight CPU-based attack classification and explainability engine.

    Attributes:
        threshold: Decision boundary probability threshold (default: 0.65).
        model_path: Filepath to serialized .joblib or .onnx model.
        model: Loaded model object or ONNX Runtime inference session.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.65,
        model: Optional[Any] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.is_onnx = False

        if model is not None:
            self.model = model
            self.model_path = None
        else:
            # Check for candidate model files: prefer ONNX for <0.1ms inference
            resolved_path: Optional[Path] = None
            if model_path is not None:
                p = Path(model_path)
                if p.exists():
                    resolved_path = p
            else:
                candidate_onnx = Path("models/attack_classifier.onnx")
                candidate_joblib = Path("models/attack_classifier.joblib")
                if candidate_onnx.exists():
                    resolved_path = candidate_onnx
                elif candidate_joblib.exists():
                    resolved_path = candidate_joblib

            self.model_path = resolved_path
            if self.model_path is not None and self.model_path.exists():
                self.model = self._load_model(self.model_path)
            else:
                logger.warning("No pre-trained model found. Initializing with default heuristic classifier.")
                self.model = self._init_fallback_model()

        # Cache global model feature importances
        self.global_importances = self._extract_global_importances()

    def _load_model(self, path: Path) -> Any:
        """Load model from .joblib or .onnx file."""
        if path.suffix.lower() == ".onnx":
            if ort is None:
                raise ImportError("onnxruntime is required to load .onnx model files.")
            self.is_onnx = True
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            logger.info("Loaded ONNX classifier session from: %s", path)
            return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        logger.info("Loading Scikit-Learn / XGBoost model from: %s", path)
        return joblib.load(str(path))

    def _init_fallback_model(self) -> Any:
        """Initialize an empirical Random Forest model if no pre-trained file exists."""
        from src.classifier.train_classifier import generate_synthetic_features
        from sklearn.ensemble import RandomForestClassifier

        logger.info("Training lightweight fallback RandomForest model on synthetic distributions...")
        X, y = generate_synthetic_features(n_samples_per_class=300)
        rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=1)
        rf.fit(X, y)
        return rf

    def _extract_global_importances(self) -> Dict[str, float]:
        """Extract global feature importances from model if available."""
        # 1. From active model instance
        if hasattr(self.model, "feature_importances_"):
            raw_imps = self.model.feature_importances_
            total = float(np.sum(raw_imps))
            norm_imps = raw_imps / (total + 1e-6)
            return {
                name: float(norm_imps[i])
                for i, name in enumerate(FEATURE_NAMES)
                if i < len(norm_imps)
            }

        # 2. If model is ONNX, check if companion joblib exists to read weights
        joblib_companion = Path("models/attack_classifier.joblib")
        if joblib_companion.exists():
            try:
                jb = joblib.load(str(joblib_companion))
                if hasattr(jb, "feature_importances_"):
                    raw_imps = jb.feature_importances_
                    total = float(np.sum(raw_imps))
                    norm_imps = raw_imps / (total + 1e-6)
                    return {
                        name: float(norm_imps[i])
                        for i, name in enumerate(FEATURE_NAMES)
                        if i < len(norm_imps)
                    }
            except Exception:
                pass

        # 3. Default equal baseline importance
        equal_weight = 1.0 / float(len(FEATURE_NAMES))
        return {name: equal_weight for name in FEATURE_NAMES}

    @staticmethod
    def _check_patch_in_person(
        patch_rois: Sequence[Sequence[float]],
        person_bbox: Optional[Sequence[float]],
    ) -> bool:
        """Check if any candidate patch ROI overlaps or lies inside person bounding box."""
        if person_bbox is None or not patch_rois:
            # If no disjoint patch ROIs are explicitly specified, assume track upper body localization
            return True

        bx1, by1, bx2, by2 = (
            float(person_bbox[0]),
            float(person_bbox[1]),
            float(person_bbox[2]),
            float(person_bbox[3]),
        )
        for roi in patch_rois:
            rx1, ry1, rx2, ry2 = float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])
            ix1 = max(rx1, bx1)
            iy1 = max(ry1, by1)
            ix2 = min(rx2, bx2)
            iy2 = min(ry2, by2)
            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            if (inter_w * inter_h) > 0.0:
                return True
        return False

    def predict(
        self,
        features: Union[np.ndarray, FeaturePacket, Sequence[float]],
        delta_p_obj: float = 0.0,
        patch_roi: Optional[Sequence[float]] = None,
        patch_rois: Optional[Sequence[Sequence[float]]] = None,
        person_bbox: Optional[Sequence[float]] = None,
        recovered_objects_count: int = 0,
        conf_drop: Optional[float] = None,
        max_grid_norm_ratio: Optional[float] = None,
    ) -> AttackPrediction:
        """Predict attack probability and evaluate Two-Stage Causal Verification Gate.

        Args:
            features: 1D feature array of shape (13 or 15,), FeaturePacket, or sequence of floats.
            delta_p_obj: Causal objectness recovery metric from detector.py Pass 2 inpainting.
            patch_roi: Optional candidate patch ROI [x1, y1, x2, y2].
            patch_rois: Optional sequence of candidate patch ROIs.
            person_bbox: Optional detected person bounding box [x1, y1, x2, y2].
            recovered_objects_count: Number of suppressed objects recovered in Pass 2 inpainting.
            conf_drop: Optional explicit drop in tracked person confidence.
            max_grid_norm_ratio: Optional DPATCH grid norm ratio.

        Returns:
            AttackPrediction containing attack flag, probability, and top contributing features.
        """
        # 1. Input normalization and contextual parameter extraction
        t_start = time.perf_counter()

        resolved_person_bbox = person_bbox
        resolved_delta_p_obj = float(delta_p_obj)
        all_patch_rois: List[Sequence[float]] = []
        if patch_roi is not None:
            all_patch_rois.append(patch_roi)
        if patch_rois:
            all_patch_rois.extend(patch_rois)

        if isinstance(features, FeaturePacket):
            raw_vec = features.feature_vector
            if resolved_person_bbox is None:
                resolved_person_bbox = features.bbox
            if resolved_delta_p_obj == 0.0:
                resolved_delta_p_obj = float(getattr(features, "delta_p_obj", 0.0))
            if not all_patch_rois and features.candidate_patch_rois:
                all_patch_rois.extend(features.candidate_patch_rois)
            if conf_drop is None and "conf_drop" in features.feature_dict:
                conf_drop = float(features.feature_dict["conf_drop"])
            if max_grid_norm_ratio is None and "max_grid_norm_ratio" in features.feature_dict:
                max_grid_norm_ratio = float(features.feature_dict["max_grid_norm_ratio"])
        elif isinstance(features, np.ndarray):
            raw_vec = features.flatten()
        else:
            raw_vec = np.array(features, dtype=np.float32).flatten()

        if len(raw_vec) not in (len(FEATURE_NAMES), 13):
            raise ValueError(
                f"Feature vector length mismatch: expected {len(FEATURE_NAMES)} or 13 features, got {len(raw_vec)}."
            )

        sanitized_vec = np.nan_to_num(raw_vec, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        input_2d = sanitized_vec.reshape(1, -1)

        # Ensure compatibility with pre-trained models expecting 13 features
        expected_features = 13
        if self.is_onnx:
            in_shape = self.model.get_inputs()[0].shape
            if len(in_shape) > 1 and isinstance(in_shape[1], int):
                expected_features = in_shape[1]
        elif hasattr(self.model, "n_features_in_"):
            expected_features = self.model.n_features_in_

        model_input = input_2d[:, :expected_features] if input_2d.shape[1] >= expected_features else input_2d

        # 2. Forward model inference
        if self.is_onnx:
            input_name = self.model.get_inputs()[0].name
            ort_outs = self.model.run(None, {input_name: model_input})
            # ONNX Scikit-Learn output: [labels, probabilities_dict or array]
            if len(ort_outs) > 1 and isinstance(ort_outs[1], list):
                # Map of {0: p0, 1: p1}
                p_attack = float(ort_outs[1][0].get(1, 0.0))
                p_clean = float(ort_outs[1][0].get(0, 1.0 - p_attack))
            elif len(ort_outs) > 1 and isinstance(ort_outs[1], np.ndarray):
                p_attack = float(ort_outs[1][0, 1])
                p_clean = float(ort_outs[1][0, 0])
            else:
                pred_label = int(ort_outs[0][0])
                p_attack = 1.0 if pred_label == 1 else 0.0
                p_clean = 1.0 - p_attack
        else:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(model_input)[0]
                p_clean = float(probs[0])
                p_attack = float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0])
            else:
                pred_label = int(self.model.predict(model_input)[0])
                p_attack = 1.0 if pred_label == 1 else 0.0
                p_clean = 1.0 - p_attack

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Extract contextual causal verification attributes
        if conf_drop is None:
            resolved_conf_drop = float(sanitized_vec[1]) if len(sanitized_vec) > 1 else 0.0
        else:
            resolved_conf_drop = float(conf_drop)

        if max_grid_norm_ratio is None:
            resolved_norm_ratio = float(sanitized_vec[13]) if len(sanitized_vec) > 13 else 1.0
        else:
            resolved_norm_ratio = float(max_grid_norm_ratio)

        has_conf_drop = bool(resolved_conf_drop >= 0.10)

        resolved_recovered_count = int(recovered_objects_count)
        if resolved_recovered_count == 0 and resolved_delta_p_obj >= 0.15:
            resolved_recovered_count = 1

        patch_in_person = self._check_patch_in_person(all_patch_rois, resolved_person_bbox)

        # 3. Two-Stage Causal Verification Gate
        # (a) Local Thys Attack Gate: The candidate patch ROI overlaps/lies inside a detected person bounding box
        #     AND (delta_p_obj >= 0.15 after Pass 2 inpainting OR classifier_prob >= 0.85 with a drop in tracked person confidence).
        local_thys_gate = bool(
            patch_in_person and (
                resolved_delta_p_obj >= 0.15 or (p_attack >= 0.85 and has_conf_drop)
            )
        )

        # (b) Global DPATCH Gate: The candidate patch ROI exhibits max_grid_norm_ratio > 3.5
        #     AND Pass 2 inpainting recovers >= 1 previously suppressed objects anywhere in the frame.
        global_dpatch_gate = bool(
            resolved_norm_ratio > 3.5 and resolved_recovered_count >= 1
        )

        causal_gate_passed = bool(local_thys_gate or global_dpatch_gate)

        # An alert MUST NOT be flagged as is_attack = True based solely on classifier texture/gradient features
        # if delta_p_obj is near zero.
        # Confirm an adversarial attack (is_attack = True) only if the causal gate passes:
        is_attack = bool(
            (p_attack >= self.threshold or resolved_delta_p_obj >= 0.15)
            and causal_gate_passed
        )

        # 4. Explainability & Feature Contribution Attribution
        top_contributors = self._compute_feature_contributions(sanitized_vec)

        return AttackPrediction(
            is_attack=is_attack,
            attack_probability=round(p_attack, 4),
            confidence_clean=round(p_clean, 4),
            feature_importances=self.global_importances,
            top_contributing_features=top_contributors,
            inference_latency_ms=round(t_elapsed_ms, 4),
            causal_verification_passed=causal_gate_passed,
            delta_p_obj=round(resolved_delta_p_obj, 4),
        )

    def _compute_feature_contributions(self, feature_vec: np.ndarray) -> List[Tuple[str, float]]:
        """Compute relative contribution scores of features explaining why the track was flagged."""
        contributions: List[Tuple[str, float]] = []

        for i, name in enumerate(FEATURE_NAMES):
            if i >= len(feature_vec):
                break
            val = float(feature_vec[i])
            mean_b, std_b = CLEAN_FEATURE_BASELINES.get(name, (0.5, 0.1))
            g_imp = self.global_importances.get(name, 1.0 / len(FEATURE_NAMES))

            # Features where higher values indicate an attack:
            # (conf_drop, entropy, variance, flicker, area_stability, DPATCH ratio, JPEG delta)
            if name in (
                "conf_drop", "conf_std", "area_stability", "aspect_ratio_stability",
                "flicker_rate", "laplacian_var_norm", "laplacian_entropy",
                "color_hue_entropy", "color_sat_entropy", "color_val_entropy",
                "max_grid_norm_ratio", "jpeg_recompression_delta",
            ):
                deviation = max(0.0, (val - mean_b) / (std_b + 1e-4))
            elif name in ("conf_current", "mean_consecutive_iou", "min_consecutive_iou"):
                # Features where lower values indicate attack:
                deviation = max(0.0, (mean_b - val) / (std_b + 1e-4))
            else:
                deviation = abs(val - mean_b) / (std_b + 1e-4)

            score = float(g_imp * deviation)
            contributions.append((name, round(score, 4)))

        # Sort by highest contribution descending
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:5]  # Top 5 explanatory features

    def predict_batch(
        self,
        feature_matrix: np.ndarray,
        delta_p_objs: Optional[Sequence[float]] = None,
        patch_rois_list: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,
        person_bboxes: Optional[Sequence[Optional[Sequence[float]]]] = None,
        recovered_counts: Optional[Sequence[int]] = None,
    ) -> List[AttackPrediction]:
        """Perform batch prediction across multiple tracks with optional causal context."""
        results = []
        n = len(feature_matrix)
        for i in range(n):
            vec = feature_matrix[i]
            d_p = delta_p_objs[i] if delta_p_objs is not None and i < len(delta_p_objs) else 0.0
            p_rois = patch_rois_list[i] if patch_rois_list is not None and i < len(patch_rois_list) else None
            p_box = person_bboxes[i] if person_bboxes is not None and i < len(person_bboxes) else None
            rec_cnt = recovered_counts[i] if recovered_counts is not None and i < len(recovered_counts) else 0
            results.append(
                self.predict(
                    vec,
                    delta_p_obj=d_p,
                    patch_rois=p_rois,
                    person_bbox=p_box,
                    recovered_objects_count=rec_cnt,
                )
            )
        return results
