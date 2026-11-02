
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

# Empirical clean clothing thresholds for spatial/compression anomaly rejection
CLEAN_CLOTHING_MAX_GRID_NORM_RATIO: float = 2.2
CLEAN_CLOTHING_JPEG_DELTA: float = 2.0


def check_is_near_frame_edge(
    box: Sequence[float],
    frame_shape: Sequence[int] = (640, 640),
    margin: float = 0.05,
) -> bool:
    """Check if bounding box touches or is near the outer frame boundary.

    Returns True if x1 <= margin*W, y1 <= margin*H, x2 >= (1-margin)*W, or y2 >= (1-margin)*H.
    """
    if len(box) < 4:
        return False
    h, w = float(frame_shape[0]), float(frame_shape[1])
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return bool(
        x1 <= margin * w
        or y1 <= margin * h
        or x2 >= (1.0 - margin) * w
        or y2 >= (1.0 - margin) * h
    )


is_near_frame_edge = check_is_near_frame_edge


def check_is_inside_torso(
    patch_box: Sequence[float],
    person_box: Sequence[float],
) -> bool:
    """Check if candidate patch ROI lies strictly inside the person's torso region.

    Returns True ONLY if the center of patch_box lies horizontally within
    [px1 + 0.15*pw, px2 - 0.15*pw] and vertically within [py1 + 0.15*ph, py2 - 0.20*ph],
    and the patch area is between 4% and 45% of the person box area.
    """
    if len(patch_box) < 4 or len(person_box) < 4:
        return False
    px1, py1, px2, py2 = float(person_box[0]), float(person_box[1]), float(person_box[2]), float(person_box[3])
    pw = max(0.0, px2 - px1)
    ph = max(0.0, py2 - py1)
    person_area = pw * ph
    if person_area <= 0.0:
        return False

    rx1, ry1, rx2, ry2 = float(patch_box[0]), float(patch_box[1]), float(patch_box[2]), float(patch_box[3])
    rw = max(0.0, rx2 - rx1)
    rh = max(0.0, ry2 - ry1)
    patch_area = rw * rh
    if patch_area <= 0.0:
        return False

    rcx = (rx1 + rx2) / 2.0
    rcy = (ry1 + ry2) / 2.0

    min_x = px1 + 0.15 * pw
    max_x = px2 - 0.15 * pw
    min_y = py1 + 0.15 * ph
    max_y = py2 - 0.20 * ph

    area_ratio = patch_area / person_area

    return bool(
        min_x <= rcx <= max_x
        and min_y <= rcy <= max_y
        and 0.04 <= area_ratio <= 0.45
    )


is_inside_torso = check_is_inside_torso


def is_in_central_frame(
    box: Sequence[float],
    frame_shape: Sequence[int] = (640, 640),
    central_fraction: float = 0.85,
) -> bool:
    """Check if bounding box center lies inside the central fraction of the frame."""
    if len(box) < 4:
        return True
    h, w = float(frame_shape[0]), float(frame_shape[1])
    margin_x = (1.0 - central_fraction) / 2.0 * w
    margin_y = (1.0 - central_fraction) / 2.0 * h
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return bool(margin_x <= cx <= (w - margin_x) and margin_y <= cy <= (h - margin_y))


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
    case_triggered: Optional[str] = None
    patch_conf: float = 0.0


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
        patch_model_path: Optional[Union[str, Path]] = "models/ad_yolo_best.pt",
        patch_model: Optional[Any] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.is_onnx = False
        self.patch_model_path = Path(patch_model_path) if patch_model_path is not None else Path("models/ad_yolo_best.pt")
        self.patch_model = patch_model

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

    def load_patch_model(self) -> Optional[Any]:
        """Lazily load fine-tuned Ad-YOLO patch detector (models/ad_yolo_best.pt)."""
        if self.patch_model is not None:
            return self.patch_model
        if self.patch_model_path is not None and self.patch_model_path.exists():
            try:
                from ultralytics import YOLO

                self.patch_model = YOLO(str(self.patch_model_path))
                logger.info("Loaded fine-tuned Ad-YOLO patch model from %s", self.patch_model_path)
            except Exception as exc:
                logger.warning("Could not load patch model from %s: %s", self.patch_model_path, exc)
                self.patch_model = None
        return self.patch_model

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

        # 2. From calibrated classifiers ensemble if present
        if hasattr(self.model, "calibrated_classifiers_"):
            imps = [
                getattr(c, "estimator", c).feature_importances_
                for c in self.model.calibrated_classifiers_
                if hasattr(getattr(c, "estimator", c), "feature_importances_")
            ]
            if imps:
                raw_imps = np.mean(imps, axis=0)
                total = float(np.sum(raw_imps))
                norm_imps = raw_imps / (total + 1e-6)
                return {
                    name: float(norm_imps[i])
                    for i, name in enumerate(FEATURE_NAMES)
                    if i < len(norm_imps)
                }

        # 3. If model is ONNX, check if companion joblib exists to read weights
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
                if hasattr(jb, "calibrated_classifiers_"):
                    imps = [
                        getattr(c, "estimator", c).feature_importances_
                        for c in jb.calibrated_classifiers_
                        if hasattr(getattr(c, "estimator", c), "feature_importances_")
                    ]
                    if imps:
                        raw_imps = np.mean(imps, axis=0)
                        total = float(np.sum(raw_imps))
                        norm_imps = raw_imps / (total + 1e-6)
                        return {
                            name: float(norm_imps[i])
                            for i, name in enumerate(FEATURE_NAMES)
                            if i < len(norm_imps)
                        }
            except Exception:
                pass

        # 4. Default equal baseline importance
        equal_weight = 1.0 / float(len(FEATURE_NAMES))
        return {name: equal_weight for name in FEATURE_NAMES}

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
        jpeg_recompression_delta: Optional[float] = None,
        class_name: str = "person",
        patch_conf: float = 0.0,
        patch_confs: Optional[Sequence[float]] = None,
        frame_shape: Sequence[int] = (640, 640),
        is_near_frame_edge: Optional[bool] = None,
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
            jpeg_recompression_delta: Optional JPEG recompression energy delta.
            class_name: Detected object class name (must be 'person' to flag attack).
            patch_conf: Detection confidence of candidate patch from models/ad_yolo_best.pt.
            patch_confs: Sequence of detection confidences corresponding to candidate patch ROIs.
            frame_shape: Frame dimensions (H, W) used for boundary margin checks.
            is_near_frame_edge: Explicit override for boundary check.

        Returns:
            AttackPrediction containing attack flag, probability, and top contributing features.
        """
        # 1. Input normalization and contextual parameter extraction
        t_start = time.perf_counter()

        resolved_person_bbox = person_bbox
        resolved_delta_p_obj = float(delta_p_obj)
        resolved_class_name = str(class_name)
        resolved_patch_conf = float(patch_conf)

        all_patch_rois: List[Sequence[float]] = []
        if patch_roi is not None:
            all_patch_rois.append(patch_roi)
            if len(patch_roi) >= 5 and resolved_patch_conf == 0.0:
                resolved_patch_conf = float(patch_roi[4])
        if patch_rois:
            for roi in patch_rois:
                all_patch_rois.append(roi)
                if len(roi) >= 5 and resolved_patch_conf == 0.0:
                    resolved_patch_conf = max(resolved_patch_conf, float(roi[4]))

        if patch_confs:
            resolved_patch_conf = max(resolved_patch_conf, max(float(c) for c in patch_confs))

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
            if jpeg_recompression_delta is None and "jpeg_recompression_delta" in features.feature_dict:
                jpeg_recompression_delta = float(features.feature_dict["jpeg_recompression_delta"])
            if hasattr(features, "class_name"):
                resolved_class_name = str(getattr(features, "class_name"))
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

        # Extract contextual spatial/compression features
        if max_grid_norm_ratio is None:
            resolved_norm_ratio = float(sanitized_vec[13]) if len(sanitized_vec) > 13 else 1.0
        else:
            resolved_norm_ratio = float(max_grid_norm_ratio)

        if jpeg_recompression_delta is None:
            resolved_jpeg_delta = float(sanitized_vec[14]) if len(sanitized_vec) > 14 else 1.0
        else:
            resolved_jpeg_delta = float(jpeg_recompression_delta)

        # Model feature dimension alignment
        expected_features = 15
        if self.is_onnx:
            in_shape = self.model.get_inputs()[0].shape
            if len(in_shape) > 1 and isinstance(in_shape[1], int):
                expected_features = in_shape[1]
        elif hasattr(self.model, "n_features_in_"):
            expected_features = self.model.n_features_in_
        elif hasattr(self.model, "calibrated_classifiers_"):
            first_est = getattr(self.model.calibrated_classifiers_[0], "estimator", self.model.calibrated_classifiers_[0])
            if hasattr(first_est, "n_features_in_"):
                expected_features = first_est.n_features_in_

        if input_2d.shape[1] < expected_features:
            padded = np.zeros((1, expected_features), dtype=np.float32)
            padded[0, :input_2d.shape[1]] = input_2d[0]
            if expected_features >= 15 and input_2d.shape[1] == 13:
                padded[0, 13] = resolved_norm_ratio
                padded[0, 14] = resolved_jpeg_delta
            model_input = padded
        else:
            model_input = input_2d[:, :expected_features]

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

        resolved_recovered_count = int(recovered_objects_count)
        if resolved_recovered_count == 0 and resolved_delta_p_obj >= 0.15:
            resolved_recovered_count = 1

        top_contributors = self._compute_feature_contributions(sanitized_vec)

        # ====================================================================
        # 3. Two-Stage Causal Verification Gate
        # ====================================================================

        # Rule 1: Never flag is_attack = True on any non-person object (class_name != 'person')
        if resolved_class_name.strip().lower() != "person":
            return AttackPrediction(
                is_attack=False,
                attack_probability=round(p_attack, 4),
                confidence_clean=round(p_clean, 4),
                feature_importances=self.global_importances,
                top_contributing_features=top_contributors,
                inference_latency_ms=round(t_elapsed_ms, 4),
                causal_verification_passed=False,
                delta_p_obj=round(resolved_delta_p_obj, 4),
                case_triggered=None,
                patch_conf=round(resolved_patch_conf, 4),
            )

        # Rule 2: Never flag is_attack = True on any person box where is_near_frame_edge is True
        if is_near_frame_edge is not None:
            near_edge = bool(is_near_frame_edge)
        elif resolved_person_bbox is not None:
            near_edge = check_is_near_frame_edge(resolved_person_bbox, frame_shape, margin=0.05)
        else:
            near_edge = False

        if near_edge:
            return AttackPrediction(
                is_attack=False,
                attack_probability=round(p_attack, 4),
                confidence_clean=round(p_clean, 4),
                feature_importances=self.global_importances,
                top_contributing_features=top_contributors,
                inference_latency_ms=round(t_elapsed_ms, 4),
                causal_verification_passed=False,
                delta_p_obj=round(resolved_delta_p_obj, 4),
                case_triggered=None,
                patch_conf=round(resolved_patch_conf, 4),
            )

        # Rule 3: For an interior person detection, set is_attack = True ONLY if a valid patch_roi
        # satisfies is_inside_torso AND meets at least one of these three conditions:
        # Case A: delta_p_obj >= 0.15 after Pass 2 Telea inpainting
        # Case B: patch_roi detected by models/ad_yolo_best.pt (patch_conf >= 0.45) AND (delta_p_obj >= 0.05 OR classifier_prob >= 0.65)
        # Case C: classifier_prob >= 0.82 AND (jpeg_recompression_delta / max_grid_norm_ratio) exceed clean clothing thresholds
        if resolved_person_bbox is not None and all_patch_rois:
            valid_torso_patches = [
                roi for roi in all_patch_rois
                if check_is_inside_torso(roi, resolved_person_bbox)
            ]
            has_valid_torso_patch = len(valid_torso_patches) > 0
        elif resolved_person_bbox is None and not all_patch_rois:
            # Fallback when evaluated solely on 1D feature vectors (e.g. unit tests / benchmarks)
            has_valid_torso_patch = True
        else:
            # Either person bbox is given without candidate patches, or patch ROIs are given without a person (background candidate)
            has_valid_torso_patch = False

        case_a = bool(resolved_delta_p_obj >= 0.15)
        case_b = bool(resolved_patch_conf >= 0.45 and (resolved_delta_p_obj >= 0.05 or p_attack >= 0.65))
        clean_clothing_threshold_exceeded = bool(
            resolved_norm_ratio > CLEAN_CLOTHING_MAX_GRID_NORM_RATIO
            or resolved_jpeg_delta > CLEAN_CLOTHING_JPEG_DELTA
        )
        case_c = bool(p_attack >= 0.82 and clean_clothing_threshold_exceeded)

        # Case D covers optical patches that suppress the detector but are not
        # localized by the spatial clipper. Require both a confident model
        # prediction and a strong temporal suppression signature so ordinary
        # patterned clothing does not become an alert.
        conf_drop_value = float(conf_drop if conf_drop is not None else sanitized_vec[1])
        flicker_value = float(sanitized_vec[7])
        case_d = bool(
            p_attack >= self.threshold
            and (conf_drop_value >= 0.25 or flicker_value >= 0.15)
        )

        local_torso_gate = bool(has_valid_torso_patch and (case_a or case_b or case_c))
        temporal_suppression_gate = bool(
            resolved_person_bbox is not None
            and not has_valid_torso_patch
            and case_d
        )

        # Rule 4: Disable global background DPATCH alerts on live streams unless
        # max_grid_norm_ratio > 5.0 inside the central 85% of the frame
        # AND Pass 2 recovers >= 1 suppressed object with delta_p_obj > 0.35.
        if all_patch_rois:
            in_central_85 = any(is_in_central_frame(roi, frame_shape, 0.85) for roi in all_patch_rois)
        elif resolved_person_bbox is not None:
            in_central_85 = is_in_central_frame(resolved_person_bbox, frame_shape, 0.85)
        else:
            in_central_85 = True

        global_dpatch_gate = bool(
            resolved_norm_ratio > 5.0
            and in_central_85
            and resolved_recovered_count >= 1
            and resolved_delta_p_obj > 0.35
        )

        causal_gate_passed = bool(local_torso_gate or temporal_suppression_gate or global_dpatch_gate)
        is_attack = causal_gate_passed

        case_triggered: Optional[str] = None
        if local_torso_gate:
            if case_a:
                case_triggered = "Case A (Causal Objectness Recovery)"
            elif case_b:
                case_triggered = "Case B (Fine-Tuned Ad-YOLO Match)"
            elif case_c:
                case_triggered = "Case C (High-Confidence Zero-Shot Anomaly)"
        elif global_dpatch_gate:
            case_triggered = "Global DPATCH Gate"
        elif temporal_suppression_gate:
            case_triggered = "Case D (Temporal Suppression)"

        return AttackPrediction(
            is_attack=is_attack,
            attack_probability=round(p_attack, 4),
            confidence_clean=round(p_clean, 4),
            feature_importances=self.global_importances,
            top_contributing_features=top_contributors,
            inference_latency_ms=round(t_elapsed_ms, 4),
            causal_verification_passed=causal_gate_passed,
            delta_p_obj=round(resolved_delta_p_obj, 4),
            case_triggered=case_triggered,
            patch_conf=round(resolved_patch_conf, 4),
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
        class_names: Optional[Sequence[str]] = None,
        patch_confs: Optional[Sequence[float]] = None,
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
            c_name = class_names[i] if class_names is not None and i < len(class_names) else "person"
            p_conf = patch_confs[i] if patch_confs is not None and i < len(patch_confs) else 0.0
            results.append(
                self.predict(
                    vec,
                    delta_p_obj=d_p,
                    patch_rois=p_rois,
                    person_bbox=p_box,
                    recovered_objects_count=rec_cnt,
                    class_name=c_name,
                    patch_conf=p_conf,
                )
            )
        return results
