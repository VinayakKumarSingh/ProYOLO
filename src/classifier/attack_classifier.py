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
}


class AttackPrediction(NamedTuple):
    """Result of real-time attack classifier inference."""

    is_attack: bool
    attack_probability: float
    confidence_clean: float
    feature_importances: Dict[str, float]
    top_contributing_features: List[Tuple[str, float]]
    inference_latency_ms: float


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

    def predict(
        self,
        features: Union[np.ndarray, FeaturePacket, Sequence[float]],
    ) -> AttackPrediction:
        """Predict attack probability and generate alert explanation for a single track.

        Args:
            features: 1D feature array of shape (13,), FeaturePacket, or sequence of floats.

        Returns:
            AttackPrediction containing attack flag, probability, and top contributing features.
        """
        # 1. Input normalization
        t_start = time.perf_counter()

        if isinstance(features, FeaturePacket):
            raw_vec = features.feature_vector
        elif isinstance(features, np.ndarray):
            raw_vec = features.flatten()
        else:
            raw_vec = np.array(features, dtype=np.float32).flatten()

        if len(raw_vec) != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature vector length mismatch: expected {len(FEATURE_NAMES)} features, got {len(raw_vec)}."
            )

        sanitized_vec = np.nan_to_num(raw_vec, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        input_2d = sanitized_vec.reshape(1, -1)

        # 2. Forward model inference
        if self.is_onnx:
            input_name = self.model.get_inputs()[0].name
            ort_outs = self.model.run(None, {input_name: input_2d})
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
                probs = self.model.predict_proba(input_2d)[0]
                p_clean = float(probs[0])
                p_attack = float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0])
            else:
                pred_label = int(self.model.predict(input_2d)[0])
                p_attack = 1.0 if pred_label == 1 else 0.0
                p_clean = 1.0 - p_attack

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 3. Decision thresholding
        is_attack = bool(p_attack >= self.threshold)

        # 4. Explainability & Feature Contribution Attribution
        top_contributors = self._compute_feature_contributions(sanitized_vec)

        return AttackPrediction(
            is_attack=is_attack,
            attack_probability=round(p_attack, 4),
            confidence_clean=round(p_clean, 4),
            feature_importances=self.global_importances,
            top_contributing_features=top_contributors,
            inference_latency_ms=round(t_elapsed_ms, 4),
        )

    def _compute_feature_contributions(self, feature_vec: np.ndarray) -> List[Tuple[str, float]]:
        """Compute relative contribution scores of features explaining why the track was flagged."""
        contributions: List[Tuple[str, float]] = []

        for i, name in enumerate(FEATURE_NAMES):
            val = float(feature_vec[i])
            mean_b, std_b = CLEAN_FEATURE_BASELINES.get(name, (0.5, 0.1))
            g_imp = self.global_importances.get(name, 1.0 / len(FEATURE_NAMES))

            # Features where higher values indicate an attack:
            # (conf_drop, entropy, variance, flicker, area_stability)
            if name in (
                "conf_drop", "conf_std", "area_stability", "aspect_ratio_stability",
                "flicker_rate", "laplacian_var_norm", "laplacian_entropy",
                "color_hue_entropy", "color_sat_entropy", "color_val_entropy",
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
    ) -> List[AttackPrediction]:
        """Perform batch prediction across multiple tracks."""
        return [self.predict(vec) for vec in feature_matrix]
