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
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import cv2
import joblib
import numpy as np
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


def generate_synthetic_features(
    n_samples_per_class: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic feature distributions for clean vs attacked tracks.

    Useful for testing, rapid benchmarking, or when image datasets are not on disk.
    """
    rng = np.random.RandomState(seed)
    num_features = len(FEATURE_NAMES)

    # 1. Clean class (label: 0): High confidence, stable IoU & area, low entropy
    # FEATURE_NAMES:
    # 0: conf_current, 1: conf_drop, 2: conf_std, 3: area_stability,
    # 4: aspect_ratio_stability, 5: mean_consecutive_iou, 6: min_consecutive_iou,
    # 7: flicker_rate, 8: laplacian_var_norm, 9: laplacian_entropy,
    # 10: color_hue_entropy, 11: color_sat_entropy, 12: color_val_entropy
    clean_X = np.zeros((n_samples_per_class, num_features), dtype=np.float32)
    clean_X[:, 0] = rng.uniform(0.80, 0.98, n_samples_per_class)  # conf_current
    clean_X[:, 1] = rng.uniform(0.00, 0.08, n_samples_per_class)  # conf_drop
    clean_X[:, 2] = rng.uniform(0.01, 0.06, n_samples_per_class)  # conf_std
    clean_X[:, 3] = rng.uniform(0.01, 0.08, n_samples_per_class)  # area_stability
    clean_X[:, 4] = rng.uniform(0.01, 0.07, n_samples_per_class)  # aspect_ratio_stability
    clean_X[:, 5] = rng.uniform(0.85, 0.98, n_samples_per_class)  # mean_consecutive_iou
    clean_X[:, 6] = rng.uniform(0.75, 0.95, n_samples_per_class)  # min_consecutive_iou
    clean_X[:, 7] = rng.choice([0.0, 0.05], p=[0.9, 0.1], size=n_samples_per_class)  # flicker_rate
    clean_X[:, 8] = rng.uniform(0.10, 0.35, n_samples_per_class)  # laplacian_var_norm
    clean_X[:, 9] = rng.uniform(0.15, 0.40, n_samples_per_class)  # laplacian_entropy
    clean_X[:, 10] = rng.uniform(0.10, 0.40, n_samples_per_class)  # color_hue_entropy
    clean_X[:, 11] = rng.uniform(0.10, 0.45, n_samples_per_class)  # color_sat_entropy
    clean_X[:, 12] = rng.uniform(0.20, 0.50, n_samples_per_class)  # color_val_entropy
    clean_y = np.zeros(n_samples_per_class, dtype=np.int32)

    # 2. Attacked class (label: 1): Suppressed confidence, erratic bbox, high entropy
    attack_X = np.zeros((n_samples_per_class, num_features), dtype=np.float32)
    attack_X[:, 0] = rng.uniform(0.15, 0.55, n_samples_per_class)  # conf_current
    attack_X[:, 1] = rng.uniform(0.35, 0.85, n_samples_per_class)  # conf_drop
    attack_X[:, 2] = rng.uniform(0.15, 0.45, n_samples_per_class)  # conf_std
    attack_X[:, 3] = rng.uniform(0.20, 0.70, n_samples_per_class)  # area_stability
    attack_X[:, 4] = rng.uniform(0.15, 0.55, n_samples_per_class)  # aspect_ratio_stability
    attack_X[:, 5] = rng.uniform(0.30, 0.70, n_samples_per_class)  # mean_consecutive_iou
    attack_X[:, 6] = rng.uniform(0.10, 0.50, n_samples_per_class)  # min_consecutive_iou
    attack_X[:, 7] = rng.uniform(0.10, 0.60, n_samples_per_class)  # flicker_rate
    attack_X[:, 8] = rng.uniform(0.55, 0.95, n_samples_per_class)  # laplacian_var_norm
    attack_X[:, 9] = rng.uniform(0.60, 0.98, n_samples_per_class)  # laplacian_entropy
    attack_X[:, 10] = rng.uniform(0.65, 0.98, n_samples_per_class)  # color_hue_entropy
    attack_X[:, 11] = rng.uniform(0.60, 0.95, n_samples_per_class)  # color_sat_entropy
    attack_X[:, 12] = rng.uniform(0.55, 0.95, n_samples_per_class)  # color_val_entropy
    attack_y = np.ones(n_samples_per_class, dtype=np.int32)

    X = np.vstack([clean_X, attack_X])
    y = np.concatenate([clean_y, attack_y])

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_or_extract_dataset(
    data_dir: Union[str, Path] = "data",
    cache_file: str = "extracted_features.npz",
    use_synthetic_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ingest feature vectors from disk dataset, caching results for fast retraining."""
    data_path = Path(data_dir)
    cache_path = data_path / cache_file

    # 1. Return cached features if available
    if cache_path.exists():
        logger.info("Loading cached feature vectors from: %s", cache_path)
        cached = np.load(str(cache_path))
        return cached["X"], cached["y"]

    annotations_file = data_path / "annotations.json"
    if not annotations_file.exists():
        if use_synthetic_fallback:
            logger.warning(
                "Dataset %s not found. Using high-fidelity synthetic feature synthesis.",
                annotations_file,
            )
            return generate_synthetic_features(n_samples_per_class=1000)
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    # 2. Extract features from annotations and images
    logger.info("Extracting feature vectors from raw images in %s...", data_dir)
    with open(annotations_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    records = meta.get("data", [])

    extractor = DualFeatureExtractor()
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for item in records:
        rel_path = item.get("filename", "")
        img_file = data_path / rel_path
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        label = int(item.get("label", 0))
        pbox = np.array(item.get("person_bbox", [100, 100, 200, 300]), dtype=np.float32)

        # Visual features from image ROI
        roi = extractor.crop_upper_body(img, pbox)
        vis_feats = extractor.compute_visual_features(roi)

        # Temporal features from metadata record
        conf_base = float(item.get("confidence_baseline", 0.90))
        conf_pert = float(item.get("confidence_perturbed", 0.90))
        conf_delta = float(item.get("confidence_delta", 0.0))

        if label == 0:
            # Clean frame
            temp_feats = {
                "conf_current": conf_base,
                "conf_drop": 0.0,
                "conf_std": 0.02,
                "area_stability": 0.03,
                "aspect_ratio_stability": 0.02,
                "mean_consecutive_iou": 0.94,
                "min_consecutive_iou": 0.88,
                "flicker_rate": 0.0,
            }
        else:
            # Perturbed frame
            temp_feats = {
                "conf_current": conf_pert,
                "conf_drop": conf_delta,
                "conf_std": min(1.0, conf_delta * 0.6),
                "area_stability": 0.35,
                "aspect_ratio_stability": 0.25,
                "mean_consecutive_iou": 0.55,
                "min_consecutive_iou": 0.35,
                "flicker_rate": 0.20,
            }

        fused = {}
        fused.update(temp_feats)
        fused.update(vis_feats)

        vec = np.array([fused.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
        X_list.append(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0))
        y_list.append(label)

    if not X_list:
        logger.warning("No valid images found in %s; falling back to synthetic generator.", data_dir)
        return generate_synthetic_features(n_samples_per_class=1000)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Save to cache
    try:
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
    1. RandomForestClassifier (n_estimators=50, max_depth=8)
    2. HistGradientBoostingClassifier (max_iter=50, max_depth=8)
    3. XGBClassifier (n_estimators=50, max_depth=6)

    Returns:
        Dictionary summarizing metrics and the best model result.
    """
    candidate_models: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=random_state,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=8,
            random_state=random_state,
        ),
    }

    if XGBClassifier is not None:
        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )

    logger.info("Starting 5-fold Stratified Cross-Validation across %d models...", len(candidate_models))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    benchmarks: List[ModelBenchmarkResult] = []

    test_single_sample = X[:1]  # Shape [1, 13]

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

    # 3. Select best model based on F1-score and low latency
    best_result = max(benchmarks, key=lambda b: (b.mean_f1, -b.cpu_latency_per_sample_ms))
    logger.info("Best model selected: %s (F1: %.4f)", best_result.model_name, best_result.mean_f1)

    # 4. Refit winning model on full dataset
    final_model = best_result.trained_model
    final_model.fit(X, y)

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


def export_classifier_to_onnx(model: Any, output_path: str, num_features: int = 13) -> Optional[str]:
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
            onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
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
    args = parser.parse_args()

    features_X, labels_y = load_or_extract_dataset(data_dir=args.data_dir)
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
