"""ProYOLO Lightweight Attack Classifier and Training Engine."""

from src.classifier.attack_classifier import (
    AttackClassifier,
    AttackPrediction,
    is_inside_torso,
    is_near_frame_edge,
)
from src.classifier.train_classifier import (
    ModelBenchmarkResult,
    load_or_extract_dataset,
    train_and_benchmark_models,
)

__all__ = [
    "AttackClassifier",
    "AttackPrediction",
    "is_near_frame_edge",
    "is_inside_torso",
    "ModelBenchmarkResult",
    "load_or_extract_dataset",
    "train_and_benchmark_models",
]
