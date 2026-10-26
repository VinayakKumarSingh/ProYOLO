"""ProYOLO Detection, Tracking, and Feature Extraction Engine."""

from src.detection.detector import (
    Detection,
    UltralyticsPyTorchWrapper,
    YOLOv8ONNXDetector,
    YOLOv8PyTorchDetector,
    neutralize_patch_regions,
)
from src.detection.feature_extractor import (
    DualFeatureExtractor,
    FeaturePacket,
    SpatialFeatureNormClipper,
)
from src.detection.tracker import ByteTracker, Track, TrackState

__all__ = [
    "Detection",
    "YOLOv8ONNXDetector",
    "YOLOv8PyTorchDetector",
    "UltralyticsPyTorchWrapper",
    "neutralize_patch_regions",
    "ByteTracker",
    "Track",
    "TrackState",
    "DualFeatureExtractor",
    "FeaturePacket",
    "SpatialFeatureNormClipper",
]
