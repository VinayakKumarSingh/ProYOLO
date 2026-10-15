"""CPU-Optimized YOLOv8 ONNX Object Detector for ProYOLO.

Uses ONNX Runtime with CPUExecutionProvider (or OpenVINOExecutionProvider)
with multi-threaded graph optimizations and letterbox preprocessing to achieve
15+ FPS on standard multi-core CPUs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

logger = logging.getLogger(__name__)

# Standard 80 COCO dataset class labels
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class Detection(NamedTuple):
    """Container representing a single detected object."""

    bbox: np.ndarray  # Format: [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        """Bounding box surface area in pixels."""
        w = max(0.0, float(self.bbox[2] - self.bbox[0]))
        h = max(0.0, float(self.bbox[3] - self.bbox[1]))
        return w * h

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio."""
        w = max(1e-4, float(self.bbox[2] - self.bbox[0]))
        h = max(1e-4, float(self.bbox[3] - self.bbox[1]))
        return w / h


class YOLOv8ONNXDetector:
    """CPU-Optimized YOLOv8 Detector using ONNX Runtime.

    Attributes:
        model_path: Path to the exported yolov8n.onnx model.
        input_size: Input spatial dimensions (width, height) fed to the neural network.
        conf_threshold: Minimum objectness/class confidence to retain proposals.
        iou_threshold: Intersection over Union threshold for Non-Maximum Suppression (NMS).
        target_classes: Sequence of class IDs to filter for (default [0] for person).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.onnx",
        input_size: Tuple[int, int] = (480, 480),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[Sequence[int]] = (0,),
        num_threads: Optional[int] = None,
        provider: str = "CPUExecutionProvider",
        session: Optional[Any] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.provider = provider

        if session is not None:
            self.session = session
        else:
            self.session = self._init_session(num_threads)

        # Inspect model input/output signatures
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name

    def _init_session(self, num_threads: Optional[int]) -> Any:
        """Initialize ONNX Runtime inference session with multi-threaded CPU optimizations."""
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Please install it with 'pip install onnxruntime'."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

        opts = ort.SessionOptions()
        threads = num_threads or os.cpu_count() or 4
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        selected_providers = []
        if self.provider in available_providers:
            selected_providers.append(self.provider)
        if "CPUExecutionProvider" not in selected_providers:
            selected_providers.append("CPUExecutionProvider")

        logger.info(
            "Initializing ONNX Runtime session for %s with %d threads on %s",
            self.model_path,
            threads,
            selected_providers,
        )
        return ort.InferenceSession(str(self.model_path), opts, providers=selected_providers)

    @staticmethod
    def letterbox(
        image: np.ndarray,
        target_size: Tuple[int, int] = (480, 480),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize image with constant aspect ratio and symmetric gray border padding."""
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        pad_w = (target_w - new_w) / 2.0
        pad_h = (target_h - new_h) / 2.0

        if (orig_w, orig_h) != (new_w, new_h):
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, scale, (pad_w, pad_h)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Execute object detection on a BGR video frame.

        Args:
            frame: Input image array [H, W, 3] in BGR color space.

        Returns:
            List of filtered, NMS-suppressed Detection objects.
        """
        orig_h, orig_w = frame.shape[:2]
        padded_img, scale, (pad_x, pad_y) = self.letterbox(frame, target_size=self.input_size)

        # Preprocessing: BGR -> RGB, HWC -> CHW, normalize to [0.0, 1.0]
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_tensor = np.ascontiguousarray(input_tensor[np.newaxis, ...])

        # ONNX Runtime forward inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_output = outputs[0]  # Shape: [1, 84, N_anchors] or [1, N_anchors, 84]

        # Post-process proposals
        return self._postprocess(raw_output, orig_w, orig_h, scale, pad_x, pad_y)

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Detection]:
        """Convert raw YOLOv8 prediction tensor into rescale bounding boxes with NMS."""
        if raw_output.ndim == 3 and raw_output.shape[1] == 84:
            # Transpose [1, 84, N] -> [N, 84]
            predictions = raw_output[0].T
        else:
            predictions = raw_output[0]

        # 1. Coordinate extraction: [cx, cy, w, h]
        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]

        # Convert center coordinates to corner coordinates [x1, y1, x2, y2]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Invert letterbox transformations to map back to original frame size
        orig_x1 = np.clip((x1 - pad_x) / scale, 0.0, orig_w)
        orig_y1 = np.clip((y1 - pad_y) / scale, 0.0, orig_h)
        orig_x2 = np.clip((x2 - pad_x) / scale, 0.0, orig_w)
        orig_y2 = np.clip((y2 - pad_y) / scale, 0.0, orig_h)

        # 2. Extract class scores (columns 4..83)
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        best_classes = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold and target classes
        mask = max_scores >= self.conf_threshold
        if self.target_classes is not None:
            class_mask = np.isin(best_classes, list(self.target_classes))
            mask = mask & class_mask

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return []

        # 3. Non-Maximum Suppression (NMS)
        boxes_for_nms: List[List[int]] = []
        scores_for_nms: List[float] = []
        classes_for_nms: List[int] = []
        coords_for_nms: List[np.ndarray] = []

        for idx in filtered_indices:
            bx1 = float(orig_x1[idx])
            by1 = float(orig_y1[idx])
            bx2 = float(orig_x2[idx])
            by2 = float(orig_y2[idx])
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)

            boxes_for_nms.append([int(bx1), int(by1), int(bw), int(bh)])
            scores_for_nms.append(float(max_scores[idx]))
            classes_for_nms.append(int(best_classes[idx]))
            coords_for_nms.append(np.array([bx1, by1, bx2, by2], dtype=np.float32))

        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_nms,
            scores=scores_for_nms,
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections: List[Detection] = []
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                cls_id = classes_for_nms[i]
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                detections.append(
                    Detection(
                        bbox=coords_for_nms[i],
                        confidence=scores_for_nms[i],
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return detections
