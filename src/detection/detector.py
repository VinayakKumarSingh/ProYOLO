"""CPU-Optimized YOLOv8 ONNX and PyTorch Object Detector for ProYOLO.

Uses ONNX Runtime (or Ultralytics PyTorch) with multi-threaded graph optimizations,
letterbox preprocessing, and an Active Two-Pass "Detect-Neutralize-Redetect" pipeline
to counter Thys et al. adversarial patch objectness suppression.
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
    delta_p_obj: float = 0.0
    recovered_by_inpainting: bool = False

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


from src.detection.feature_extractor import SpatialFeatureNormClipper


def bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def box_contains(outer: Sequence[float], inner: Sequence[float], margin_ratio: float = 0.05) -> bool:
    """Check if outer bounding box contains inner bounding box (with optional tolerance margin)."""
    ow = max(1.0, float(outer[2] - outer[0]))
    oh = max(1.0, float(outer[3] - outer[1]))
    mx = ow * margin_ratio
    my = oh * margin_ratio
    return bool(
        (float(outer[0]) - mx) <= float(inner[0])
        and (float(outer[1]) - my) <= float(inner[1])
        and (float(outer[2]) + mx) >= float(inner[2])
        and (float(outer[3]) + my) >= float(inner[3])
    )


def box_overlaps(box1: Sequence[float], box2: Sequence[float]) -> bool:
    """Check if two bounding boxes have positive spatial intersection."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    return bool(x2 > x1 and y2 > y1)


def neutralize_patch_regions(
    frame: np.ndarray,
    patch_rois: Sequence[Sequence[float]],
    dilate_ratio: float = 0.10,
) -> np.ndarray:
    """Inpaint adversarial patch regions using surrounding visual context.

    Args:
        frame: Input image array [H, W, 3] in BGR color space.
        patch_rois: Sequence of candidate patch bounding boxes [x1, y1, x2, y2].
        dilate_ratio: Percentage to expand patch boxes before inpainting (default: 0.10).

    Returns:
        Inpainted BGR image where patch regions are smoothly blended with surrounding pixels.
    """
    if frame is None or frame.size == 0 or not patch_rois:
        return frame.copy() if frame is not None else frame

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for roi in patch_rois:
        if len(roi) < 4:
            continue
        rx1, ry1, rx2, ry2 = float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])
        rw = max(1.0, rx2 - rx1)
        rh = max(1.0, ry2 - ry1)

        dw = rw * dilate_ratio
        dh = rh * dilate_ratio

        dx1 = max(0, int(round(rx1 - dw)))
        dy1 = max(0, int(round(ry1 - dh)))
        dx2 = min(w, int(round(rx2 + dw)))
        dy2 = min(h, int(round(ry2 + dh)))

        if dx2 > dx1 and dy2 > dy1:
            mask[dy1:dy2, dx1:dx2] = 255

    if not np.any(mask):
        return frame.copy()

    return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


class YOLOv8ONNXDetector:
    """CPU-Optimized YOLOv8 Detector using ONNX Runtime.

    Attributes:
        model_path: Path to the exported yolov8n.onnx model.
        input_size: Input spatial dimensions (width, height) fed to the neural network.
        conf_threshold: Minimum objectness/class confidence to retain proposals.
        iou_threshold: Intersection over Union threshold for Non-Maximum Suppression (NMS).
        target_classes: Sequence of class IDs to filter for (default [0] for person).
        clipper: SpatialFeatureNormClipper forward hook instance.
    """

    neutralize_patch_regions = staticmethod(neutralize_patch_regions)

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
        model: Optional[Any] = None,
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.provider = provider

        # 1. Register SpatialFeatureNormClipper hook onto YOLO model during initialization
        self.clipper = clipper or SpatialFeatureNormClipper(image_size=self.input_size)
        self.model = model

        # If a PyTorch YOLO model or .pt path is provided, attach the clipper hook
        if self.model is not None:
            self._register_clipper(self.model)
        elif self.model_path.suffix.lower() == ".pt" and self.model_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
                self._register_clipper(self.model)
            except Exception as exc:
                logger.warning("Could not auto-load PyTorch model for hook registration: %s", exc)

        # Initialize ONNX session if session or onnx model path
        if session is not None:
            self.session = session
        elif self.model_path.suffix.lower() != ".pt" and self.model_path.exists():
            self.session = self._init_session(num_threads)
        elif self.model is None:
            self.session = self._init_session(num_threads)
        else:
            self.session = None

        if self.session is not None and hasattr(self.session, "get_inputs"):
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.output_name = self.session.get_outputs()[0].name
        else:
            self.input_name = "images"
            self.output_name = "output"

    def _register_clipper(self, model: Any) -> None:
        """Register the SpatialFeatureNormClipper forward hook onto the YOLO model."""
        if self.clipper is None or model is None:
            return
        try:
            self.clipper.attach(model)
            if (
                hasattr(model, "predictor")
                and model.predictor is not None
                and hasattr(model.predictor, "model")
            ):
                try:
                    self.clipper.attach(model.predictor.model)
                except Exception:
                    pass
            logger.info("SpatialFeatureNormClipper hook registered on %s", type(model).__name__)
        except Exception as exc:
            logger.warning(
                "Failed to register SpatialFeatureNormClipper hook onto %s: %s",
                type(model).__name__,
                exc,
            )

    def register_clipper(self, model: Any) -> None:
        """Public alias to register/attach SpatialFeatureNormClipper hook onto a YOLO model."""
        self.model = model
        self._register_clipper(model)

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

    def _detect_pytorch(
        self, frame: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """Execute detection using PyTorch YOLO model."""
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        classes = list(self.target_classes) if self.target_classes is not None else None
        results = self.model.predict(
            source=frame,
            conf=thresh,
            iou=self.iou_threshold,
            classes=classes,
            device="cpu",
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for box, conf, cid in zip(xyxy, confs, cls_ids):
            c_int = int(cid)
            name = COCO_CLASSES[c_int] if c_int < len(COCO_CLASSES) else f"class_{c_int}"
            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=c_int,
                    class_name=name,
                )
            )
        return detections

    def detect(
        self, frame: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """Execute object detection on a BGR video frame.

        Args:
            frame: Input image array [H, W, 3] in BGR color space.
            conf_threshold: Optional override for confidence threshold.

        Returns:
            List of filtered, NMS-suppressed Detection objects.
        """
        if self.clipper is not None:
            self.clipper.set_image(frame)

        if self.session is None and self.model is not None:
            return self._detect_pytorch(frame, conf_threshold=conf_threshold)

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
        return self._postprocess(
            raw_output, orig_w, orig_h, scale, pad_x, pad_y, conf_threshold=conf_threshold
        )

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Convert raw YOLOv8 prediction tensor into rescale bounding boxes with NMS."""
        active_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

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
        mask = max_scores >= active_conf
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
            score_threshold=active_conf,
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

    def detect_with_recovery(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.4,
        low_conf_thresh: float = 0.12,
        patch_rois: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Detection]:
        """Active Two-Pass 'Detect-Neutralize-Redetect' pipeline.

        Recovers persons missed due to Thys et al. adversarial patch objectness suppression:
        1. Pass 1: Run YOLO inference on raw frame using low_conf_thresh.
           Separate high-confidence person detections (conf >= conf_thresh),
           low-confidence suppressed candidates (low_conf_thresh <= conf < conf_thresh),
           and candidate patch_rois (from dedicated 'patch' class or feature_extractor anomaly ROIs).
        2. Trigger Condition: If patch_rois is non-empty AND (any patch_roi does not have
           a high-confidence person box containing it, OR overlaps with a low-confidence candidate):
        3. Pass 2: Inpaint patch regions and re-run YOLO inference on the neutralized frame.
        4. Calculate Causal Objectness Recovery (\\Delta P_{obj}): For each person detected in Pass 2,
           match with Pass 1 boxes via IoU: delta_p_obj = conf_pass2 - conf_pass1 (0.0 if missed).
        5. Merge Pass 1 and Pass 2 person detections using NMS, attaching delta_p_obj and
           recovered_by_inpainting to detection metadata.
        """
        # Pass 1: Low-threshold proposal generation
        pass1_all = self.detect(frame, conf_threshold=low_conf_thresh)

        # Separate persons by confidence tier
        high_conf_persons = [
            d for d in pass1_all
            if (d.class_id == 0 or d.class_name == "person") and d.confidence >= conf_thresh
        ]
        low_conf_persons = [
            d for d in pass1_all
            if (d.class_id == 0 or d.class_name == "person")
            and low_conf_thresh <= d.confidence < conf_thresh
        ]

        # Collect candidate patch ROIs
        all_patch_rois: List[List[float]] = []
        if patch_rois is not None:
            all_patch_rois.extend([list(r) for r in patch_rois])

        if hasattr(self, "clipper") and self.clipper is not None and self.clipper.patch_rois:
            all_patch_rois.extend([list(r) for r in self.clipper.patch_rois])

        for det in pass1_all:
            if det.class_name.lower() in ("patch", "adversarial_patch") or det.class_id == 80:
                all_patch_rois.append(det.bbox.tolist())

        # Trigger Condition:
        # If patch_rois is non-empty AND (any patch_roi does not have a high-confidence person box
        # containing it, OR overlaps with a low-confidence suppressed person candidate)
        should_trigger = False
        if all_patch_rois:
            for roi in all_patch_rois:
                contained_in_high = any(box_contains(p.bbox, roi) for p in high_conf_persons)
                overlaps_with_low = any(box_overlaps(roi, p.bbox) for p in low_conf_persons)

                if (not contained_in_high) or overlaps_with_low:
                    should_trigger = True
                    break

        if not should_trigger:
            # No suppressed person to recover; return high-confidence detections directly
            return [
                Detection(
                    bbox=d.bbox,
                    confidence=float(d.confidence),
                    class_id=d.class_id,
                    class_name=d.class_name,
                    delta_p_obj=0.0,
                    recovered_by_inpainting=False,
                )
                for d in pass1_all
                if d.confidence >= conf_thresh
            ]

        # Pass 2: Inpaint patch regions and redetect
        inpainted_frame = self.neutralize_patch_regions(frame, all_patch_rois, dilate_ratio=0.10)
        pass2_all = self.detect(inpainted_frame, conf_threshold=low_conf_thresh)
        pass2_persons = [d for d in pass2_all if d.class_id == 0 or d.class_name == "person"]

        # Calculate Causal Objectness Recovery (\Delta P_{obj})
        all_pass1_persons = high_conf_persons + low_conf_persons
        pass2_enriched: List[Detection] = []

        for d2 in pass2_persons:
            best_iou = 0.0
            conf_pass1 = 0.0
            for d1 in all_pass1_persons:
                iou = bbox_iou(d2.bbox, d1.bbox)
                if iou > best_iou:
                    best_iou = iou
                    conf_pass1 = float(d1.confidence)

            if best_iou < 0.25:
                conf_pass1 = 0.0

            delta_p_obj = float(d2.confidence - conf_pass1)
            recovered = bool(conf_pass1 < conf_thresh and d2.confidence >= conf_thresh)

            pass2_enriched.append(
                Detection(
                    bbox=d2.bbox,
                    confidence=float(d2.confidence),
                    class_id=d2.class_id,
                    class_name=d2.class_name,
                    delta_p_obj=round(delta_p_obj, 4),
                    recovered_by_inpainting=recovered,
                )
            )

        # Merge Pass 1 and Pass 2 person detections using Non-Maximum Suppression (NMS)
        pass1_candidates = [
            Detection(
                bbox=d.bbox,
                confidence=float(d.confidence),
                class_id=d.class_id,
                class_name=d.class_name,
                delta_p_obj=0.0,
                recovered_by_inpainting=False,
            )
            for d in high_conf_persons
            if d.confidence >= conf_thresh
        ]
        pass2_candidates = [d for d in pass2_enriched if d.confidence >= conf_thresh]

        all_candidates = pass1_candidates + pass2_candidates
        all_candidates.sort(key=lambda d: d.confidence, reverse=True)

        merged_persons: List[Detection] = []
        while all_candidates:
            best = all_candidates.pop(0)
            merged_persons.append(best)
            all_candidates = [
                c for c in all_candidates
                if bbox_iou(best.bbox, c.bbox) < self.iou_threshold
            ]

        # Preserve any non-person high-confidence detections from Pass 1
        other_detections = [
            Detection(
                bbox=d.bbox,
                confidence=float(d.confidence),
                class_id=d.class_id,
                class_name=d.class_name,
                delta_p_obj=0.0,
                recovered_by_inpainting=False,
            )
            for d in pass1_all
            if d.class_id != 0 and d.class_name != "person" and d.confidence >= conf_thresh
        ]

        return merged_persons + other_detections


class UltralyticsPyTorchWrapper:
    """Wrapper for PyTorch (.pt) weights using Ultralytics with Active Two-Pass Recovery."""

    neutralize_patch_regions = staticmethod(neutralize_patch_regions)

    def __init__(
        self,
        model_path: Union[str, Path, Any],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        clipper: Optional[SpatialFeatureNormClipper] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required to run .pt models. Please install it."
            ) from exc

        if hasattr(model_path, "predict"):
            self.model = model_path
            self.model_path = Path("yolov8n.pt")
        else:
            self.model_path = Path(model_path)
            self.model = YOLO(str(self.model_path))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Register SpatialFeatureNormClipper hook
        self.clipper = clipper or SpatialFeatureNormClipper()
        try:
            self.clipper.attach(self.model)
            if (
                hasattr(self.model, "predictor")
                and self.model.predictor is not None
                and hasattr(self.model.predictor, "model")
            ):
                self.clipper.attach(self.model.predictor.model)
        except Exception as exc:
            logger.warning("Could not attach clipper hook: %s", exc)

    def detect(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        if self.clipper is not None:
            self.clipper.set_image(image)

        results = self.model.predict(
            source=image,
            conf=thresh,
            iou=self.iou_threshold,
            classes=[0],  # Person class only
            device="cpu",
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for box, conf, cid in zip(xyxy, confs, cls_ids):
            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cid),
                    class_name="person",
                )
            )
        return detections

    detect_with_recovery = YOLOv8ONNXDetector.detect_with_recovery


YOLOv8PyTorchDetector = UltralyticsPyTorchWrapper
