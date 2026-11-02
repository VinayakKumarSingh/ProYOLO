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
    is_attack: bool = False
    patch_roi: Optional[Sequence[float]] = None

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

    def to_dict(self) -> dict:
        """Return dictionary representation of detection."""
        return {
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else list(self.bbox),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "delta_p_obj": float(self.delta_p_obj),
            "recovered_by_inpainting": bool(self.recovered_by_inpainting),
            "is_attack": bool(self.is_attack),
            "patch_roi": list(self.patch_roi) if self.patch_roi is not None else None,
        }

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if hasattr(self, key):
                return getattr(self, key)
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


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


def is_near_frame_edge(
    box: Sequence[float],
    frame_shape: Sequence[int],
    margin: float = 0.05,
) -> bool:
    """Check if bounding box touches or is near the outer frame boundary.

    Returns True if x1 <= margin*W, y1 <= margin*H, x2 >= (1-margin)*W, or y2 >= (1-margin)*H.
    """
    h, w = float(frame_shape[0]), float(frame_shape[1])
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return bool(
        x1 <= margin * w
        or y1 <= margin * h
        or x2 >= (1.0 - margin) * w
        or y2 >= (1.0 - margin) * h
    )


def is_inside_torso(
    patch_box: Sequence[float],
    person_box: Sequence[float],
) -> bool:
    """Check if candidate patch ROI lies strictly inside the person's torso region.

    Returns True ONLY if the center of patch_box lies horizontally within
    [px1 + 0.15*pw, px2 - 0.15*pw] and vertically within [py1 + 0.15*ph, py2 - 0.20*ph],
    and the patch area is between 4% and 45% of the person box area.
    """
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
    is_near_frame_edge = staticmethod(is_near_frame_edge)
    is_inside_torso = staticmethod(is_inside_torso)

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.onnx",
        input_size: Tuple[int, int] = (480, 480),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[Sequence[int]] = None,
        num_threads: Optional[int] = None,
        provider: str = "CPUExecutionProvider",
        session: Optional[Any] = None,
        model: Optional[Any] = None,
        clipper: Optional[SpatialFeatureNormClipper] = None,
        patch_model: Optional[Any] = None,
        patch_model_path: Optional[Union[str, Path]] = "models/ad_yolo_best.pt",
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

        # Custom secondary patch model
        self.patch_model_path = Path(patch_model_path) if patch_model_path is not None else Path("models/ad_yolo_best.pt")
        self.patch_model = patch_model
        if self.patch_model is None and self.patch_model_path.exists():
            try:
                from ultralytics import YOLO

                self.patch_model = YOLO(str(self.patch_model_path))
                logger.info("Loaded custom patch detector from %s", self.patch_model_path)
            except Exception as exc:
                logger.warning("Could not load patch model from %s: %s", self.patch_model_path, exc)
                self.patch_model = None

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

    def predict_patch_rois(
        self, frame: np.ndarray, conf_threshold: float = 0.25
    ) -> List[List[float]]:
        """Predict tight adversarial_patch bounding boxes using secondary patch model if available."""
        if self.patch_model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self.patch_model.predict(
                source=frame,
                conf=conf_threshold,
                device="cpu",
                verbose=False,
            )
            rois: List[List[float]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                for b in xyxy:
                    rois.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            return rois
        except Exception as exc:
            logger.warning("Error running patch_model inference: %s", exc)
            return []

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
        names_dict = getattr(self.model, "names", None)

        for box, conf, cid in zip(xyxy, confs, cls_ids):
            c_int = int(cid)
            if names_dict and c_int in names_dict:
                name = str(names_dict[c_int])
            elif c_int < len(COCO_CLASSES):
                name = COCO_CLASSES[c_int]
            else:
                name = f"class_{c_int}"
            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=c_int,
                    class_name=name,
                    delta_p_obj=0.0,
                    recovered_by_inpainting=False,
                    is_attack=False,
                    patch_roi=None,
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
        names_dict = getattr(self.model, "names", None)
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                cls_id = classes_for_nms[i]
                if names_dict and cls_id in names_dict:
                    cls_name = str(names_dict[cls_id])
                elif cls_id < len(COCO_CLASSES):
                    cls_name = COCO_CLASSES[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                detections.append(
                    Detection(
                        bbox=coords_for_nms[i],
                        confidence=scores_for_nms[i],
                        class_id=cls_id,
                        class_name=cls_name,
                        delta_p_obj=0.0,
                        recovered_by_inpainting=False,
                        is_attack=False,
                        patch_roi=None,
                    )
                )

        return detections

    def detect_with_recovery(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.40,
        low_conf_thresh: float = 0.15,
        patch_rois: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Detection]:
        """Active Two-Pass 'Detect-Neutralize-Redetect' pipeline with strict edge/torso guards.

        Recovers persons missed due to Thys et al. adversarial patch objectness suppression:
        1. Pass 1: Run YOLO inference on raw frame using low_conf_thresh.
           Keep all non-person COCO detections untouched with is_attack=False.
           Separate high-confidence persons and low-confidence suppressed candidates.
        2. Collect candidate patch_rois (from self.patch_model if loaded, clipper from
           feature_extractor.py, or caller), filtering out any ROI near frame edge.
        3. Pass 2 Trigger: Only run Telea Inpainting if a valid interior patch_roi overlaps
           a person box (not near frame edge) via is_inside_torso, OR overlaps a low-confidence
           suppressed interior person candidate.
        4. Pass 2: Inpaint patch regions and re-run YOLO inference on neutralized frame.
        5. Calculate Causal Objectness Recovery (\\Delta P_{obj}): delta_p_obj = conf_pass2 - conf_pass1.
           Merge recovered person boxes via NMS and attach delta_p_obj, patch_roi, and
           recovered_by_inpainting to the detection metadata.
        """
        if frame is None or frame.size == 0:
            return []

        frame_shape = frame.shape[:2]

        # Pass 1: Proposal generation with low confidence threshold
        pass1_all = self.detect(frame, conf_threshold=low_conf_thresh)

        # Preserve all non-person COCO detections untouched with is_attack=False
        non_person_detections = [
            Detection(
                bbox=d.bbox,
                confidence=float(d.confidence),
                class_id=d.class_id,
                class_name=d.class_name,
                delta_p_obj=0.0,
                recovered_by_inpainting=False,
                is_attack=False,
                patch_roi=None,
            )
            for d in pass1_all
            if d.class_id != 0 and d.class_name != "person" and d.confidence >= conf_thresh
        ]

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
        raw_patch_rois: List[List[float]] = []
        if patch_rois is not None:
            raw_patch_rois.extend([list(r) for r in patch_rois])

        if hasattr(self, "patch_model") and self.patch_model is not None:
            raw_patch_rois.extend(self.predict_patch_rois(frame))

        if hasattr(self, "clipper") and self.clipper is not None and self.clipper.patch_rois:
            raw_patch_rois.extend([list(r) for r in self.clipper.patch_rois])

        for det in pass1_all:
            if det.class_name.lower() in ("patch", "adversarial_patch") or det.class_id == 80:
                raw_patch_rois.append(det.bbox.tolist() if isinstance(det.bbox, np.ndarray) else list(det.bbox))

        # Filter out candidate patch ROIs near the frame edge (margin=0.05)
        valid_patch_rois: List[List[float]] = [
            roi for roi in raw_patch_rois
            if len(roi) >= 4 and not is_near_frame_edge(roi, frame_shape, margin=0.05)
        ]

        # Pass 2 Trigger Condition:
        # Only run Pass 2 Telea Inpainting if a valid interior patch_roi overlaps a person box
        # (that is NOT near the frame edge) via is_inside_torso, OR overlaps a low-confidence
        # suppressed interior person candidate.
        should_trigger = False
        triggering_rois: List[List[float]] = []

        for roi in valid_patch_rois:
            matched = False

            # Check high-confidence persons (not near frame edge) via is_inside_torso
            for p in high_conf_persons:
                if not is_near_frame_edge(p.bbox, frame_shape, margin=0.05) and is_inside_torso(roi, p.bbox):
                    matched = True
                    break

            # Check low-confidence suppressed interior person candidates
            if not matched:
                for lp in low_conf_persons:
                    if not is_near_frame_edge(lp.bbox, frame_shape, margin=0.05) and (
                        is_inside_torso(roi, lp.bbox) or box_overlaps(roi, lp.bbox)
                    ):
                        matched = True
                        break

            # Fallback for complete evasion (where person was completely missed in Pass 1 and patch ROIs / patch model provided)
            if not matched and not high_conf_persons and not low_conf_persons:
                if patch_rois is not None or (hasattr(self, "patch_model") and self.patch_model is not None):
                    matched = True

            if matched:
                should_trigger = True
                triggering_rois.append(roi)

        if not should_trigger:
            pass1_persons = [
                Detection(
                    bbox=d.bbox,
                    confidence=float(d.confidence),
                    class_id=d.class_id,
                    class_name=d.class_name,
                    delta_p_obj=0.0,
                    recovered_by_inpainting=False,
                    is_attack=False,
                    patch_roi=None,
                )
                for d in high_conf_persons
            ]
            return pass1_persons + non_person_detections

        # Pass 2: Telea Inpainting (radius=5, dilated by 10%) and redetect
        inpaint_rois = triggering_rois if triggering_rois else valid_patch_rois
        inpainted_frame = neutralize_patch_regions(frame, inpaint_rois, dilate_ratio=0.10)
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

            # Associate corresponding patch ROI
            matched_patch: Optional[List[float]] = None
            for r in inpaint_rois:
                if is_inside_torso(r, d2.bbox) or box_overlaps(r, d2.bbox):
                    matched_patch = r
                    break
            if matched_patch is None and inpaint_rois:
                matched_patch = inpaint_rois[0]

            pass2_enriched.append(
                Detection(
                    bbox=d2.bbox,
                    confidence=float(d2.confidence),
                    class_id=d2.class_id,
                    class_name=d2.class_name,
                    delta_p_obj=round(delta_p_obj, 4),
                    recovered_by_inpainting=recovered,
                    is_attack=bool(recovered or delta_p_obj >= 0.15),
                    patch_roi=matched_patch,
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
                is_attack=False,
                patch_roi=None,
            )
            for d in high_conf_persons
        ]
        pass2_candidates = [d for d in pass2_enriched if d.confidence >= conf_thresh]

        all_candidates = pass2_candidates + pass1_candidates
        all_candidates.sort(key=lambda d: (d.recovered_by_inpainting, d.confidence), reverse=True)

        merged_persons: List[Detection] = []
        while all_candidates:
            best = all_candidates.pop(0)
            merged_persons.append(best)
            all_candidates = [
                c for c in all_candidates
                if bbox_iou(best.bbox, c.bbox) < self.iou_threshold
            ]

        return merged_persons + non_person_detections


class UltralyticsPyTorchWrapper:
    """Wrapper for PyTorch (.pt) weights using Ultralytics with Active Two-Pass Recovery."""

    neutralize_patch_regions = staticmethod(neutralize_patch_regions)
    is_near_frame_edge = staticmethod(is_near_frame_edge)
    is_inside_torso = staticmethod(is_inside_torso)

    def __init__(
        self,
        model_path: Union[str, Path, Any] = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[Sequence[int]] = None,
        clipper: Optional[SpatialFeatureNormClipper] = None,
        patch_model: Optional[Any] = None,
        patch_model_path: Optional[Union[str, Path]] = "models/ad_yolo_best.pt",
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
        self.target_classes = set(target_classes) if target_classes is not None else None

        # Custom secondary patch model
        self.patch_model_path = Path(patch_model_path) if patch_model_path is not None else Path("models/ad_yolo_best.pt")
        self.patch_model = patch_model
        if self.patch_model is None and self.patch_model_path.exists():
            try:
                self.patch_model = YOLO(str(self.patch_model_path))
                logger.info("Loaded custom patch detector from %s", self.patch_model_path)
            except Exception as exc:
                logger.warning("Could not load patch model from %s: %s", self.patch_model_path, exc)
                self.patch_model = None

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

    def predict_patch_rois(
        self, frame: np.ndarray, conf_threshold: float = 0.25
    ) -> List[List[float]]:
        """Predict tight adversarial_patch bounding boxes using secondary patch model if available."""
        if self.patch_model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self.patch_model.predict(
                source=frame,
                conf=conf_threshold,
                device="cpu",
                verbose=False,
            )
            rois: List[List[float]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                for b in xyxy:
                    rois.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            return rois
        except Exception as exc:
            logger.warning("Error running patch_model inference: %s", exc)
            return []

    def detect(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        if self.clipper is not None:
            self.clipper.set_image(image)

        classes = list(self.target_classes) if self.target_classes is not None else None
        results = self.model.predict(
            source=image,
            conf=thresh,
            iou=self.iou_threshold,
            classes=classes,  # All 80 COCO classes by default
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
        names_dict = getattr(self.model, "names", None)

        for box, conf, cid in zip(xyxy, confs, cls_ids):
            c_int = int(cid)
            if names_dict and c_int in names_dict:
                name = str(names_dict[c_int])
            elif c_int < len(COCO_CLASSES):
                name = COCO_CLASSES[c_int]
            else:
                name = f"class_{c_int}"

            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=c_int,
                    class_name=name,
                    delta_p_obj=0.0,
                    recovered_by_inpainting=False,
                    is_attack=False,
                    patch_roi=None,
                )
            )
        return detections

    detect_with_recovery = YOLOv8ONNXDetector.detect_with_recovery


YOLOv8PyTorchDetector = UltralyticsPyTorchWrapper

__all__ = [
    "COCO_CLASSES",
    "Detection",
    "YOLOv8ONNXDetector",
    "YOLOv8PyTorchDetector",
    "UltralyticsPyTorchWrapper",
    "bbox_iou",
    "box_contains",
    "box_overlaps",
    "is_near_frame_edge",
    "is_inside_torso",
    "neutralize_patch_regions",
]
