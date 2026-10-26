"""Real-Time OpenCV Video Stream Processing Engine for ProYOLO DefOps.

Executes live webcam / RTSP / uploaded video processing:
1. Video Capture: Reads from webcam (cv2.VideoCapture(0)), RTSP streams, or uploaded video files.
2. Real-Time Detection & Defense:
   - YOLOv8n object detection (person class).
   - ByteTrack multi-object tracking with 10-frame sliding window history.
   - Dual feature extraction (temporal anomaly + HSV/Laplacian spatial entropy).
   - Attack classification (<2ms on CPU) flagging physical adversarial optical patches.
3. Live Visual Annotations:
   - Green bounding boxes for verified clean pedestrians ([CLEAN] conf: 0.xx).
   - Glowing Red bounding boxes with warning badges for detected attacks ([ATTACK] prob: 0.xx).
   - Real-time OSD telemetry HUD (FPS, latency, camera mode).
4. Multipart Motion-JPEG Streaming:
   - Yields multipart/x-mixed-replace JPEG frames for GET /api/v1/video-feed.
   - Automatically dispatches alert events over WebSocket with base64 patch crops.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.classifier.attack_classifier import AttackClassifier, AttackPrediction
from src.detection.detector import Detection, YOLOv8ONNXDetector
from src.detection.feature_extractor import DualFeatureExtractor
from src.detection.tracker import ByteTracker, Track

logger = logging.getLogger("proyolo.stream_engine")


class StreamEngine:
    """Active OpenCV video processing and Motion-JPEG streaming engine."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        model_path: Union[str, Path] = "yolov8n.onnx",
        classifier_path: Union[str, Path] = "models/attack_classifier.joblib",
        camera_id: str = "CAM-01",
        target_fps: float = 15.0,
        threshold: float = 0.65,
        alert_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        redis_manager: Optional[Any] = None,
        detector: Optional[Any] = None,
        tracker: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        classifier: Optional[Any] = None,
    ) -> None:
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.threshold = threshold
        self.alert_callback = alert_callback
        self.telemetry_callback = telemetry_callback
        self.redis_manager = redis_manager

        # Internal state
        self.current_source: Union[int, str] = source
        self.source_label: str = "Webcam (Live)" if source == 0 else str(source)
        self.is_paused: bool = False
        self.is_looping: bool = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames_processed: int = 0
        self.total_attacks_flagged: int = 0
        self.recent_latencies: List[float] = []
        self._last_frame_bytes: Optional[bytes] = None
        self._fallback_frame_idx: int = 0

        # Initialize detector with CPU-optimized 384x384 resolution
        if detector is not None:
            self.detector = detector
        else:
            model_p = Path(model_path)
            if not model_p.exists():
                data_model_p = Path(WORKSPACE_ROOT) / "data" / model_p.name
                if data_model_p.exists():
                    model_p = data_model_p
            self.detector = YOLOv8ONNXDetector(model_p, input_size=(384, 384), conf_threshold=0.35)

        # Initialize tracker & defense models
        self.tracker = tracker if tracker is not None else ByteTracker(track_thresh=0.40, match_thresh=0.7)
        self.feature_extractor = feature_extractor if feature_extractor is not None else DualFeatureExtractor()
        self.classifier = classifier if classifier is not None else AttackClassifier(model_path=classifier_path, threshold=threshold)

        # Open initial capture source
        self._open_capture(source)

    def _open_capture(self, source: Union[int, str]) -> None:
        """Open cv2.VideoCapture with appropriate backend settings."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.current_source = source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            self.source_label = f"Webcam (Device {source})"
            if os.environ.get("PROYOLO_TEST_MODE") == "1":
                self.cap = None
            else:
                self.cap = cv2.VideoCapture(source)
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.source_label = Path(str(source)).name if os.path.exists(str(source)) else str(source)
            self.cap = cv2.VideoCapture(str(source))

        # Check if source opened successfully
        if self.cap is None or not self.cap.isOpened():
            logger.warning(
                "Could not open video source '%s'. Will stream with graceful synthetic/dataset loop.",
                source,
            )
            self.cap = None

    def set_source(self, source: Union[int, str], loop: bool = True) -> None:
        """Switch video capture source dynamically (webcam, uploaded video, or RTSP)."""
        logger.info("Switching stream engine source to: %s (loop=%s)", source, loop)
        self.is_looping = loop
        self.is_paused = False
        self.tracker = ByteTracker(track_thresh=0.40, match_thresh=0.7)
        self._open_capture(source)

    def set_paused(self, paused: bool) -> None:
        """Pause or unpause video processing."""
        self.is_paused = paused

    def toggle_pause(self) -> bool:
        """Toggle video pause state and return new state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def _get_fallback_frame(self) -> np.ndarray:
        """Generate a simulated CCTV frame when webcam or file is not accessible."""
        self._fallback_frame_idx += 1
        h, w = 540, 960
        frame = np.full((h, w, 3), 20, dtype=np.uint8)

        # Perspective ground lines
        cv2.line(frame, (0, int(h * 0.7)), (w, int(h * 0.7)), (45, 55, 75), 2)
        for x_step in range(0, w, 120):
            cv2.line(frame, (w // 2, int(h * 0.3)), (x_step, h), (30, 40, 55), 1)

        # Walking pedestrian silhouette
        pos_x = int(150 + ((self._fallback_frame_idx * 3) % (w - 300)))
        pos_y = int(180 + np.sin(self._fallback_frame_idx * 0.1) * 8)
        pw, ph = 100, 240

        # Draw body & head
        cv2.rectangle(frame, (pos_x, pos_y + 40), (pos_x + pw, pos_y + ph), (100, 110, 125), -1)
        cv2.circle(frame, (pos_x + pw // 2, pos_y + 25), 20, (120, 130, 145), -1)

        # Intermittently display an adversarial patch
        is_attack_frame = (self._fallback_frame_idx % 120) > 50
        if is_attack_frame:
            patch_w, patch_h = 50, 45
            ptx = pos_x + (pw - patch_w) // 2
            pty = pos_y + 55
            # Draw multi-color optical patch
            cv2.rectangle(frame, (ptx, pty), (ptx + patch_w, pty + patch_h), (95, 42, 255), -1)
            cv2.rectangle(frame, (ptx + 6, pty + 6), (ptx + 24, pty + 24), (212, 182, 6), -1)
            cv2.rectangle(frame, (ptx + 24, pty + 20), (ptx + 44, pty + 40), (11, 158, 245), -1)

        return frame

    def read_raw_frame(self) -> np.ndarray:
        """Read next raw frame from capture device or fallback."""
        if self.cap is not None and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    if frame.shape[1] > 800:
                        scale = 800.0 / frame.shape[1]
                        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    return frame
                elif self.is_looping and isinstance(self.current_source, (str, Path)):
                    # Rewind video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret2, frame2 = self.cap.read()
                    if ret2 and frame2 is not None and frame2.size > 0:
                        if frame2.shape[1] > 800:
                            scale = 800.0 / frame2.shape[1]
                            frame2 = cv2.resize(frame2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        return frame2
            except Exception as exc:
                logger.warning("Error reading frame from capture device: %s", exc)

        # Return simulated CCTV frame if live capture unavailable
        return self._get_fallback_frame()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Execute full YOLOv8 Active Two-Pass + ByteTrack + Feature Extraction + Classifier pipeline."""
        t0 = time.perf_counter()
        display_frame = frame.copy()
        h_img, w_img = frame.shape[:2]

        # 1. Object Detection with Active Two-Pass Inpainting Recovery
        if hasattr(self.detector, "detect_with_recovery"):
            detections = self.detector.detect_with_recovery(frame, conf_thresh=0.40, low_conf_thresh=0.12)
        else:
            detections = self.detector.detect(frame)
        person_detections = [d for d in detections if d.class_id == 0]

        # 2. Multi-Object Tracking (ByteTrack with dual-threshold association)
        active_tracks = self.tracker.update(person_detections)

        telemetry_items: List[Dict[str, Any]] = []

        for track in active_tracks:
            feat_packet = self.feature_extractor.extract_single(frame, track)
            prediction: AttackPrediction = self.classifier.predict(feat_packet)

            is_threat = prediction.is_attack
            # Enforce temporal persistence: K=3 out of N=5 consecutive frames
            confirmed_alert = track.record_attack_prediction(is_threat)
            prob = float(prediction.attack_probability)
            conf = float(track.latest_confidence)
            bbox = [float(v) for v in track.latest_bbox]
            recovered_by_inpainting = bool(getattr(track, "recovered_by_inpainting", False))
            delta_p_obj = float(getattr(track, "delta_p_obj", 0.0))

            item = {
                "track_id": track.track_id,
                "label": "Attacker" if confirmed_alert else ("Suspect" if is_threat else "Person"),
                "bbox": bbox,
                "confidence": round(conf, 3),
                "is_attack": is_threat,
                "confirmed_attack_alert": confirmed_alert,
                "attack_probability": round(prob, 3),
                "recovered_by_inpainting": recovered_by_inpainting,
                "delta_p_obj": round(delta_p_obj, 3),
            }
            telemetry_items.append(item)

            # Draw Real Annotations (Person / Attacker / Inpainting Recovered)
            self._draw_detection_box(
                display_frame,
                track.track_id,
                bbox,
                conf,
                confirmed_alert,
                prob,
                recovered_by_inpainting=recovered_by_inpainting,
                delta_p_obj=delta_p_obj,
            )

            # Only fire alerts when confirmed by temporal persistence (K=3 / N=5)
            if confirmed_alert:
                self.total_attacks_flagged += 1

                # Generate base64 thumbnail crop
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = frame[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                thumb_b64 = None
                if crop.size > 0:
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

                alert_data = {
                    "camera_id": self.camera_id,
                    "track_id": track.track_id,
                    "classification": "Attacker",
                    "confidence": round(conf, 4),
                    "attack_probability": round(prob, 4),
                    "bbox": [round(float(v), 2) for v in bbox],
                    "timestamp": timestamp,
                    "thumbnail_b64": thumb_b64,
                    "top_features": dict(prediction.top_contributing_features),
                    "confirmed_attack_alert": True,
                    "recovered_by_inpainting": recovered_by_inpainting,
                    "delta_p_obj": round(delta_p_obj, 4),
                }

                # Publish to Redis Manager if configured
                if hasattr(self, "redis_manager") and self.redis_manager is not None:
                    try:
                        self.redis_manager.publish_alert(alert_data)
                    except Exception as exc:
                        logger.debug("Redis alert publish error: %s", exc)

                if self.alert_callback:
                    try:
                        res = self.alert_callback(alert_data)
                        if asyncio.iscoroutine(res):
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(res)
                            except RuntimeError:
                                res.close()
                    except Exception as exc:
                        logger.debug("Alert callback error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.recent_latencies.append(elapsed_ms)
        if len(self.recent_latencies) > 30:
            self.recent_latencies.pop(0)

        avg_lat = float(np.mean(self.recent_latencies)) if self.recent_latencies else elapsed_ms
        current_fps = 1000.0 / max(1.0, avg_lat)

        # Draw OSD Telemetry HUD
        self._draw_osd_hud(display_frame, current_fps, avg_lat, len(active_tracks))

        self.total_frames_processed += 1

        if self.telemetry_callback:
            try:
                res = self.telemetry_callback({
                    "fps": round(current_fps, 1),
                    "latency_ms": round(avg_lat, 1),
                    "clean_detections": sum(1 for d in telemetry_items if not d["is_attack"]),
                    "flagged_attacks": self.total_attacks_flagged,
                    "persons_detected": sum(1 for d in telemetry_items if not d["is_attack"]),
                    "attackers_flagged": self.total_attacks_flagged,
                    "processed_frames": self.total_frames_processed,
                })
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        res.close()
            except Exception:
                pass

        return display_frame, telemetry_items

    @staticmethod
    def _draw_detection_box(
        frame: np.ndarray,
        track_id: int,
        bbox: List[float],
        conf: float,
        is_attack: bool,
        prob: float,
        recovered_by_inpainting: bool = False,
        delta_p_obj: float = 0.0,
    ) -> None:
        """Render annotated bounding boxes and threat labels on display frame."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if is_attack:
            color = (95, 42, 255)  # Crimson Red in BGR
            thick = 3
            label = f"Attacker #{track_id} ({prob * 100:.0f}%)"
            badge_color = (95, 42, 255)
        elif recovered_by_inpainting:
            color = (212, 182, 6)  # Cyan/Teal in BGR for Inpainting Recovered
            thick = 2
            label = f"Recovered #{track_id} ({conf * 100:.0f}% ΔP={delta_p_obj:+.2f})"
            badge_color = (212, 182, 6)
        else:
            color = (129, 185, 16)  # Emerald Green in BGR
            thick = 2
            label = f"Person #{track_id} ({conf * 100:.0f}%)"
            badge_color = (129, 185, 16)

        # Draw Main Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Corner Brackets
        c_len = 12
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, thick + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, thick + 1)
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, thick + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, thick + 1)
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, thick + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), color, thick + 1)
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, thick + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, thick + 1)

        # Label Banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), badge_color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(15, y1 - 6)),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_osd_hud(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        num_tracks: int,
    ) -> None:
        """Render surveillance OSD HUD across the top of the video frame."""
        h, w = frame.shape[:2]
        # Semi-transparent HUD header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # REC indicator
        cv2.circle(frame, (18, 18), 5, (95, 42, 255), -1)
        cv2.putText(frame, "LIVE REC", (28, 22), font, 0.45, (95, 42, 255), 1, cv2.LINE_AA)

        # Source & Camera OSD
        src_info = f"{self.camera_id} • {self.source_label}"
        if self.is_paused:
            src_info += " [PAUSED]"
        cv2.putText(frame, src_info, (115, 22), font, 0.45, (220, 230, 245), 1, cv2.LINE_AA)

        # Telemetry metrics on right side
        hud_right = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Attackers: {self.total_attacks_flagged}"
        (rw, _), _ = cv2.getTextSize(hud_right, font, 0.45, 1)
        cv2.putText(frame, hud_right, (w - rw - 15, 22), font, 0.45, (6, 182, 212), 1, cv2.LINE_AA)

    async def stream_generator(self) -> AsyncGenerator[bytes, None]:
        """Continuous Motion-JPEG generator yielding multipart frame chunks."""
        frame_interval = 1.0 / max(1.0, self.target_fps)

        try:
            while True:
                t_start = time.perf_counter()

                if not self.is_paused or self._last_frame_bytes is None:
                    raw_frame = self.read_raw_frame()
                    ts = time.time()
                    annotated_frame, _ = self.process_frame(raw_frame, ts)

                    # JPEG encode
                    ok, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self._last_frame_bytes = jpeg_buf.tobytes()

                if self._last_frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + self._last_frame_bytes + b"\r\n"
                    )

                # Maintain smooth frame rate
                elapsed = time.perf_counter() - t_start
                delay = max(0.005, frame_interval - elapsed)
                await asyncio.sleep(delay)
        except (asyncio.CancelledError, GeneratorExit):
            return
