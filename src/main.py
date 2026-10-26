"""ProYOLO Unified Pipeline Orchestrator and Surveillance Runner.

Coordinates:
1. Video Ingestion Thread: Ingests from RTSP, local webcam, or MP4 file into Redis queue.
2. Inference Worker: Dequeues frames, executes CPU-optimized YOLOv8n person detection,
   maintains ByteTrack track histories (N=10), extracts fused dual features, and performs
   sub-2ms attack classification.
3. Alert Dispatcher: Pushes attack events and live telemetry to the FastAPI backend.
4. Optional GUI Console: Displays live bounding boxes and threat warnings with OpenCV.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import httpx
import numpy as np

# Ensure project root is in sys.path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.classifier.attack_classifier import AttackClassifier, AttackPrediction
from src.detection.detector import (
    Detection,
    UltralyticsPyTorchWrapper,
    YOLOv8ONNXDetector,
)
from src.detection.feature_extractor import DualFeatureExtractor
from src.detection.tracker import ByteTracker, Track
from src.pipeline.redis_manager import RedisManager
from src.pipeline.stream_ingestion import StreamIngestor

logger = logging.getLogger("proyolo.main")


def create_detector(
    model_path: Union[str, Path],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Any:
    """Create and return detector instance based on file extension (.onnx or .pt)."""
    p = Path(model_path)
    if not p.exists():
        # Fallback check under data/ directory
        data_p = Path(WORKSPACE_ROOT) / "data" / p.name
        if data_p.exists():
            p = data_p

    if p.suffix.lower() == ".pt":
        logger.info("Initializing Ultralytics PyTorch detector from: %s", p)
        return UltralyticsPyTorchWrapper(p, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    else:
        logger.info("Initializing ONNX Runtime CPU detector from: %s", p)
        return YOLOv8ONNXDetector(p, input_size=(384, 384), conf_threshold=conf_threshold, iou_threshold=iou_threshold)


class AlertDispatcher:
    """Dispatches detected physical adversarial alerts and telemetry to the backend."""

    def __init__(self, server_url: Optional[str] = "http://localhost:8000") -> None:
        self.server_url = server_url.rstrip("/") if server_url else None
        self.client = httpx.Client(timeout=2.0) if self.server_url else None

    def dispatch_alert(
        self,
        camera_id: str,
        track_id: int,
        confidence: float,
        attack_probability: float,
        bbox: List[float],
        timestamp: float,
        thumbnail_b64: Optional[str] = None,
        top_features: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Send attack alert to FastAPI REST endpoint."""
        if not self.client or not self.server_url:
            return False

        payload = {
            "camera_id": camera_id,
            "track_id": int(track_id),
            "confidence": round(float(confidence), 4),
            "attack_probability": round(float(attack_probability), 4),
            "bbox": [round(float(v), 2) for v in bbox],
            "timestamp": float(timestamp),
            "thumbnail_b64": thumbnail_b64,
            "top_features": top_features or {},
        }
        try:
            res = self.client.post(f"{self.server_url}/api/v1/alerts", json=payload)
            return res.status_code == 201
        except Exception as exc:
            logger.debug("Telemetry dispatch connection error: %s", exc)
            return False

    def dispatch_telemetry(
        self,
        camera_id: str,
        timestamp: float,
        fps: float,
        latency_ms: float,
        detections: List[Dict[str, Any]],
    ) -> bool:
        """Broadcast live frame telemetry to WebSocket dashboard."""
        if not self.client or not self.server_url:
            return False

        payload = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1),
            "detections": detections,
        }
        try:
            res = self.client.post(f"{self.server_url}/api/v1/frame_telemetry", json=payload)
            return res.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        if self.client:
            self.client.close()


class ProYOLOPipeline:
    """Orchestrates end-to-end detection, tracking, feature extraction, and classification."""

    def __init__(
        self,
        source: Union[str, int],
        model_path: Union[str, Path] = "yolov8n.onnx",
        classifier_path: Union[str, Path] = "models/attack_classifier.joblib",
        camera_id: str = "cam-01",
        target_fps: float = 15.0,
        threshold: float = 0.65,
        server_url: Optional[str] = "http://localhost:8000",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        headless: bool = False,
    ) -> None:
        self.source = source
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.threshold = threshold
        self.headless = headless

        # 1. Initialize Pipeline Ingestion & Queue
        self.redis_manager = RedisManager(
            host=redis_host,
            port=redis_port,
            queue_name=f"stream_{camera_id}",
            allow_mock_fallback=True,
        )
        self.ingestor = StreamIngestor(
            source=source,
            redis_manager=self.redis_manager,
            camera_id=camera_id,
            target_fps=target_fps,
            loop_file=True,
        )

        # 2. Initialize Models
        self.detector = create_detector(model_path)
        self.tracker = ByteTracker(track_thresh=0.40, low_thresh=0.12, match_thresh=0.7)
        self.feature_extractor = DualFeatureExtractor()
        self.classifier = AttackClassifier(model_path=classifier_path, threshold=threshold)

        # 3. Alert Dispatcher
        self.dispatcher = AlertDispatcher(server_url=server_url)

        # 4. State & Diagnostics
        self.is_running = False
        self.total_frames_processed = 0
        self.total_attacks_flagged = 0
        self.recent_latencies: List[float] = []

    def start(self) -> None:
        """Start the ingestion thread and begin processing."""
        logger.info("Starting ProYOLO pipeline for camera: %s (source: %s)", self.camera_id, self.source)
        self.ingestor.start()
        self.is_running = True

    def stop(self) -> None:
        """Gracefully stop ingestion, dispatcher, and display windows."""
        logger.info("Stopping ProYOLO pipeline...")
        self.is_running = False
        self.ingestor.stop(timeout=2.0)
        self.dispatcher.close()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Execute full inference cycle on a single video frame.

        Returns:
            Tuple of annotated display frame and detection telemetry dictionaries.
        """
        t0 = time.perf_counter()

        # Step 1: Detect Persons with Active Two-Pass Inpainting Recovery
        if hasattr(self.detector, "detect_with_recovery"):
            detections = self.detector.detect_with_recovery(frame, conf_thresh=0.40, low_conf_thresh=0.12)
        else:
            detections = self.detector.detect(frame)
        person_detections = [d for d in detections if d.class_id == 0]

        # Step 2: Multi-Object Tracking (ByteTrack with dual-threshold association)
        active_tracks = self.tracker.update(person_detections)

        # Step 3: Dual Feature Extraction & Attack Classification
        telemetry_items: List[Dict[str, Any]] = []
        display_frame = frame.copy() if not self.headless else frame

        for track in active_tracks:
            feat_packet = self.feature_extractor.extract_single(frame, track)
            prediction: AttackPrediction = self.classifier.predict(feat_packet)

            is_threat = prediction.is_attack
            confirmed_alert = track.record_attack_prediction(is_threat)
            prob = float(prediction.attack_probability)
            conf = float(track.latest_confidence)
            bbox = [float(v) for v in track.latest_bbox]
            recovered_by_inpainting = bool(getattr(track, "recovered_by_inpainting", False))
            delta_p_obj = float(getattr(track, "delta_p_obj", 0.0))

            item = {
                "track_id": track.track_id,
                "bbox": bbox,
                "confidence": round(conf, 3),
                "is_attack": is_threat,
                "confirmed_attack_alert": confirmed_alert,
                "attack_probability": round(prob, 3),
                "recovered_by_inpainting": recovered_by_inpainting,
                "delta_p_obj": round(delta_p_obj, 3),
            }
            telemetry_items.append(item)

            # Enforce temporal persistence: K=3 out of N=5 consecutive frames
            if confirmed_alert:
                self.total_attacks_flagged += 1
                logger.warning(
                    "🚨 CONFIRMED PHYSICAL ADVERSARIAL ATTACK on Track #%d (P=%.2f, Conf=%.2f, K=3/5 persistence)",
                    track.track_id,
                    prob,
                    conf,
                )

                # Generate thumbnail snapshot
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h_img, w_img = frame.shape[:2]
                crop = frame[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                thumb_b64 = None
                if crop.size > 0:
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

                # Dispatch Alert to Redis Manager and Backend
                top_features = dict(prediction.top_contributing_features)
                alert_payload = {
                    "camera_id": self.camera_id,
                    "track_id": track.track_id,
                    "confidence": conf,
                    "attack_probability": prob,
                    "bbox": bbox,
                    "timestamp": timestamp,
                    "thumbnail_b64": thumb_b64,
                    "top_features": top_features,
                    "confirmed_attack_alert": True,
                    "recovered_by_inpainting": recovered_by_inpainting,
                    "delta_p_obj": round(delta_p_obj, 4),
                }

                if hasattr(self, "redis_manager") and self.redis_manager is not None:
                    try:
                        self.redis_manager.publish_alert(alert_payload)
                    except Exception as exc:
                        logger.debug("Redis alert publish error: %s", exc)

                self.dispatcher.dispatch_alert(
                    camera_id=self.camera_id,
                    track_id=track.track_id,
                    confidence=conf,
                    attack_probability=prob,
                    bbox=bbox,
                    timestamp=timestamp,
                    thumbnail_b64=thumb_b64,
                    top_features=top_features,
                )

            # Draw annotations if GUI is active
            if not self.headless:
                self._draw_box(
                    display_frame,
                    track.track_id,
                    bbox,
                    conf,
                    confirmed_alert,
                    prob,
                    recovered_by_inpainting=recovered_by_inpainting,
                    delta_p_obj=delta_p_obj,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.recent_latencies.append(elapsed_ms)
        if len(self.recent_latencies) > 30:
            self.recent_latencies.pop(0)

        # Dispatch frame telemetry
        avg_latency = float(np.mean(self.recent_latencies)) if self.recent_latencies else elapsed_ms
        current_fps = 1000.0 / max(1.0, avg_latency)
        self.dispatcher.dispatch_telemetry(
            camera_id=self.camera_id,
            timestamp=timestamp,
            fps=current_fps,
            latency_ms=avg_latency,
            detections=telemetry_items,
        )

        self.total_frames_processed += 1
        return display_frame, telemetry_items

    @staticmethod
    def _draw_box(
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
        elif recovered_by_inpainting:
            color = (212, 182, 6)  # Cyan/Teal in BGR
            thick = 2
            label = f"Recovered #{track_id} ({conf * 100:.0f}% ΔP={delta_p_obj:+.2f})"
        else:
            color = (129, 185, 16)  # Emerald Green in BGR
            thick = 2
            label = f"Person #{track_id} ({conf * 100:.0f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Draw header banner
        (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + w_txt + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def run(self, max_frames: Optional[int] = None) -> None:
        """Main inference processing loop."""
        self.start()
        win_name = f"ProYOLO Surveillance Defense Console - {self.camera_id}"
        if not self.headless:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        try:
            while self.is_running:
                packet = self.redis_manager.pop_frame()
                if packet is None:
                    time.sleep(0.005)
                    continue

                frame = packet.frame
                ts = packet.timestamp

                display_frame, _ = self.process_frame(frame, ts)

                if not self.headless:
                    # Draw OSD Banner
                    avg_lat = float(np.mean(self.recent_latencies)) if self.recent_latencies else 0.0
                    fps_val = 1000.0 / max(1.0, avg_lat)
                    osd_text = f"FPS: {fps_val:.1f} | Latency: {avg_lat:.1f}ms | Flagged: {self.total_attacks_flagged}"
                    cv2.putText(
                        display_frame,
                        osd_text,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 242, 254),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(win_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        logger.info("Exit requested by user via GUI.")
                        break

                if max_frames and self.total_frames_processed >= max_frames:
                    logger.info("Processed target maximum of %d frames.", max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by KeyboardInterrupt.")
        finally:
            self.stop()
            logger.info(
                "ProYOLO finished: Processed %d frames, Flagged %d attacks.",
                self.total_frames_processed,
                self.total_attacks_flagged,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        description="ProYOLO Real-Time Physical Adversarial Patch Attack Detection Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: webcam index (e.g. '0'), local MP4/AVI file path, or RTSP URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.onnx",
        help="Path to YOLOv8 weights (.onnx or .pt format).",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="models/attack_classifier.joblib",
        help="Path to attack classifier model (.joblib or .onnx).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without GUI window; streams telemetry and alerts to web server.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam-01",
        help="Camera identifier string.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="ProYOLO telemetry backend URL.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Physical patch attack probability threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Video capture target frame rate.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for testing).",
    )
    return parser


def main() -> None:
    """Main execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parse source as int if purely numeric
    source: Union[str, int] = int(args.source) if args.source.isdigit() else args.source

    pipeline = ProYOLOPipeline(
        source=source,
        model_path=args.model,
        classifier_path=args.classifier,
        camera_id=args.camera_id,
        target_fps=args.target_fps,
        threshold=args.threshold,
        server_url=args.server_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        headless=args.headless,
    )

    # Register OS signal handlers for graceful shutdown
    def _signal_handler(signum, frame):
        logger.info("Received termination signal (%d). Shutting down...", signum)
        pipeline.is_running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
