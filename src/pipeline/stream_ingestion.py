"""Stream Ingestion Module for ProYOLO.

Handles asynchronous video capture from RTSP streams, local webcams, and video files
in a dedicated background worker thread, with frame rate throttling and resilient
auto-reconnect logic using exponential backoff.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from src.pipeline.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class StreamIngestor:
    """Asynchronous video stream ingestion worker with auto-reconnection and rate control.

    Attributes:
        source: Source identifier (RTSP URL string, video file path, or webcam integer index).
        redis_manager: RedisManager instance to push captured frames to.
        camera_id: Identifier tag for this video stream.
        target_fps: Desired frame ingestion rate (default 15.0 FPS).
        max_reconnect_attempts: Max consecutive reconnection attempts before stopping (None for infinite).
        initial_backoff: Initial backoff delay in seconds on disconnection.
        max_backoff: Upper bound for backoff delay in seconds.
        backoff_factor: Multiplier for exponential backoff delay.
        loop_file: If True and the source is a file, automatically rewinds to beginning on EOF.
    """

    def __init__(
        self,
        source: Union[str, int],
        redis_manager: RedisManager,
        camera_id: str = "camera_0",
        target_fps: float = 15.0,
        max_reconnect_attempts: Optional[int] = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        loop_file: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source = self._normalize_source(source)
        self.redis_manager = redis_manager
        self.camera_id = camera_id
        self.target_fps = max(0.1, float(target_fps))
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_backoff = max(0.01, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.backoff_factor = max(1.0, float(backoff_factor))
        self.loop_file = loop_file
        self.metadata = metadata or {}

        self._frame_interval = 1.0 / self.target_fps
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry & state
        self._frame_count = 0
        self._dropped_frames = 0
        self._reconnect_count = 0
        self._is_connected = False
        self._current_backoff = self.initial_backoff

    @staticmethod
    def _normalize_source(source: Union[str, int]) -> Union[str, int]:
        """Normalize camera index or file path / RTSP URL."""
        if isinstance(source, int):
            return source
        if isinstance(source, str):
            # Check if string is an integer (e.g., "0", "1")
            stripped = source.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        raise ValueError(f"Invalid video source type: {type(source)}. Expected int or str.")

    def _is_file_source(self) -> bool:
        """Determine if source points to a local video file."""
        if isinstance(self.source, int):
            return False
        if any(self.source.startswith(proto) for proto in ("rtsp://", "http://", "https://")):
            return False
        # Likely a local file
        return os.path.isfile(self.source) or any(
            self.source.lower().endswith(ext) for ext in (".mp4", ".avi", ".mkv", ".mov")
        )

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Attempt to open cv2.VideoCapture for the configured source."""
        logger.info("Opening video source: %s (camera_id=%s)", self.source, self.camera_id)
        try:
            cap = cv2.VideoCapture(self.source)
            if cap is not None and cap.isOpened():
                if isinstance(self.source, int):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._is_connected = True
                self._current_backoff = self.initial_backoff
                logger.info("Successfully opened video source: %s", self.source)
                return cap
            if cap is not None:
                cap.release()
        except Exception as exc:
            logger.warning("Exception opening video source %s: %s", self.source, exc)
        self._is_connected = False
        return None

    def _worker_loop(self) -> None:
        """Main background ingestion loop with frame rate throttling and reconnect logic."""
        logger.info("Starting ingestion worker thread for camera_id=%s", self.camera_id)
        consecutive_failures = 0

        while not self._stop_event.is_set():
            cap = self._open_capture()

            if cap is None:
                consecutive_failures += 1
                self._reconnect_count += 1
                logger.warning(
                    "Connection failed for %s (attempt %d). Backing off for %.2fs...",
                    self.camera_id,
                    consecutive_failures,
                    self._current_backoff,
                )

                if (
                    self.max_reconnect_attempts is not None
                    and consecutive_failures >= self.max_reconnect_attempts
                ):
                    logger.error(
                        "Max reconnection attempts (%d) reached for %s. Exiting worker.",
                        self.max_reconnect_attempts,
                        self.camera_id,
                    )
                    break

                # Non-busy wait with instant stop response
                self._stop_event.wait(self._current_backoff)
                self._current_backoff = min(
                    self._current_backoff * self.backoff_factor, self.max_backoff
                )
                continue

            # Connection succeeded, reset failure count
            consecutive_failures = 0
            is_file = self._is_file_source()

            # Active frame capture loop
            while not self._stop_event.is_set() and cap.isOpened():
                loop_start = time.perf_counter()

                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    logger.warning("Exception during frame read from %s: %s", self.source, exc)
                    ret, frame = False, None

                if not ret or frame is None:
                    if is_file and self.loop_file:
                        logger.debug("Reached EOF for file %s. Rewinding to start.", self.source)
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        except Exception as exc:
                            logger.warning("Exception rewinding file %s: %s", self.source, exc)
                            ret, frame = False, None

                        if not ret or frame is None:
                            logger.warning("Failed to rewind file %s; reconnecting.", self.source)
                            break
                    else:
                        logger.warning("Stream disconnected for %s; triggering reconnect.", self.source)
                        break

                # Downscale large resolution frames to standard 800px width for fast pipeline throughput
                if frame.shape[1] > 800:
                    scale = 800.0 / frame.shape[1]
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                self._frame_count += 1
                meta = dict(self.metadata)
                meta.update(
                    {
                        "frame_index": self._frame_count,
                        "source": str(self.source),
                        "ingest_timestamp": time.time(),
                        "fps_target": self.target_fps,
                    }
                )

                # Push frame to Redis
                try:
                    self.redis_manager.push_frame(
                        camera_id=self.camera_id,
                        frame=frame,
                        metadata=meta,
                    )
                except Exception as exc:
                    self._dropped_frames += 1
                    logger.error("Failed to push frame #%d: %s", self._frame_count, exc)

                # Frame rate throttling: maintain target FPS
                elapsed = time.perf_counter() - loop_start
                delay = self._frame_interval - elapsed
                if delay > 0:
                    self._stop_event.wait(delay)

            # Cleanup capture before potential reconnect or exit
            try:
                cap.release()
            except Exception:
                pass
            self._is_connected = False

            if not self._stop_event.is_set():
                consecutive_failures += 1
                self._reconnect_count += 1
                logger.info(
                    "Stream stopped. Auto-reconnecting in %.2fs (attempt %d)...",
                    self._current_backoff,
                    consecutive_failures,
                )
                self._stop_event.wait(self._current_backoff)
                self._current_backoff = min(
                    self._current_backoff * self.backoff_factor, self.max_backoff
                )

        logger.info(
            "Ingestion worker finished for %s. Total frames: %d, Reconnects: %d, Dropped: %d",
            self.camera_id,
            self._frame_count,
            self._reconnect_count,
            self._dropped_frames,
        )

    def start(self) -> None:
        """Start the background stream capture worker."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Stream ingestion worker for %s is already running.", self.camera_id)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"StreamIngestor-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Launched StreamIngestor worker thread for %s", self.camera_id)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker thread to stop and wait for termination."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("StreamIngestor worker thread stopped for %s", self.camera_id)

    @property
    def is_alive(self) -> bool:
        """Check if worker thread is active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        """Total number of frames read and processed."""
        return self._frame_count

    @property
    def reconnect_count(self) -> int:
        """Total number of reconnection attempts triggered."""
        return self._reconnect_count

    @property
    def dropped_frames(self) -> int:
        """Total number of frames that failed to push to Redis."""
        return self._dropped_frames

    @property
    def is_connected(self) -> bool:
        """Check whether the video stream is currently connected."""
        return self._is_connected

    def __enter__(self) -> "StreamIngestor":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
