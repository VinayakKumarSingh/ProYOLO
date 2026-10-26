"""Redis Manager for asynchronous video frame queueing and latency control.

Handles JPEG compression, queue capping (dropping oldest frames),
and non-blocking frame retrieval between stream ingestion and inference workers.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, NamedTuple, Optional, Union

import cv2
import numpy as np

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None  # type: ignore

try:
    import fakeredis
except ImportError:  # pragma: no cover
    fakeredis = None  # type: ignore

logger = logging.getLogger(__name__)


class FramePacket(NamedTuple):
    """Container for popped frame data supporting tuple unpacking and attribute access."""

    camera_id: str
    frame: np.ndarray
    metadata: Dict[str, Any]

    @property
    def timestamp(self) -> float:
        """Convenience property to extract the capture timestamp."""
        return float(self.metadata.get("timestamp", 0.0))


class RedisManager:
    """Manages Redis queue operations for video streaming with latency mitigation.

    Attributes:
        host: Hostname of the Redis server.
        port: Port of the Redis server.
        db: Redis database index.
        queue_name: Key name used for the frame queue list.
        max_queue_size: Maximum frames retained in queue; oldest frames are dropped.
        jpeg_quality: Quality factor (1-100) for OpenCV JPEG compression.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        queue_name: str = "video_frames",
        max_queue_size: int = 30,
        jpeg_quality: int = 80,
        redis_client: Optional[Any] = None,
        socket_timeout: float = 2.0,
        allow_mock_fallback: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.queue_name = queue_name
        self.max_queue_size = max(1, int(max_queue_size))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.allow_mock_fallback = allow_mock_fallback

        if redis_client is not None:
            self.client = redis_client
        else:
            self.client = self._init_redis_client(socket_timeout=socket_timeout)

    def _init_redis_client(self, socket_timeout: float) -> Any:
        """Initialize connection to Redis with optional mock fallback."""
        if redis is None:
            raise ImportError(
                "The 'redis' package is not installed. Please install it with 'pip install redis'."
            )

        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                decode_responses=False,
            )
            client.ping()
            logger.info("Connected to Redis at %s:%d/%d", self.host, self.port, self.db)
            return client
        except Exception as exc:
            if self.allow_mock_fallback and fakeredis is not None:
                logger.warning(
                    "Failed to connect to Redis at %s:%d (%s). Falling back to fakeredis in-memory mock.",
                    self.host,
                    self.port,
                    exc,
                )
                return fakeredis.FakeRedis(decode_responses=False)
            raise ConnectionError(
                f"Could not connect to Redis at {self.host}:{self.port}/{self.db}: {exc}"
            ) from exc

    def push_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Encode a video frame to JPEG (quality=80) and push it to the Redis queue.

        Drops the oldest frames if queue length exceeds max_queue_size to prevent
        latency accumulation.

        Args:
            camera_id: Unique identifier for the video source camera.
            frame: BGR image frame as a NumPy array.
            metadata: Optional dictionary of frame-associated telemetry.

        Returns:
            The size of the compressed JPEG buffer in bytes.

        Raises:
            ValueError: If the frame is invalid or cannot be encoded.
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: expected a non-empty numpy.ndarray.")

        # 1. Encode frame to JPEG with configured quality (default 80)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded_buf = cv2.imencode(".jpg", frame, encode_params)
        if not success or encoded_buf is None:
            raise ValueError("OpenCV failed to encode frame to JPEG format.")

        jpeg_bytes = encoded_buf.tobytes()

        # 2. Prepare payload
        payload_meta = dict(metadata) if metadata is not None else {}
        timestamp = payload_meta.setdefault("timestamp", time.time())

        payload_dict = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "metadata": payload_meta,
            "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
        }
        payload_raw = json.dumps(payload_dict).encode("utf-8")

        # 3. Push to queue tail (RPUSH)
        self.client.rpush(self.queue_name, payload_raw)

        # 4. Enforce max_queue_size (drop oldest frames from left)
        self._trim_overflow()

        return len(jpeg_bytes)

    def pop_frame(self, latest_only: bool = False) -> Optional[FramePacket]:
        """Perform a non-blocking pop from the frame queue.

        If the queue length exceeds max_queue_size, oldest frames are dropped.

        Args:
            latest_only: If True, retrieves the most recent frame and discards older
                         pending frames. Defaults to False (FIFO order).

        Returns:
            A FramePacket containing (camera_id, frame, metadata), or None if queue is empty.
        """
        # Ensure queue size does not exceed cap before popping
        self._trim_overflow()

        if latest_only:
            raw_payload = self.client.rpop(self.queue_name)
            # Clear remaining stale frames if any
            if raw_payload is not None:
                self.client.delete(self.queue_name)
        else:
            raw_payload = self.client.lpop(self.queue_name)

        if raw_payload is None:
            return None

        # Decode payload
        try:
            payload_dict = json.loads(raw_payload.decode("utf-8"))
            camera_id = payload_dict["camera_id"]
            metadata = payload_dict.get("metadata", {})
            if "timestamp" in payload_dict and "timestamp" not in metadata:
                metadata["timestamp"] = payload_dict["timestamp"]

            jpeg_bytes = base64.b64decode(payload_dict["jpeg_b64"])
            np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode JPEG bytes into valid frame image.")
                return None

            return FramePacket(camera_id=camera_id, frame=frame, metadata=metadata)
        except Exception as exc:
            logger.error("Error deserializing frame packet from Redis: %s", exc)
            return None

    def _trim_overflow(self) -> int:
        """Drop oldest frames if queue length exceeds max_queue_size.

        Returns:
            Number of frames dropped.
        """
        try:
            q_len = self.client.llen(self.queue_name)
            if q_len > self.max_queue_size:
                # Keep the rightmost max_queue_size items (latest frames)
                self.client.ltrim(self.queue_name, -self.max_queue_size, -1)
                dropped = q_len - self.max_queue_size
                logger.debug(
                    "Queue %s capped at %d; dropped %d oldest frames.",
                    self.queue_name,
                    self.max_queue_size,
                    dropped,
                )
                return dropped
        except Exception as exc:
            logger.warning("Error checking or trimming Redis queue length: %s", exc)
        return 0

    def get_queue_length(self) -> int:
        """Return the current number of frames in the queue."""
        try:
            return int(self.client.llen(self.queue_name))
        except Exception:
            return 0

    def clear_queue(self) -> None:
        """Purge all frames from the active queue."""
        try:
            self.client.delete(self.queue_name)
        except Exception as exc:
            logger.warning("Failed to clear queue %s: %s", self.queue_name, exc)

    def ping(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def publish_alert(self, alert_payload: Dict[str, Any], channel: str = "attack_alerts") -> bool:
        """Publish an attack alert JSON payload to a Redis pub/sub channel.

        Args:
            alert_payload: Dictionary containing confirmed attack alert details.
            channel: Redis pub/sub channel name (default: 'attack_alerts').

        Returns:
            True if published successfully, False otherwise.
        """
        try:
            payload_str = json.dumps(alert_payload)
            if hasattr(self.client, "publish"):
                self.client.publish(channel, payload_str)
            return True
        except Exception as exc:
            logger.debug("Failed to publish alert to Redis channel %s: %s", channel, exc)
            return False

    def close(self) -> None:
        """Close connection to Redis."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception as exc:
            logger.debug("Exception during Redis client close: %s", exc)
