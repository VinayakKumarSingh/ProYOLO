"""Unit and integration tests for RedisManager and StreamIngestor.

Mocks Redis to verify frame encoding to JPEG (quality=80),
queue size capping (dropping oldest frames), non-blocking popping,
and graceful background worker thread shutdown.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import cv2
import fakeredis
import numpy as np
import pytest

from src.pipeline.redis_manager import FramePacket, RedisManager
from src.pipeline.stream_ingestion import StreamIngestor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_redis():
    """Provides an isolated in-memory fakeredis client for testing."""
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def sample_bgr_frame():
    """Generates a synthetic 64x64 BGR test frame with color gradient."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)  # Blue gradient
    frame[:, :, 1] = 128  # Green constant
    frame[:, :, 2] = np.linspace(255, 0, 64, dtype=np.uint8)  # Red gradient
    return frame


@pytest.fixture
def redis_manager(mock_redis):
    """Instantiates a RedisManager backed by fakeredis."""
    return RedisManager(
        queue_name="test_video_queue",
        max_queue_size=30,
        jpeg_quality=80,
        redis_client=mock_redis,
    )


# ============================================================================
# RedisManager Tests
# ============================================================================


class TestRedisManager:
    """Tests for RedisManager JPEG compression, queue capping, and non-blocking pops."""

    def test_push_and_pop_single_frame(self, redis_manager, sample_bgr_frame):
        """Verify pushing a frame encodes it and popping reconstructs valid image."""
        meta = {"source": "cam_front", "frame_idx": 1}
        num_bytes = redis_manager.push_frame(
            camera_id="cam_01", frame=sample_bgr_frame, metadata=meta
        )

        assert num_bytes > 0
        assert redis_manager.get_queue_length() == 1

        packet = redis_manager.pop_frame()
        assert packet is not None
        assert isinstance(packet, FramePacket)
        assert packet.camera_id == "cam_01"
        assert packet.metadata["source"] == "cam_front"
        assert packet.metadata["frame_idx"] == 1
        assert "timestamp" in packet.metadata

        # Frame should match dimensions and uint8 dtype
        assert isinstance(packet.frame, np.ndarray)
        assert packet.frame.shape == sample_bgr_frame.shape
        assert packet.frame.dtype == np.uint8

        # After popping, queue should be empty
        assert redis_manager.get_queue_length() == 0
        assert redis_manager.pop_frame() is None

    def test_tuple_unpacking(self, redis_manager, sample_bgr_frame):
        """Verify FramePacket can be unpacked directly as (camera_id, frame, metadata)."""
        redis_manager.push_frame(
            camera_id="cam_02", frame=sample_bgr_frame, metadata={"key": "value"}
        )
        packet = redis_manager.pop_frame()
        assert packet is not None

        camera_id, frame, metadata = packet
        assert camera_id == "cam_02"
        assert isinstance(frame, np.ndarray)
        assert metadata["key"] == "value"

    def test_jpeg_encoding_quality_80(self, mock_redis, sample_bgr_frame):
        """Verify frame is encoded as JPEG with SOI and EOI markers and quality=80 is used."""
        with patch("cv2.imencode", wraps=cv2.imencode) as spy_imencode:
            mgr = RedisManager(
                queue_name="quality_check_queue",
                jpeg_quality=80,
                redis_client=mock_redis,
            )
            mgr.push_frame("cam_01", sample_bgr_frame)

            spy_imencode.assert_called_once()
            args, _ = spy_imencode.call_args
            assert args[0] == ".jpg"
            # cv2.IMWRITE_JPEG_QUALITY is flag 1
            expected_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            assert args[2] == expected_params

        # Verify raw bytes in redis are indeed a valid JPEG image (SOI = 0xFFD8, EOI = 0xFFD9)
        raw_payload = mock_redis.lindex("quality_check_queue", 0)
        assert raw_payload is not None
        payload_data = json.loads(raw_payload.decode("utf-8"))
        jpeg_bytes = base64.b64decode(payload_data["jpeg_b64"])

        assert jpeg_bytes[:2] == b"\xff\xd8", "Payload does not start with JPEG SOI marker"
        assert jpeg_bytes[-2:] == b"\xff\xd9", "Payload does not end with JPEG EOI marker"

    def test_queue_size_capping_drops_oldest(self, mock_redis, sample_bgr_frame):
        """Verify queue length never exceeds max_queue_size and oldest frames are dropped."""
        max_size = 5
        mgr = RedisManager(
            queue_name="capping_queue",
            max_queue_size=max_size,
            redis_client=mock_redis,
        )

        # Push 12 frames, each with an incrementing index
        total_pushed = 12
        for i in range(total_pushed):
            mgr.push_frame("cam_01", sample_bgr_frame, metadata={"frame_seq": i})

        # Queue length must be strictly capped at max_size
        assert mgr.get_queue_length() == max_size

        # The oldest frames (0 to 6) should have been dropped.
        # Popping in FIFO order should yield frames 7 to 11.
        expected_indices = list(range(total_pushed - max_size, total_pushed))
        for expected_idx in expected_indices:
            packet = mgr.pop_frame()
            assert packet is not None
            assert packet.metadata["frame_seq"] == expected_idx

        # Queue should now be empty
        assert mgr.get_queue_length() == 0
        assert mgr.pop_frame() is None

    def test_pop_latest_only(self, mock_redis, sample_bgr_frame):
        """Verify latest_only=True retrieves the most recent frame and flushes stale ones."""
        mgr = RedisManager(
            queue_name="latest_only_queue",
            max_queue_size=10,
            redis_client=mock_redis,
        )

        for i in range(5):
            mgr.push_frame("cam_01", sample_bgr_frame, metadata={"seq": i})

        assert mgr.get_queue_length() == 5
        # Pop latest
        packet = mgr.pop_frame(latest_only=True)
        assert packet is not None
        assert packet.metadata["seq"] == 4

        # Queue should have been flushed
        assert mgr.get_queue_length() == 0
        assert mgr.pop_frame() is None

    def test_invalid_frame_raises_error(self, redis_manager):
        """Verify non-ndarray or empty array inputs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid frame"):
            redis_manager.push_frame("cam_01", None)  # type: ignore

        with pytest.raises(ValueError, match="Invalid frame"):
            redis_manager.push_frame("cam_01", np.array([]))

    def test_mock_fallback_on_connection_error(self):
        """Verify allow_mock_fallback=True instantiates fakeredis when real server is offline."""
        mgr = RedisManager(
            host="127.0.0.1",
            port=59999,  # Unused port
            allow_mock_fallback=True,
            socket_timeout=0.2,
        )
        assert mgr.ping() is True
        assert mgr.client is not None

    def test_connection_error_raised_when_fallback_disabled(self):
        """Verify ConnectionError is raised when Redis is unreachable and fallback=False."""
        with pytest.raises(ConnectionError):
            RedisManager(
                host="127.0.0.1",
                port=59999,  # Unused port
                allow_mock_fallback=False,
                socket_timeout=0.2,
            )


# ============================================================================
# StreamIngestor Tests
# ============================================================================


class TestStreamIngestion:
    """Tests for StreamIngestor threading, rate throttling, reconnect, and shutdown."""

    def test_source_normalization(self, redis_manager):
        """Verify webcam indices and URL/path strings are properly normalized."""
        ing_int = StreamIngestor(source=0, redis_manager=redis_manager)
        assert ing_int.source == 0

        ing_str_int = StreamIngestor(source=" 1 ", redis_manager=redis_manager)
        assert ing_str_int.source == 1

        ing_rtsp = StreamIngestor(
            source="rtsp://192.168.1.10:554/live", redis_manager=redis_manager
        )
        assert ing_rtsp.source == "rtsp://192.168.1.10:554/live"

        with pytest.raises(ValueError):
            StreamIngestor(source=None, redis_manager=redis_manager)  # type: ignore

    def test_graceful_shutdown_with_mock_capture(self, redis_manager, sample_bgr_frame):
        """Verify background worker starts, pushes frames, and terminates cleanly on stop()."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            ingestor = StreamIngestor(
                source="test_cam.mp4",
                redis_manager=redis_manager,
                camera_id="cam_test",
                target_fps=20.0,
            )

            ingestor.start()
            assert ingestor.is_alive is True

            # Let worker run briefly
            time.sleep(0.15)

            # Signal stop and wait
            ingestor.stop(timeout=1.0)
            assert ingestor.is_alive is False
            assert ingestor.frame_count > 0
            assert redis_manager.get_queue_length() > 0

            # Verify mock_cap was released on shutdown
            mock_cap.release.assert_called()

    def test_context_manager_lifecycle(self, redis_manager, sample_bgr_frame):
        """Verify StreamIngestor works as a context manager and shuts down on block exit."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with StreamIngestor(
                source=0,
                redis_manager=redis_manager,
                camera_id="cam_context",
                target_fps=30.0,
            ) as ingestor:
                assert ingestor.is_alive is True
                time.sleep(0.08)

            # Once exited, worker thread must be shut down
            assert ingestor.is_alive is False

    def test_reconnect_exponential_backoff(self, redis_manager):
        """Verify failed capture triggers reconnection attempts and backoff increases."""
        # Setup VideoCapture mock that fails to open
        failing_cap = MagicMock()
        failing_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=failing_cap):
            ingestor = StreamIngestor(
                source="rtsp://invalid-host:554/stream",
                redis_manager=redis_manager,
                camera_id="cam_fail",
                initial_backoff=0.05,
                backoff_factor=2.0,
                max_backoff=0.5,
                max_reconnect_attempts=4,
            )

            ingestor.start()
            # Wait for attempts to complete
            time.sleep(0.4)
            ingestor.stop(timeout=1.0)

            assert ingestor.is_alive is False
            assert ingestor.reconnect_count >= 2

    def test_rate_throttling(self, redis_manager, sample_bgr_frame):
        """Verify that target_fps controls the rate of frame ingestion."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, sample_bgr_frame)

        target_fps = 10.0
        with patch("cv2.VideoCapture", return_value=mock_cap):
            ingestor = StreamIngestor(
                source=0,
                redis_manager=redis_manager,
                target_fps=target_fps,
            )
            ingestor.start()
            # Run for 0.35s -> at 10 FPS, expect ~3-5 frames
            time.sleep(0.35)
            ingestor.stop(timeout=1.0)

            # Frames captured should be close to 10 * 0.35 = 3.5 frames
            assert 2 <= ingestor.frame_count <= 6

    def test_file_looping_on_eof(self, redis_manager, sample_bgr_frame):
        """Verify video file loops automatically when EOF is encountered if loop_file=True."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        # Sequence: frame1, frame2, EOF (False), then loop -> continuous frames
        def frame_generator():
            yield (True, sample_bgr_frame)
            yield (True, sample_bgr_frame)
            yield (False, None)  # Triggers EOF
            while True:
                yield (True, sample_bgr_frame)

        gen = frame_generator()
        mock_cap.read.side_effect = lambda: next(gen)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            ingestor = StreamIngestor(
                source="sample.mp4",
                redis_manager=redis_manager,
                loop_file=True,
                target_fps=50.0,
            )
            ingestor.start()
            time.sleep(0.12)
            ingestor.stop(timeout=1.0)

            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0) should have been called upon EOF
            mock_cap.set.assert_any_call(cv2.CAP_PROP_POS_FRAMES, 0)
            assert ingestor.frame_count >= 3
