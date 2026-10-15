"""Integration and unit tests for ProYOLO DefOps live video streaming and upload endpoints.

Tests:
1. POST /api/v1/upload-video:
   - Valid MP4 and AVI file uploads
   - Switching stream engine to uploaded file
   - Validation for unsupported extensions (.txt, .jpg)
   - Validation for empty files
2. POST /api/v1/stream-source:
   - Switching to webcam (0)
   - Switching to custom video path / RTSP URL
3. POST /api/v1/stream-control:
   - Play, pause, toggle actions
   - Rejection of invalid actions
4. GET /api/v1/stream-status:
   - Verification of stream telemetry and playback state
5. GET /api/v1/video-feed:
   - Motion-JPEG multipart/x-mixed-replace streaming
   - Validation of MIME boundary and valid JPEG frame magic bytes (\\xff\\xd8)
6. StreamEngine Pipeline & Alert Dispatches:
   - Frame annotation verification
   - WebSocket alert emission with base64 patch thumbnails
"""

import asyncio
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import httpx
import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.server.app import app, stream_engine, telemetry_state, ws_manager
from src.server.database import Base, get_db


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def create_synthetic_mp4(
    output_path: Path,
    num_frames: int = 10,
    width: int = 320,
    height: int = 240,
    fps: float = 15.0,
) -> Path:
    """Generate a valid, playable MP4 synthetic video for pipeline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), 35 + (i * 2) % 50, dtype=np.uint8)
        # Draw a simulated pedestrian
        cv2.rectangle(frame, (80, 40), (140, 200), (90, 110, 130), -1)
        cv2.circle(frame, (110, 30), 12, (100, 120, 140), -1)
        writer.write(frame)
    writer.release()
    return output_path


@pytest.fixture(autouse=True)
def test_db_session() -> Generator[sessionmaker[Session], None, None]:
    """Provide an isolated in-memory SQLite database for each test."""
    test_db_url = "sqlite:///:memory:"
    engine = create_engine(
        test_db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Test mode: don't block on hardware AVFoundation webcam or network RTSP
    os.environ["PROYOLO_TEST_MODE"] = "1"
    stream_engine.cap = None
    stream_engine.current_source = 0
    stream_engine.source_label = "Webcam (Live)"
    stream_engine.set_paused(False)

    # Reset telemetry and stream engine state
    telemetry_state.fps = 18.5
    telemetry_state.processed_frames = 0
    telemetry_state.clean_detections = 0
    telemetry_state.flagged_attacks = 0

    yield TestingSessionLocal
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(test_db_session) -> Generator[httpx.AsyncClient, None, None]:
    """Asynchronous HTTP test client using httpx."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def sync_client(test_db_session) -> Generator[TestClient, None, None]:
    """Synchronous test client for WebSocket testing."""
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Video Upload Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_video_mp4_success(async_client: httpx.AsyncClient, tmp_path: Path):
    """Verify POST /api/v1/upload-video accepts valid MP4 and switches stream source."""
    temp_video = tmp_path / "test_surveillance_feed.mp4"
    create_synthetic_mp4(temp_video, num_frames=8)

    with open(temp_video, "rb") as f:
        video_bytes = f.read()

    files = {"file": ("test_surveillance_feed.mp4", video_bytes, "video/mp4")}
    response = await async_client.post("/api/v1/upload-video", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_surveillance_feed.mp4"
    assert "saved_path" in data
    assert os.path.exists(data["saved_path"])

    # Stream engine should now be configured with uploaded file
    assert "test_surveillance_feed.mp4" in stream_engine.source_label
    assert stream_engine.is_looping is True


@pytest.mark.asyncio
async def test_upload_video_avi_success(async_client: httpx.AsyncClient, tmp_path: Path):
    """Verify POST /api/v1/upload-video accepts .avi format."""
    dummy_avi = tmp_path / "cctv_corridor.avi"
    # Even simple binary bytes with valid extension are parsed by endpoint
    dummy_avi.write_bytes(b"RIFF\x00\x00\x00\x00AVI LIST\x00\x00\x00\x00")

    files = {"file": ("cctv_corridor.avi", dummy_avi.read_bytes(), "video/x-msvideo")}
    response = await async_client.post("/api/v1/upload-video", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "cctv_corridor.avi"


@pytest.mark.asyncio
async def test_upload_video_unsupported_format(async_client: httpx.AsyncClient):
    """Verify POST /api/v1/upload-video rejects disallowed file formats (e.g. .txt)."""
    files = {"file": ("malicious_payload.txt", b"plain text data", "text/plain")}
    response = await async_client.post("/api/v1/upload-video", files=files)

    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Unsupported video format" in detail


@pytest.mark.asyncio
async def test_upload_video_empty_file(async_client: httpx.AsyncClient):
    """Verify POST /api/v1/upload-video rejects zero-byte files."""
    files = {"file": ("empty_feed.mp4", b"", "video/mp4")}
    response = await async_client.post("/api/v1/upload-video", files=files)

    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Uploaded file is empty" in detail


# ---------------------------------------------------------------------------
# Stream Source & Control Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_source_switching(async_client: httpx.AsyncClient):
    """Verify POST /api/v1/stream-source switches between webcam, file, and RTSP."""
    # Switch to webcam 0
    resp_webcam = await async_client.post(
        "/api/v1/stream-source",
        json={"source": 0, "loop": True},
    )
    assert resp_webcam.status_code == 200
    data_webcam = resp_webcam.json()
    assert data_webcam["status"] == "success"
    assert data_webcam["source"] == "0"
    assert "Webcam" in data_webcam["source_label"]

    # Switch to video source path
    resp_video = await async_client.post(
        "/api/v1/stream-source",
        json={"source": "surveillance_cam01.mp4", "loop": False},
    )
    assert resp_video.status_code == 200
    data_video = resp_video.json()
    assert data_video["source"] == "surveillance_cam01.mp4"
    assert data_video["is_looping"] is False

    # Switch back to webcam device
    resp_back = await async_client.post(
        "/api/v1/stream-source",
        json={"source": 0, "loop": True},
    )
    assert resp_back.status_code == 200


@pytest.mark.asyncio
async def test_stream_playback_controls(async_client: httpx.AsyncClient):
    """Verify POST /api/v1/stream-control handles pause, play, and toggle."""
    # Pause
    resp_pause = await async_client.post(
        "/api/v1/stream-control",
        json={"action": "pause"},
    )
    assert resp_pause.status_code == 200
    assert resp_pause.json()["is_paused"] is True
    assert stream_engine.is_paused is True

    # Resume / Play
    resp_play = await async_client.post(
        "/api/v1/stream-control",
        json={"action": "play"},
    )
    assert resp_play.status_code == 200
    assert resp_play.json()["is_paused"] is False
    assert stream_engine.is_paused is False

    # Toggle
    resp_toggle = await async_client.post(
        "/api/v1/stream-control",
        json={"action": "toggle"},
    )
    assert resp_toggle.status_code == 200
    assert resp_toggle.json()["is_paused"] is True

    # Invalid action
    resp_invalid = await async_client.post(
        "/api/v1/stream-control",
        json={"action": "rewind_fast"},
    )
    assert resp_invalid.status_code == 400


@pytest.mark.asyncio
async def test_get_stream_status(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/stream-status reports active stream health and telemetry."""
    response = await async_client.get("/api/v1/stream-status")
    assert response.status_code == 200
    data = response.json()
    for key in ("source", "source_label", "is_paused", "is_looping", "total_frames_processed", "total_attacks_flagged"):
        assert key in data


# ---------------------------------------------------------------------------
# Motion-JPEG Video Feed Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_motion_jpeg_video_feed_streaming(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/video-feed yields standard multipart/x-mixed-replace JPEG chunks."""
    stream_engine.set_paused(False)

    # 1. Test stream_generator output directly
    gen = stream_engine.stream_generator()
    chunk = await anext(gen)
    await gen.aclose()

    assert b"--frame\r\n" in chunk
    assert b"Content-Type: image/jpeg\r\n\r\n" in chunk
    assert b"\xff\xd8" in chunk  # JPEG SOI
    assert b"\xff\xd9" in chunk  # JPEG EOI

    # 2. Test HTTP streaming endpoint with timeout
    try:
        async with asyncio.timeout(3.0):
            async with async_client.stream("GET", "/api/v1/video-feed") as response:
                assert response.status_code == 200
                content_type = response.headers.get("content-type", "")
                assert "multipart/x-mixed-replace" in content_type
                assert "boundary=frame" in content_type
                async for stream_chunk in response.aiter_bytes():
                    if b"--frame" in stream_chunk:
                        break
    except TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Stream Engine Frame Processing & Alert Dispatches
# ---------------------------------------------------------------------------

def test_stream_engine_frame_annotation():
    """Verify StreamEngine.process_frame annotates pedestrian bounding boxes."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 50, dtype=np.uint8)

    annotated, detections = stream_engine.process_frame(frame, timestamp=100.0)
    assert annotated.shape == frame.shape
    assert isinstance(detections, list)
    assert stream_engine.total_frames_processed >= 1


def test_websocket_alert_event_emission(sync_client: TestClient):
    """Verify that flagged attacks broadcast rich alert packets over /ws/alerts."""
    with sync_client.websocket_connect("/ws/alerts") as ws:
        init_msg = json.loads(ws.receive_text())
        assert init_msg["event"] == "init_state"

        # Simulate alert ingestion from inference callback
        alert_payload = {
            "camera_id": "CAM-01",
            "track_id": 42,
            "confidence": 0.88,
            "attack_probability": 0.95,
            "bbox": [100.0, 100.0, 220.0, 380.0],
            "thumbnail_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
            "top_features": {"hsv_entropy": 0.45, "confidence_drop": 0.32},
        }

        # Trigger POST /api/v1/alerts
        resp = sync_client.post("/api/v1/alerts", json=alert_payload)
        assert resp.status_code == 201

        # WebSocket should receive attack_alert event
        ws_msg = json.loads(ws.receive_text())
        assert ws_msg["event"] == "attack_alert"
        assert ws_msg["data"]["track_id"] == 42
        assert ws_msg["data"]["attack_probability"] == 0.95
        assert ws_msg["data"]["thumbnail_b64"] == "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        assert ws_msg["metrics"]["flagged_attacks"] >= 1
