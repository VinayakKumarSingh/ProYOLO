"""Unit and integration tests for ProYOLO telemetry backend and WebSocket broadcaster.

Tests:
- REST endpoints: POST /api/v1/alerts, GET /api/v1/alerts, GET /api/v1/metrics, POST /api/v1/metrics
- Live frame telemetry: POST /api/v1/frame_telemetry
- WebSocket streaming: WS /ws/alerts (connection, initial handshake, ping-pong, alert broadcast)
- Static dashboard serving: GET /
"""

import asyncio
from typing import Generator
import pytest
import pytest_asyncio
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.server.app import app, telemetry_state
from src.server.database import Base, get_db


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
    
    # Reset telemetry counters for test isolation
    telemetry_state.fps = 18.5
    telemetry_state.processed_frames = 0
    telemetry_state.clean_detections = 0
    telemetry_state.flagged_attacks = 0
    telemetry_state.cpu_usage = 24.8
    telemetry_state.latency_ms = 14.2

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
# REST Endpoint Tests (using httpx)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_metrics(async_client: httpx.AsyncClient):
    """Verify that GET /api/v1/metrics returns system telemetry."""
    response = await async_client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "fps" in data
    assert "processed_frames" in data
    assert "clean_detections" in data
    assert "flagged_attacks" in data
    assert "cpu_usage" in data
    assert "latency_ms" in data
    assert data["fps"] > 0


@pytest.mark.asyncio
async def test_update_metrics(async_client: httpx.AsyncClient):
    """Verify that POST /api/v1/metrics updates telemetry state."""
    update_payload = {
        "fps": 22.4,
        "processed_frames": 1500,
        "clean_detections": 320,
        "flagged_attacks": 12,
        "cpu_usage": 35.5,
        "latency_ms": 11.8,
    }
    response = await async_client.post("/api/v1/metrics", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fps"] == 22.4
    assert data["processed_frames"] == 1500
    assert data["clean_detections"] == 320
    assert data["flagged_attacks"] == 12

    # Verify subsequent GET returns the updated values
    get_res = await async_client.get("/api/v1/metrics")
    assert get_res.json()["fps"] == 22.4


@pytest.mark.asyncio
async def test_post_alert_success(async_client: httpx.AsyncClient):
    """Verify successful ingestion of an attack alert via POST /api/v1/alerts."""
    payload = {
        "camera_id": "cam-01",
        "track_id": 42,
        "confidence": 0.58,
        "attack_probability": 0.945,
        "bbox": [120.0, 80.0, 240.0, 360.0],
        "thumbnail_b64": "data:image/jpeg;base64,sample_jpeg_data",
        "top_features": {
            "hsv_val_entropy": 0.44,
            "confidence_drop": 0.38,
            "laplacian_var": 0.22,
        },
    }
    response = await async_client.post("/api/v1/alerts", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["camera_id"] == "cam-01"
    assert data["track_id"] == 42
    assert data["confidence"] == 0.58
    assert data["attack_probability"] == 0.945
    assert data["bbox"] == [120.0, 80.0, 240.0, 360.0]
    assert data["thumbnail_b64"] == "data:image/jpeg;base64,sample_jpeg_data"
    assert data["top_features"]["hsv_val_entropy"] == 0.44
    assert "created_at" in data

    # Verify flagged attack count increased in metrics
    metrics_res = await async_client.get("/api/v1/metrics")
    assert metrics_res.json()["flagged_attacks"] == 1


@pytest.mark.asyncio
async def test_post_alert_validation_errors(async_client: httpx.AsyncClient):
    """Verify input validation for invalid alert payloads."""
    # Invalid confidence > 1.0
    res1 = await async_client.post("/api/v1/alerts", json={
        "camera_id": "cam-01",
        "track_id": 1,
        "confidence": 1.5,
        "attack_probability": 0.9,
        "bbox": [10.0, 10.0, 50.0, 50.0],
    })
    assert res1.status_code == 422

    # Invalid attack probability < 0.0
    res2 = await async_client.post("/api/v1/alerts", json={
        "camera_id": "cam-01",
        "track_id": 1,
        "confidence": 0.8,
        "attack_probability": -0.1,
        "bbox": [10.0, 10.0, 50.0, 50.0],
    })
    assert res2.status_code == 422

    # Malformed bbox with only 2 items instead of 4
    res3 = await async_client.post("/api/v1/alerts", json={
        "camera_id": "cam-01",
        "track_id": 1,
        "confidence": 0.8,
        "attack_probability": 0.7,
        "bbox": [10.0, 10.0],
    })
    assert res3.status_code == 422


@pytest.mark.asyncio
async def test_get_alerts_pagination_and_filtering(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/alerts pagination, camera_id filtering, and probability filtering."""
    # Seed 5 alerts: 3 from cam-01, 2 from cam-02
    for i in range(5):
        cam = "cam-01" if i < 3 else "cam-02"
        prob = 0.70 + (i * 0.05)
        await async_client.post("/api/v1/alerts", json={
            "camera_id": cam,
            "track_id": 100 + i,
            "confidence": 0.80,
            "attack_probability": prob,
            "bbox": [10.0 * i, 10.0 * i, 50.0 * i, 50.0 * i],
        })

    # Test pagination (limit=2)
    res_page1 = await async_client.get("/api/v1/alerts?skip=0&limit=2")
    assert res_page1.status_code == 200
    data_page1 = res_page1.json()
    assert data_page1["total"] == 5
    assert len(data_page1["items"]) == 2
    assert data_page1["skip"] == 0
    assert data_page1["limit"] == 2

    # Test page 2 (skip=2, limit=2)
    res_page2 = await async_client.get("/api/v1/alerts?skip=2&limit=2")
    assert res_page2.status_code == 200
    data_page2 = res_page2.json()
    assert len(data_page2["items"]) == 2
    assert data_page2["items"][0]["id"] != data_page1["items"][0]["id"]

    # Test camera_id filter
    res_cam2 = await async_client.get("/api/v1/alerts?camera_id=cam-02")
    assert res_cam2.status_code == 200
    data_cam2 = res_cam2.json()
    assert data_cam2["total"] == 2
    assert all(item["camera_id"] == "cam-02" for item in data_cam2["items"])

    # Test min_probability filter (>= 0.85)
    res_prob = await async_client.get("/api/v1/alerts?min_probability=0.85")
    assert res_prob.status_code == 200
    data_prob = res_prob.json()
    assert all(item["attack_probability"] >= 0.85 for item in data_prob["items"])


@pytest.mark.asyncio
async def test_frame_telemetry_endpoint(async_client: httpx.AsyncClient):
    """Verify live frame detection pushing updates counters."""
    payload = {
        "camera_id": "cam-01",
        "fps": 19.2,
        "latency_ms": 13.5,
        "detections": [
            {
                "track_id": 1,
                "bbox": [50.0, 60.0, 150.0, 200.0],
                "confidence": 0.85,
                "is_attack": False,
                "attack_probability": 0.10,
            },
            {
                "track_id": 2,
                "bbox": [200.0, 80.0, 320.0, 240.0],
                "confidence": 0.45,
                "is_attack": True,
                "attack_probability": 0.92,
            },
        ],
    }
    response = await async_client.post("/api/v1/frame_telemetry", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "broadcasted"

    # Verify metrics updated
    metrics = (await async_client.get("/api/v1/metrics")).json()
    assert metrics["fps"] == 19.2
    assert metrics["latency_ms"] == 13.5
    assert metrics["processed_frames"] == 1
    assert metrics["clean_detections"] == 1


@pytest.mark.asyncio
async def test_dashboard_static_page(async_client: httpx.AsyncClient):
    """Verify GET / returns the dashboard HTML interface."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    content = response.text
    assert "ProYOLO" in content
    assert "streamCanvas" in content
    assert "alertsContainer" in content


# ---------------------------------------------------------------------------
# WebSocket Tests
# ---------------------------------------------------------------------------

def test_websocket_initial_handshake_and_ping_pong(sync_client: TestClient):
    """Verify WebSocket connection sends initial state and responds to ping."""
    with sync_client.websocket_connect("/ws/alerts") as websocket:
        init_message = websocket.receive_json()
        assert init_message["event"] == "init_state"
        assert "metrics" in init_message
        assert "timestamp" in init_message

        # Send ping and receive pong
        websocket.send_json({"type": "ping"})
        response = websocket.receive_json()
        assert response["type"] == "pong"
        assert "time" in response


def test_websocket_broadcast_on_alert_ingestion(sync_client: TestClient):
    """Verify that posting an alert broadcasts attack_alert event to connected WebSockets."""
    with sync_client.websocket_connect("/ws/alerts") as websocket:
        # Read the initial connection message
        _ = websocket.receive_json()

        # Ingest an alert via REST
        alert_payload = {
            "camera_id": "cam-radar-01",
            "track_id": 77,
            "confidence": 0.62,
            "attack_probability": 0.96,
            "bbox": [100.0, 150.0, 250.0, 450.0],
            "thumbnail_b64": "data:image/jpeg;base64,radar_frame",
            "top_features": {"hsv_val_entropy": 0.45},
        }
        post_res = sync_client.post("/api/v1/alerts", json=alert_payload)
        assert post_res.status_code == 201

        # Check WebSocket received the broadcasted event
        broadcast_msg = websocket.receive_json()
        assert broadcast_msg["event"] == "attack_alert"
        assert broadcast_msg["data"]["camera_id"] == "cam-radar-01"
        assert broadcast_msg["data"]["track_id"] == 77
        assert broadcast_msg["data"]["attack_probability"] == 0.96
        assert broadcast_msg["data"]["bbox"] == [100.0, 150.0, 250.0, 450.0]
        assert broadcast_msg["metrics"]["flagged_attacks"] >= 1
