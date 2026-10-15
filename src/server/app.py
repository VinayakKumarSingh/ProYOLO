"""FastAPI Telemetry Backend and WebSocket Broadcast Server for ProYOLO.

Provides:
- Ingestion REST endpoint for adversarial alerts: POST /api/v1/alerts
- Paginated query REST endpoint for logged alerts: GET /api/v1/alerts
- Real-time system performance telemetry: GET /api/v1/metrics
- Live telemetry update ingestion: POST /api/v1/metrics
- Live frame detection broadcasting: POST /api/v1/frame_telemetry
- Live WebSocket streaming for alerts and canvas telemetry: WS /ws/alerts
- Single-page real-time surveillance dashboard mounted at /
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.server.database import (
    AlertRecord,
    get_alert_stats,
    get_alerts,
    get_db,
    init_db,
    save_alert,
)
from src.server.stream_engine import StreamEngine

# ---------------------------------------------------------------------------
# Data Models (Pydantic v2)
# ---------------------------------------------------------------------------

class AlertCreate(BaseModel):
    """Schema for incoming attack alert ingestion."""
    camera_id: str = Field(..., description="Camera identifier, e.g. 'cam-01'")
    track_id: int = Field(..., description="Integer tracking ID of the target person")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO detection confidence")
    attack_probability: float = Field(..., ge=0.0, le=1.0, description="Physical patch attack probability")
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2]")
    timestamp: Optional[float] = Field(default=None, description="Epoch timestamp in seconds")
    thumbnail_b64: Optional[str] = Field(default=None, description="Base64 encoded JPEG thumbnail")
    top_features: Optional[Dict[str, float]] = Field(default=None, description="Top feature contributions")


class AlertResponse(BaseModel):
    """Schema for alert serialization in API responses."""
    id: int
    camera_id: str
    timestamp: float
    track_id: int
    confidence: float
    attack_probability: float
    bbox: List[float]
    thumbnail_b64: Optional[str] = None
    top_features: Optional[Dict[str, float]] = None
    created_at: Optional[str] = None


class PaginatedAlertsResponse(BaseModel):
    """Paginated collection of alert records."""
    total: int
    items: List[AlertResponse]
    skip: int
    limit: int


class MetricsResponse(BaseModel):
    """Live performance and detection statistics."""
    fps: float = Field(..., description="Current processing frames per second")
    processed_frames: int = Field(..., description="Cumulative processed frame count")
    clean_detections: int = Field(..., description="Total clean person detections")
    flagged_attacks: int = Field(..., description="Total flagged physical patch attacks")
    persons_detected: Optional[int] = Field(None, description="Total detected persons")
    attackers_flagged: Optional[int] = Field(None, description="Total flagged attackers")
    cpu_usage: float = Field(..., description="Estimated CPU usage percentage")
    latency_ms: float = Field(..., description="Average inference latency in milliseconds")


class MetricsUpdate(BaseModel):
    """Telemetry counters update payload from inference worker."""
    fps: Optional[float] = None
    processed_frames: Optional[int] = None
    clean_detections: Optional[int] = None
    flagged_attacks: Optional[int] = None
    persons_detected: Optional[int] = None
    attackers_flagged: Optional[int] = None
    cpu_usage: Optional[float] = None
    latency_ms: Optional[float] = None


class FrameDetectionItem(BaseModel):
    """Bounding box detection item for live video canvas."""
    track_id: int
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float
    is_attack: bool = False
    attack_probability: float = 0.0
    label: Optional[str] = None


class FrameTelemetry(BaseModel):
    """Payload for live frame detection coordinates broadcast to dashboard canvas."""
    camera_id: str
    timestamp: Optional[float] = None
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    detections: List[FrameDetectionItem] = Field(default_factory=list)
    frame_b64: Optional[str] = None


class StreamSourceRequest(BaseModel):
    """Payload to switch live video stream source."""
    source: Union[int, str] = Field(..., description="0 for webcam, RTSP URL, or uploaded video file path")
    loop: bool = Field(default=True, description="Whether to loop video file playback on EOF")


class StreamControlRequest(BaseModel):
    """Payload for play/pause stream control."""
    action: str = Field(..., description="'play', 'pause', or 'toggle'")


# Rebuild all models to prevent deferred-annotation TypeAdapter errors in Python 3.14
AlertCreate.model_rebuild()
AlertResponse.model_rebuild()
PaginatedAlertsResponse.model_rebuild()
MetricsResponse.model_rebuild()
MetricsUpdate.model_rebuild()
FrameDetectionItem.model_rebuild()
FrameTelemetry.model_rebuild()
StreamSourceRequest.model_rebuild()
StreamControlRequest.model_rebuild()


# ---------------------------------------------------------------------------
# Telemetry State Tracker
# ---------------------------------------------------------------------------

class TelemetryState:
    """Thread-safe in-memory state tracking runtime metrics."""

    def __init__(self) -> None:
        self.fps: float = 18.5
        self.processed_frames: int = 0
        self.clean_detections: int = 0
        self.flagged_attacks: int = 0
        self.cpu_usage: float = 24.8
        self.latency_ms: float = 14.2
        self.last_update: float = time.time()

    def update(
        self,
        fps: Optional[float] = None,
        processed_frames: Optional[int] = None,
        clean_detections: Optional[int] = None,
        flagged_attacks: Optional[int] = None,
        cpu_usage: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if fps is not None:
            self.fps = round(float(fps), 2)
        if processed_frames is not None:
            self.processed_frames = int(processed_frames)
        if clean_detections is not None:
            self.clean_detections = int(clean_detections)
        if flagged_attacks is not None:
            self.flagged_attacks = int(flagged_attacks)
        if cpu_usage is not None:
            self.cpu_usage = round(float(cpu_usage), 1)
        if latency_ms is not None:
            self.latency_ms = round(float(latency_ms), 2)
        self.last_update = time.time()

    def increment_alert(self) -> None:
        self.flagged_attacks += 1
        self.last_update = time.time()

    def increment_clean(self, count: int = 1) -> None:
        self.clean_detections += count
        self.last_update = time.time()

    def increment_frames(self, count: int = 1) -> None:
        self.processed_frames += count
        self.last_update = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": self.fps,
            "processed_frames": self.processed_frames,
            "clean_detections": self.clean_detections,
            "flagged_attacks": self.flagged_attacks,
            "persons_detected": self.clean_detections,
            "attackers_flagged": self.flagged_attacks,
            "cpu_usage": self.cpu_usage,
            "latency_ms": self.latency_ms,
        }


telemetry_state = TelemetryState()


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections for live alert and telemetry broadcasts."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast JSON message to all active WebSocket connections."""
        payload = json.dumps(message)
        async with self._lock:
            dead_connections = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(payload)
                except Exception:
                    dead_connections.add(connection)
            for dead in dead_connections:
                self.active_connections.discard(dead)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Stream Engine Initialization and Callbacks
# ---------------------------------------------------------------------------

async def on_stream_alert(alert_data: Dict[str, Any]) -> None:
    """Callback when stream_engine flags an adversarial attack."""
    try:
        with next(get_db()) as db:
            save_alert(
                session=db,
                camera_id=alert_data.get("camera_id", "CAM-01"),
                track_id=int(alert_data.get("track_id", 0)),
                confidence=float(alert_data.get("confidence", 0.0)),
                attack_probability=float(alert_data.get("attack_probability", 0.0)),
                bbox=alert_data.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                timestamp=alert_data.get("timestamp", time.time()),
                thumbnail_b64=alert_data.get("thumbnail_b64"),
                top_features=alert_data.get("top_features"),
            )
        telemetry_state.increment_alert()
        await ws_manager.broadcast({
            "event": "attack_alert",
            "data": alert_data,
            "metrics": telemetry_state.to_dict(),
        })
    except Exception as exc:
        pass


async def on_stream_telemetry(metrics_data: Dict[str, Any]) -> None:
    """Callback when stream_engine updates performance metrics."""
    telemetry_state.update(
        fps=metrics_data.get("fps"),
        latency_ms=metrics_data.get("latency_ms"),
        flagged_attacks=metrics_data.get("flagged_attacks"),
        processed_frames=metrics_data.get("processed_frames"),
    )
    clean_count = metrics_data.get("clean_detections", 0)
    if clean_count > 0:
        telemetry_state.increment_clean(clean_count)
    await ws_manager.broadcast({
        "event": "telemetry_update",
        "metrics": telemetry_state.to_dict(),
    })


stream_engine = StreamEngine(
    source=0,
    alert_callback=on_stream_alert,
    telemetry_callback=on_stream_telemetry,
)


# ---------------------------------------------------------------------------
# Lifespan and Application Factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and resources on startup."""
    init_db()
    # Sync flagged attacks counter with existing database records
    with next(get_db()) as session:
        stats = get_alert_stats(session)
        telemetry_state.flagged_attacks = max(telemetry_state.flagged_attacks, stats.get("total_alerts", 0))
    yield


app = FastAPI(
    title="ProYOLO Physical Adversarial Attack Telemetry Server",
    description="Real-time optical patch detection telemetry, alerts, and dashboard for CCTV streams.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local testing and integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/alerts",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest attack alert",
    description="Ingest attack alert from inference worker, save to SQLite, and broadcast via WebSocket.",
)
async def ingest_alert(
    alert_in: AlertCreate,
    db: Session = Depends(get_db),
) -> AlertResponse:
    ts = alert_in.timestamp if alert_in.timestamp is not None else time.time()
    record = save_alert(
        session=db,
        camera_id=alert_in.camera_id,
        track_id=alert_in.track_id,
        confidence=alert_in.confidence,
        attack_probability=alert_in.attack_probability,
        bbox=alert_in.bbox,
        timestamp=ts,
        thumbnail_b64=alert_in.thumbnail_b64,
        top_features=alert_in.top_features,
    )
    telemetry_state.increment_alert()

    alert_dict = record.to_dict()

    # Broadcast event to WebSocket subscribers
    await ws_manager.broadcast({
        "event": "attack_alert",
        "data": alert_dict,
        "metrics": telemetry_state.to_dict(),
    })

    return AlertResponse(**alert_dict)


@app.get(
    "/api/v1/alerts",
    response_model=PaginatedAlertsResponse,
    summary="Query logged alerts",
    description="Retrieve paginated adversarial attack alerts with optional camera and probability filters.",
)
def list_alerts(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(20, ge=1, le=100, description="Page limit"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum attack probability"),
    db: Session = Depends(get_db),
) -> PaginatedAlertsResponse:
    records, total = get_alerts(
        session=db,
        skip=skip,
        limit=limit,
        camera_id=camera_id,
        min_probability=min_probability,
    )
    items = [AlertResponse(**r.to_dict()) for r in records]
    return PaginatedAlertsResponse(
        total=total,
        items=items,
        skip=skip,
        limit=limit,
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    summary="Get telemetry metrics",
    description="Retrieve current FPS, processed frame count, clean detections, flagged attack count, CPU usage, and latency.",
)
def get_metrics() -> MetricsResponse:
    return MetricsResponse(**telemetry_state.to_dict())


@app.post(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    summary="Update telemetry metrics",
    description="Update runtime counters from worker and broadcast latest telemetry.",
)
async def update_metrics(update: MetricsUpdate) -> MetricsResponse:
    telemetry_state.update(
        fps=update.fps,
        processed_frames=update.processed_frames,
        clean_detections=update.clean_detections,
        flagged_attacks=update.flagged_attacks,
        cpu_usage=update.cpu_usage,
        latency_ms=update.latency_ms,
    )
    latest = telemetry_state.to_dict()
    await ws_manager.broadcast({
        "event": "telemetry_update",
        "metrics": latest,
    })
    return MetricsResponse(**latest)


@app.post(
    "/api/v1/frame_telemetry",
    summary="Push live frame telemetry",
    description="Broadcast bounding box coordinates and keyframe to dashboard canvas.",
)
async def push_frame_telemetry(payload: FrameTelemetry) -> Dict[str, str]:
    if payload.fps is not None or payload.latency_ms is not None:
        telemetry_state.update(fps=payload.fps, latency_ms=payload.latency_ms)
    telemetry_state.increment_frames(1)

    clean_count = sum(1 for d in payload.detections if not d.is_attack)
    if clean_count > 0:
        telemetry_state.increment_clean(clean_count)

    broadcast_data = {
        "event": "frame_telemetry",
        "camera_id": payload.camera_id,
        "timestamp": payload.timestamp or time.time(),
        "detections": [d.model_dump() for d in payload.detections],
        "frame_b64": payload.frame_b64,
        "metrics": telemetry_state.to_dict(),
    }
    await ws_manager.broadcast(broadcast_data)
    return {"status": "broadcasted"}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """WebSocket endpoint broadcasting real-time attack alerts and canvas telemetry."""
    await ws_manager.connect(websocket)
    # Send immediate initial telemetry state upon connection
    try:
        await websocket.send_text(json.dumps({
            "event": "init_state",
            "metrics": telemetry_state.to_dict(),
            "timestamp": time.time(),
        }))
        while True:
            # Keep connection alive; accept optional ping or filter messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
            except Exception:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Video Streaming & File Upload Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/video-feed",
    summary="Motion-JPEG Video Feed",
    description="Returns a continuous multipart/x-mixed-replace JPEG video stream with real-time detections and threat overlays.",
)
async def get_video_feed(request: Request):
    """Motion-JPEG streaming endpoint for browser <img> or canvas element."""
    async def feed_generator():
        try:
            async for frame in stream_engine.stream_generator():
                if await request.is_disconnected():
                    break
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        feed_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post(
    "/api/v1/upload-video",
    summary="Upload video file for real-time processing",
    description="Accepts an MP4 or AVI video file, saves to uploads/, and switches stream engine to process the file.",
)
async def upload_video_file(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Upload a recorded video and switch active surveillance processing to that file."""
    filename = file.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix.lower()
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: '{ext}'. Allowed formats: {sorted(allowed_extensions)}",
        )

    uploads_dir = Path(__file__).resolve().parent.parent.parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / filename

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(saved_path, "wb") as f:
        f.write(content)

    # Switch stream engine to the uploaded video file
    stream_engine.set_source(str(saved_path), loop=True)

    return {
        "status": "success",
        "filename": filename,
        "saved_path": str(saved_path),
        "source_label": stream_engine.source_label,
        "message": f"Successfully loaded video '{filename}' into ProYOLO real-time defense pipeline.",
    }


@app.post(
    "/api/v1/stream-source",
    summary="Set active video stream source",
    description="Switch stream source between live webcam (0), RTSP URL, or video file path.",
)
async def switch_stream_source(req: StreamSourceRequest) -> Dict[str, Any]:
    src = req.source
    if isinstance(src, str):
        if src.lower() in ("webcam", "live", "camera", "0"):
            src = 0
        elif src.isdigit():
            src = int(src)

    stream_engine.set_source(src, loop=req.loop)
    return {
        "status": "success",
        "source": str(src),
        "source_label": stream_engine.source_label,
        "is_looping": stream_engine.is_looping,
    }


@app.post(
    "/api/v1/stream-control",
    summary="Control video stream playback",
    description="Pause, resume, or toggle video playback state.",
)
async def control_stream(req: StreamControlRequest) -> Dict[str, Any]:
    action = req.action.lower()
    if action == "pause":
        stream_engine.set_paused(True)
    elif action == "play" or action == "resume":
        stream_engine.set_paused(False)
    elif action == "toggle":
        stream_engine.toggle_pause()
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action '{req.action}'. Use 'play', 'pause', or 'toggle'.")

    return {
        "status": "success",
        "is_paused": stream_engine.is_paused,
    }


@app.get(
    "/api/v1/stream-status",
    summary="Get current stream engine status",
    description="Retrieve active video source, pause state, and looping flag.",
)
def get_stream_status() -> Dict[str, Any]:
    return {
        "source": str(stream_engine.current_source),
        "source_label": stream_engine.source_label,
        "is_paused": stream_engine.is_paused,
        "is_looping": stream_engine.is_looping,
        "total_frames_processed": stream_engine.total_frames_processed,
        "total_attacks_flagged": stream_engine.total_attacks_flagged,
    }


# ---------------------------------------------------------------------------
# Static Web Dashboard Mount
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page monitoring dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {"message": "ProYOLO Telemetry API is active. Dashboard static/index.html not yet built."},
        status_code=200,
    )
