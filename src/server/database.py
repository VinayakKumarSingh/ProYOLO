"""Database management and SQLite ORM models for ProYOLO alert telemetry.

Uses SQLAlchemy to persist attack alerts, track coordinates, and keyframe snapshots.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = os.environ.get("PROYOLO_DB_URL", "sqlite:///surveillance_alerts.db")


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy ORM models."""
    pass


class AlertRecord(Base):
    """SQLAlchemy ORM model for storing adversarial attack alerts."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(Float, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    attack_probability = Column(Float, nullable=False, index=True)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    thumbnail_b64 = Column(Text, nullable=True)
    top_features_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record to a clean dictionary representation."""
        top_features = {}
        if self.top_features_json:
            try:
                top_features = json.loads(self.top_features_json)
            except Exception:
                top_features = {}

        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "confidence": round(float(self.confidence), 4),
            "attack_probability": round(float(self.attack_probability), 4),
            "bbox": [
                round(float(self.bbox_x1), 2),
                round(float(self.bbox_y1), 2),
                round(float(self.bbox_x2), 2),
                round(float(self.bbox_y2), 2),
            ],
            "thumbnail_b64": self.thumbnail_b64,
            "top_features": top_features,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Global engine and sessionmaker singletons
_ENGINES: Dict[str, Engine] = {}
_SESSION_FACTORIES: Dict[str, sessionmaker[Session]] = {}


from sqlalchemy.pool import StaticPool

def get_engine(db_url: Optional[str] = None) -> Engine:
    """Retrieve or create an SQLAlchemy engine for the specified database URL."""
    url = db_url or os.environ.get("PROYOLO_DB_URL", DEFAULT_DB_URL)
    if url not in _ENGINES:
        connect_args = {}
        kwargs = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        engine = create_engine(url, connect_args=connect_args, echo=False, **kwargs)
        Base.metadata.create_all(bind=engine)
        _ENGINES[url] = engine
    return _ENGINES[url]


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker[Session]:
    """Retrieve or create a sessionmaker factory."""
    url = db_url or os.environ.get("PROYOLO_DB_URL", DEFAULT_DB_URL)
    if url not in _SESSION_FACTORIES:
        engine = get_engine(url)
        factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        _SESSION_FACTORIES[url] = factory
    return _SESSION_FACTORIES[url]


def init_db(db_url: Optional[str] = None) -> None:
    """Create all tables in the database if they do not already exist."""
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)


def get_db(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    factory = get_session_factory(db_url)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def save_alert(
    session: Session,
    camera_id: str,
    track_id: int,
    confidence: float,
    attack_probability: float,
    bbox: List[float],
    timestamp: Optional[float] = None,
    thumbnail_b64: Optional[str] = None,
    top_features: Optional[Dict[str, float]] = None,
) -> AlertRecord:
    """Persist a new adversarial alert into the SQLite database.

    Args:
        session: Active SQLAlchemy session.
        camera_id: Identifier of the source camera (e.g. 'cam-01').
        track_id: Tracking integer ID of the target person.
        confidence: YOLO detection confidence score (0.0 to 1.0).
        attack_probability: Classifier adversarial attack probability (0.0 to 1.0).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        timestamp: Unix epoch timestamp in seconds.
        thumbnail_b64: Optional base64 JPEG keyframe thumbnail.
        top_features: Optional dictionary of feature attributions.

    Returns:
        The committed AlertRecord instance.
    """
    if timestamp is None:
        timestamp = time.time()

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    top_features_json = json.dumps(top_features) if top_features else None

    record = AlertRecord(
        camera_id=camera_id,
        timestamp=float(timestamp),
        track_id=int(track_id),
        confidence=float(confidence),
        attack_probability=float(attack_probability),
        bbox_x1=float(x1),
        bbox_y1=float(y1),
        bbox_x2=float(x2),
        bbox_y2=float(y2),
        thumbnail_b64=thumbnail_b64,
        top_features_json=top_features_json,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_alerts(
    session: Session,
    skip: int = 0,
    limit: int = 20,
    camera_id: Optional[str] = None,
    min_probability: Optional[float] = None,
) -> Tuple[List[AlertRecord], int]:
    """Query logged alerts with pagination and optional filtering.

    Args:
        session: Active SQLAlchemy session.
        skip: Number of records to skip (offset).
        limit: Maximum number of records to return.
        camera_id: Optional camera ID filter.
        min_probability: Optional minimum attack probability filter.

    Returns:
        Tuple of (list of AlertRecord, total count matching query).
    """
    stmt = select(AlertRecord)
    count_stmt = select(func.count(AlertRecord.id))

    if camera_id:
        stmt = stmt.where(AlertRecord.camera_id == camera_id)
        count_stmt = count_stmt.where(AlertRecord.camera_id == camera_id)

    if min_probability is not None:
        stmt = stmt.where(AlertRecord.attack_probability >= min_probability)
        count_stmt = count_stmt.where(AlertRecord.attack_probability >= min_probability)

    total = session.scalar(count_stmt) or 0
    records = session.scalars(
        stmt.order_by(desc(AlertRecord.timestamp), desc(AlertRecord.id))
        .offset(skip)
        .limit(limit)
    ).all()

    return list(records), total


def get_alert_stats(session: Session) -> Dict[str, Any]:
    """Compute aggregate alert statistics from the database."""
    total_alerts = session.scalar(select(func.count(AlertRecord.id))) or 0
    avg_probability = session.scalar(select(func.avg(AlertRecord.attack_probability))) or 0.0
    return {
        "total_alerts": total_alerts,
        "avg_attack_probability": round(float(avg_probability), 4),
    }
