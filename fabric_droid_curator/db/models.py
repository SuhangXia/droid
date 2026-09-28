from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True, default="")
    raw_path: Mapped[str] = mapped_column(Text, unique=True)
    raw_signature: Mapped[str] = mapped_column(String(128), default="")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    start_ns: Mapped[int] = mapped_column(Integer, default=0)
    end_ns: Mapped[int] = mapped_column(Integer, default=0)
    robot_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    d435_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    wrist_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    gelsight_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    ati_sample_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    sensor_availability_json: Mapped[str] = mapped_column(Text, default="{}")
    camera_session_id: Mapped[str] = mapped_column(String(128), default="unknown", index=True)
    review_status: Mapped[str] = mapped_column(String(32), default="unreviewed", index=True)
    action_branch: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    swatch_uid: Mapped[str] = mapped_column(String(128), default="unknown", index=True)
    split: Mapped[str] = mapped_column(String(32), default="unassigned", index=True)
    start_pose_bucket: Mapped[str] = mapped_column(String(128), default="unknown")
    current_annotation_version: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    streams: Mapped[list["SensorStream"]] = relationship(back_populates="episode", cascade="all, delete-orphan")


class SensorStream(Base):
    __tablename__ = "sensor_streams"
    __table_args__ = (UniqueConstraint("episode_pk", "name", name="uq_stream_episode_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    source_path: Mapped[str] = mapped_column(Text)
    timestamp_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    start_ns: Mapped[int] = mapped_column(Integer, default=0)
    end_ns: Mapped[int] = mapped_column(Integer, default=0)
    measured_hz: Mapped[float] = mapped_column(Float, default=0.0)
    encoded_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    monotonic: Mapped[bool] = mapped_column(Boolean, default=False)
    decodable: Mapped[bool] = mapped_column(Boolean, default=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}")

    episode: Mapped[Episode] = relationship(back_populates="streams")


class EventProposal(Base):
    __tablename__ = "event_proposals"
    __table_args__ = (UniqueConstraint("episode_pk", "event_name", "detector_version", name="uq_event_detector"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    event_name: Mapped[str] = mapped_column(String(64), index=True)
    candidate_timestamp_ns: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_json: Mapped[str] = mapped_column(Text, default="{}")
    detector_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("episode_pk", "version", name="uq_annotation_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str] = mapped_column(Text, default="")
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    segment_id: Mapped[str] = mapped_column(String(192), unique=True, index=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    annotation_version: Mapped[int] = mapped_column(Integer)
    segment_type: Mapped[str] = mapped_column(String(64), index=True)
    start_ns: Mapped[int] = mapped_column(Integer)
    end_ns: Mapped[int] = mapped_column(Integer)
    prompt_template_id: Mapped[str] = mapped_column(String(128))
    prompt: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    annotation_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), index=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[int] = mapped_column(primary_key=True)
    export_version: Mapped[str] = mapped_column(String(128), unique=True)
    output_path: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    leakage_audit_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CameraSession(Base):
    __tablename__ = "camera_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    reference_episode_id: Mapped[str] = mapped_column(String(128))
    reference_frame_path: Mapped[str] = mapped_column(Text)
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    image_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_translation: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_rotation: Mapped[float | None] = mapped_column(Float, nullable=True)
    homography_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    consistency_class: Mapped[str] = mapped_column(String(32), default="unknown")
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class QCResult(Base):
    __tablename__ = "qc_results"
    __table_args__ = (UniqueConstraint("episode_pk", "qc_version", name="uq_qc_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    qc_version: Mapped[str] = mapped_column(String(128))
    severity: Mapped[str] = mapped_column(String(16), index=True)
    scores_json: Mapped[str] = mapped_column(Text)
    hard_failures_json: Mapped[str] = mapped_column(Text)
    warnings_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_pk: Mapped[int | None] = mapped_column(ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(192))
    action: Mapped[str] = mapped_column(String(64))
    old_value_json: Mapped[str] = mapped_column(Text, default="null")
    new_value_json: Mapped[str] = mapped_column(Text, default="null")
    reviewer: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str] = mapped_column(Text, default="")
    annotation_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
