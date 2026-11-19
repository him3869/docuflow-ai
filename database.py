import os
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/docuflow_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # RBAC & governance metadata
    department_access: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    sensitivity_level: Mapped[str] = mapped_column(String(32), default="internal", nullable=False)

    # Vector embedding (adjust dimension to match your embedding model output)
    embedding: Mapped[List[float]] = mapped_column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., INGEST, QUERY
    actor_roles: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)




def init_db():
    """Enable pgvector extension and create tables."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


