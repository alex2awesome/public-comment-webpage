"""
Database persistence for normalized document metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from sqlalchemy import JSON, Boolean, Column, Integer, LargeBinary, PrimaryKeyConstraint, String, create_engine
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON  # type: ignore
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ai_corpus.pipelines.normalize import NormalizedDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'ai_corpus.db').as_posix()}"


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    submitter_name: Mapped[str | None] = mapped_column(String(256))
    submitter_type: Mapped[str | None] = mapped_column(String(64))
    org: Mapped[str | None] = mapped_column(String(256))
    submitted_at: Mapped[str | None] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(16))
    url_html: Mapped[str | None] = mapped_column(String(1024))
    url_pdf: Mapped[str | None] = mapped_column(String(1024))
    url_json: Mapped[str | None] = mapped_column(String(1024))
    sha256_pdf: Mapped[str | None] = mapped_column(String(64))
    sha256_text: Mapped[str | None] = mapped_column(String(64))
    bytes_pdf: Mapped[int | None] = mapped_column(Integer)
    bytes_text: Mapped[int | None] = mapped_column(Integer)
    text_path: Mapped[str | None] = mapped_column(String(1024))
    pdf_path: Mapped[str | None] = mapped_column(String(1024))
    raw_meta: Mapped[dict] = mapped_column(JSON().with_variant(SQLITE_JSON(), "sqlite"), default=dict)  # type: ignore[arg-type]


class DownloadRecord(Base):
    __tablename__ = "downloads"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    connector: Mapped[str] = mapped_column(String(64), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON().with_variant(SQLITE_JSON(), "sqlite"), default=dict)  # type: ignore[arg-type]
    downloaded_at: Mapped[str | None] = mapped_column(String(32))
    extracted: Mapped[bool] = mapped_column(Boolean, default=False)
    text_path: Mapped[str | None] = mapped_column(String(1024))
    sha256_text: Mapped[str | None] = mapped_column(String(64))


class CollectionHarvest(Base):
    __tablename__ = "collection_harvests"
    __table_args__ = (
        PrimaryKeyConstraint("connector", "collection_id", "artifact_type"),
    )

    connector: Mapped[str] = mapped_column(String(64), nullable=False)
    collection_id: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    output_path: Mapped[str | None] = mapped_column(String(1024))
    extra: Mapped[dict] = mapped_column(JSON().with_variant(SQLITE_JSON(), "sqlite"), default=dict)  # type: ignore[arg-type]
    last_run_at: Mapped[str | None] = mapped_column(String(32))


class Database:
    """
    Lightweight wrapper around SQLAlchemy for persisting normalized metadata.
    """

    def __init__(self, url: str = DEFAULT_DB_URL) -> None:
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def upsert_documents(self, docs: Iterable[NormalizedDocument]) -> None:
        """
        Insert or update documents by UID.
        """
        with Session(self.engine) as session:
            for doc in docs:
                payload = asdict(doc)
                existing = session.query(Document).filter_by(uid=doc.uid).one_or_none()
                if existing:
                    for key, value in payload.items():
                        setattr(existing, key, value)
                else:
                    session.add(Document(**payload))
            session.commit()

    # Download helpers --------------------------------------------------

    def download_exists(self, doc_id: str) -> bool:
        with Session(self.engine) as session:
            return session.get(DownloadRecord, doc_id) is not None

    def get_download(self, doc_id: str) -> Optional[Dict[str, object]]:
        with Session(self.engine) as session:
            record = session.get(DownloadRecord, doc_id)
            if not record:
                return None
            return {
                "doc_id": record.doc_id,
                "connector": record.connector,
                "collection_id": record.collection_id,
                "payload": record.payload,
                "downloaded_at": record.downloaded_at,
                "extracted": record.extracted,
                "text_path": record.text_path,
                "sha256_text": record.sha256_text,
            }

    def record_download(
        self,
        *,
        connector_name: str,
        collection_id: str,
        doc_id: str,
        payload: Dict[str, object],
    ) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with Session(self.engine) as session:
            record = session.get(DownloadRecord, doc_id)
            if record:
                record.connector = connector_name
                record.collection_id = collection_id
                record.payload = payload
                record.downloaded_at = timestamp
            else:
                session.add(
                    DownloadRecord(
                        doc_id=doc_id,
                        connector=connector_name,
                        collection_id=collection_id,
                        payload=payload,
                        downloaded_at=timestamp,
                    )
                )
            session.commit()

    def iter_downloads(self, *, extracted: Optional[bool] = None) -> Iterable[Dict[str, object]]:
        with Session(self.engine) as session:
            query = session.query(DownloadRecord)
            if extracted is not None:
                query = query.filter(DownloadRecord.extracted.is_(extracted))
            records = query.all()
        for record in records:
            yield {
                "doc_id": record.doc_id,
                "connector": record.connector,
                "collection_id": record.collection_id,
                "payload": record.payload,
                "downloaded_at": record.downloaded_at,
                "extracted": record.extracted,
                "text_path": record.text_path,
                "sha256_text": record.sha256_text,
            }

    def mark_extracted(self, doc_id: str, *, text_path: Optional[str], sha256_text: Optional[str]) -> None:
        with Session(self.engine) as session:
            record = session.get(DownloadRecord, doc_id)
            if not record:
                return
            record.extracted = True
            record.text_path = text_path
            record.sha256_text = sha256_text
            session.commit()

    # Harvest cache helpers ---------------------------------------------

    def get_harvest(
        self,
        connector: str,
        collection_id: str,
        artifact_type: str,
    ) -> Optional[Dict[str, object]]:
        with Session(self.engine) as session:
            record = (
                session.query(CollectionHarvest)
                .filter_by(
                    connector=connector,
                    collection_id=collection_id,
                    artifact_type=artifact_type,
                )
                .one_or_none()
            )
            if not record:
                return None
            return {
                "connector": record.connector,
                "collection_id": record.collection_id,
                "artifact_type": record.artifact_type,
                "item_count": record.item_count,
                "output_path": record.output_path,
                "metadata": record.extra,
                "last_run_at": record.last_run_at,
            }

    def record_harvest(
        self,
        *,
        connector: str,
        collection_id: str,
        artifact_type: str,
        item_count: int,
        output_path: Optional[str],
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with Session(self.engine) as session:
            record = (
                session.query(CollectionHarvest)
                .filter_by(
                    connector=connector,
                    collection_id=collection_id,
                    artifact_type=artifact_type,
                )
                .one_or_none()
            )
            if record:
                record.item_count = item_count
                record.output_path = output_path
                if metadata is not None:
                    record.extra = metadata
                record.last_run_at = timestamp
            else:
                session.add(
                    CollectionHarvest(
                        connector=connector,
                        collection_id=collection_id,
                        artifact_type=artifact_type,
                        item_count=item_count,
                        output_path=output_path,
                        extra=metadata or {},
                        last_run_at=timestamp,
                    )
                )
            session.commit()

    def list_harvests(self, *, artifact_type: Optional[str] = None) -> Iterable[Dict[str, object]]:
        with Session(self.engine) as session:
            query = session.query(CollectionHarvest)
            if artifact_type:
                query = query.filter(CollectionHarvest.artifact_type == artifact_type)
            records = query.all()
        for record in records:
            yield {
                "connector": record.connector,
                "collection_id": record.collection_id,
                "artifact_type": record.artifact_type,
                "item_count": record.item_count,
                "output_path": record.output_path,
                "metadata": record.extra,
                "last_run_at": record.last_run_at,
            }
