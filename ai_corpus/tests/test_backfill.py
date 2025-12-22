import json
from pathlib import Path

from ai_corpus.cli.main import _run_backfill_stage
from ai_corpus.storage.db import Database


def test_backfill_processes_existing_pdfs(monkeypatch, tmp_path):
    db_path = tmp_path / "downloads.sqlite"
    blob_dir = tmp_path / "blobs"
    db = Database(f"sqlite:///{db_path}")
    db.record_download(
        connector_name="regulations_gov",
        collection_id="TEST-DOCKET",
        doc_id="TEST-001",
        payload={},
    )

    def fake_extract(artifacts):
        assert artifacts.get("pdf") == str(target_pdf)
        return "parsed text"

    monkeypatch.setattr("ai_corpus.cli.main.extract_text_from_artifacts", fake_extract)

    workspace_root = tmp_path / "workspace"
    raw_dir = workspace_root / "regulations_gov" / "TEST-DOCKET" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = raw_dir / "TEST-001.pdf"
    target_pdf.write_bytes(b"PDF DATA")
    _run_backfill_stage(db_path, blob_dir, workspace_root, "raw")

    records = list(db.iter_downloads())
    assert len(records) == 1
    record = records[0]
    assert record["text_path"]
    assert Path(record["text_path"]).exists()


def test_backfill_downloads_missing_pdfs(monkeypatch, tmp_path):
    db_path = tmp_path / "downloads.sqlite"
    blob_dir = tmp_path / "blobs"
    db = Database(f"sqlite:///{db_path}")
    db.record_download(
        connector_name="regulations_gov",
        collection_id="TEST-DOCKET",
        doc_id="TEST-002",
        payload={},
    )

    workspace_root = tmp_path / "workspace"
    meta_dir = workspace_root / "regulations_gov" / "TEST-DOCKET"
    raw_dir = meta_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "TEST-DOCKET.meta.jsonl"
    meta_path.write_text(
        json.dumps({"doc_id": "TEST-002", "urls": {"pdf": "https://example.com/TEST-002.pdf"}}) + "\n",
        encoding="utf-8",
    )

    def fake_download(url, dest_path, *, force=False):
        dest_path.write_bytes(b"PDF DATA")
        return True

    def fake_extract(artifacts):
        return "downloaded text"

    monkeypatch.setattr("ai_corpus.cli.main._download_pdf_from_url", fake_download)
    monkeypatch.setattr("ai_corpus.cli.main.extract_text_from_artifacts", fake_extract)

    _run_backfill_stage(db_path, blob_dir, workspace_root, "raw")

    record = next(iter(db.iter_downloads()))
    assert record["text_path"]
    assert Path(record["text_path"]).read_text() == "downloaded text"
