import json
from pathlib import Path

import pytest

from ai_corpus.cli.main import main
from ai_corpus.connectors.base import Collection, DocMeta
from ai_corpus.connectors.factory import CONNECTOR_MAP


class StubConnector:
    name = "stub"

    def __init__(self, config=None, global_config=None, session=None):  # noqa: ANN001
        self.config = config or {}
        self.global_config = global_config or {}
        self._collection = Collection(
            source=self.name,
            collection_id="STUB-1",
            title="Stub Collection",
            url="http://example.com/stub",
            jurisdiction="Test-Jurisdiction",
            topic="AI",
            extra={},
        )
        self._doc = DocMeta(
            source=self.name,
            collection_id="STUB-1",
            doc_id="STUB-DOC-1",
            title="Stub Doc",
            submitter="Stub Submitter",
            submitter_type="ngo",
            org="Stub Org",
            submitted_at="2024-01-01",
            language="en",
            urls={"html": "http://example.com/stub/doc"},
            extra={},
        )
        self._call_doc = DocMeta(
            source=self.name,
            collection_id="STUB-1",
            doc_id="STUB-CALL-1",
            title="Stub Call",
            submitter="Stub Agency",
            submitter_type="agency",
            org="Stub Agency",
            submitted_at="2023-12-01",
            language="en",
            urls={"html": "http://example.com/stub/call"},
            extra={"document_role": "call"},
            kind="call",
        )

    def discover(self, **_):
        yield self._collection

    def list_documents(self, collection_id, **_):  # noqa: ANN001
        if collection_id == "STUB-1":
            yield self._doc

    def get_call_document(self, collection_id, **_):  # noqa: ANN001
        if collection_id == "STUB-1":
            return self._call_doc
        return None

    def fetch(self, doc, out_dir: Path, **kwargs):  # noqa: ANN001
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{doc.doc_id}.html"
        html_path.write_text("<html><body>Stub Content</body></html>", encoding="utf-8")
        return {"doc_id": doc.doc_id, "html": str(html_path)}


@pytest.fixture(autouse=True)
def register_stub_connector():
    original = CONNECTOR_MAP.get("stub")
    CONNECTOR_MAP["stub"] = StubConnector
    try:
        yield
    finally:
        if original is not None:
            CONNECTOR_MAP["stub"] = original
        else:
            CONNECTOR_MAP.pop("stub", None)


def _stub_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "global:\n"
        "  user_agent: stub-agent\n"
        "stub:\n"
        "  type: test\n",
        encoding="utf-8",
    )
    return cfg


def test_cli_discover(capsys, tmp_path: Path):
    config_path = _stub_config(tmp_path)
    exit_code = main(["--config", str(config_path), "discover", "--connector", "stub"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert "stub" in data
    assert data["stub"][0]["collection_id"] == "STUB-1"


def test_cli_crawl_download_extract_export(tmp_path: Path, capsys):
    config_path = _stub_config(tmp_path)
    meta_path = tmp_path / "meta.jsonl"
    downloads_dir = tmp_path / "downloads"
    blob_dir = tmp_path / "blobs"
    download_db = tmp_path / "downloads.sqlite"
    export_db_url = f"sqlite:///{tmp_path/'cli_export.db'}"

    # crawl (collect call + responses)
    assert (
        main(
            [
                "--config",
                str(config_path),
                "crawl",
                "--connector",
                "stub",
                "--collection-id",
                "STUB-1",
                "--output",
                str(meta_path),
                "--target",
                "all",
            ]
        )
        == 0
    )
    assert meta_path.exists()

    # download call
    assert (
        main(
            [
                "--config",
                str(config_path),
                "download-call",
                "--connector",
                "stub",
                "--collection-id",
                "STUB-1",
                "--meta-file",
                str(meta_path),
                "--out-dir",
                str(downloads_dir),
                "--database",
                str(download_db),
            ]
        )
        == 0
    )

    # download responses
    assert (
        main(
            [
                "--config",
                str(config_path),
                "download-responses",
                "--connector",
                "stub",
                "--collection-id",
                "STUB-1",
                "--meta-file",
                str(meta_path),
                "--out-dir",
                str(downloads_dir),
                "--database",
                str(download_db),
                "--max-workers",
                "1",
            ]
        )
        == 0
    )

    # extract
    assert (
        main(
            [
                "--config",
                str(config_path),
                "extract",
                "--database",
                str(download_db),
                "--blob-dir",
                str(blob_dir),
                "--max-workers",
                "1",
            ]
        )
        == 0
    )

    # export
    assert (
        main(
            [
                "--config",
                str(config_path),
                "export",
                "--meta-file",
                str(meta_path),
                "--database",
                str(download_db),
                "--database-url",
                export_db_url,
            ]
        )
        == 0
    )

    # Ensure normalized document stored
    from ai_corpus.storage.db import Database, Document  # noqa: WPS433

    db = Database(export_db_url)
    with db.engine.connect() as conn:
        rows = conn.execute(Document.__table__.select()).fetchall()
        assert len(rows) == 1
        assert rows[0].collection_id == "STUB-1"
