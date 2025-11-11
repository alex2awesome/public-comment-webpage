from __future__ import annotations

import json
from pathlib import Path

from ai_corpus.connectors.connecticut_eregulations import ConnecticutEregsConnector
from ai_corpus.connectors.base import DocMeta

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "connecticut_eregulations"


def load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_discover_reads_open_comment_periods(monkeypatch):
    connector = ConnecticutEregsConnector(config={})
    monkeypatch.setattr(connector, "_fetch_open_comment_periods", lambda: load_json("open_comment_periods.json"))
    collections = list(connector.discover())
    assert collections
    first = collections[0]
    assert first.collection_id == "PR2023-021"
    assert "agency" in first.extra


def test_list_documents_converts_history_rows(monkeypatch):
    connector = ConnecticutEregsConnector(config={})
    monkeypatch.setattr(connector, "_fetch_history", lambda _: load_json("rmr_history_PR2023-021.json"))
    docs = list(connector.list_documents("PR2023-021"))
    assert docs
    doc_ids = {doc.doc_id for doc in docs}
    assert "{00CF599A-0000-C91E-9683-A70D4A543AE3}" in doc_ids


def test_fetch_downloads_binary(tmp_path, monkeypatch):
    connector = ConnecticutEregsConnector(config={})
    doc = DocMeta(
        source=connector.name,
        collection_id="PR2023-021",
        doc_id="example",
        title="Example Doc",
        submitter=None,
        submitter_type=None,
        org=None,
        submitted_at=None,
        language="en",
        urls={"download": "https://example.test/doc.pdf"},
        extra={"document_guid": "{abc}", "filename": "doc.pdf"},
    )

    class DummyResponse:
        status_code = 200
        content = b"pdf bytes"
        text = ""

    monkeypatch.setattr(
        "ai_corpus.connectors.connecticut_eregulations.backoff_get",
        lambda *args, **kwargs: DummyResponse(),
    )

    artifacts = connector.fetch(doc, tmp_path)
    saved = Path(artifacts["binary"])
    assert saved.exists()
    assert saved.read_bytes() == b"pdf bytes"
