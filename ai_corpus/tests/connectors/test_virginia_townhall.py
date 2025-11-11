from __future__ import annotations

from pathlib import Path

from ai_corpus.connectors.base import DocMeta
from ai_corpus.connectors.virginia_townhall import VirginiaTownhallConnector

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "virginia_townhall"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_discover_parses_forum_rows(monkeypatch):
    connector = VirginiaTownhallConnector(config={})

    def fake_fetch(path, *, absolute=False):
        if "Forums.cfm" in path:
            return load_fixture("forums.html")
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(connector, "_fetch_html", fake_fetch)
    collections = list(connector.discover())
    assert collections, "expected at least one forum entry"
    sample = collections[0]
    assert sample.collection_id
    assert sample.extra["board"]
    assert sample.extra["comment_count"] >= 0


def test_list_documents_reads_comment_rows(monkeypatch):
    connector = VirginiaTownhallConnector(config={})

    def fake_fetch(path, *, absolute=False):
        if "comments.cfm" in path:
            return load_fixture("comments_stage_10600.html")
        raise AssertionError("unexpected path")

    monkeypatch.setattr(connector, "_fetch_html", fake_fetch)
    docs = list(connector.list_documents("10600"))
    assert len(docs) == 5
    first = docs[0]
    assert first.doc_id == "237585"
    assert first.submitter
    assert first.submitted_at == "2025-11-08T14:22:00"


def test_fetch_writes_comment_files(tmp_path, monkeypatch):
    connector = VirginiaTownhallConnector(config={})
    detail_html = load_fixture("comment_237585.html")

    def fake_fetch(path, *, absolute=False):
        if absolute:
            return detail_html
        raise AssertionError("expected absolute URL fetch")

    monkeypatch.setattr(connector, "_fetch_html", fake_fetch)
    doc = DocMeta(
        source=connector.name,
        collection_id="10600",
        doc_id="237585",
        title="Sample comment",
        submitter="Allison",
        submitter_type=None,
        org=None,
        submitted_at="2025-11-08T14:22:00",
        language="en",
        urls={"html": "https://townhall.virginia.gov/L/viewcomments.cfm?commentid=237585"},
        extra={},
    )
    artifacts = connector.fetch(doc, tmp_path)
    assert "html" in artifacts and Path(artifacts["html"]).exists()
    assert "text" in artifacts and Path(artifacts["text"]).read_text(encoding="utf-8")
