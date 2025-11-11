from __future__ import annotations

from pathlib import Path

from ai_corpus.connectors.base import DocMeta
from ai_corpus.connectors.washington_register import WashingtonRegisterConnector

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "washington_register"


def test_discover_reads_issue_links(monkeypatch):
    connector = WashingtonRegisterConnector(config={"max_issues": 2})
    index_html = (FIXTURES / "issues.html").read_text(encoding="utf-8")
    monkeypatch.setattr(connector, "_get_html", lambda url: index_html)
    issues = list(connector.discover())
    assert issues
    assert issues[0].collection_id


def test_list_documents_parses_issue(monkeypatch):
    connector = WashingtonRegisterConnector(config={})
    issue_html = (FIXTURES / "issue_25-01.htm").read_text(encoding="utf-8")
    monkeypatch.setattr(connector, "_get_html", lambda url: issue_html)
    docs = list(connector.list_documents("25-01"))
    assert docs
    assert docs[0].doc_id


def test_fetch_downloads_pdf(tmp_path, monkeypatch):
    connector = WashingtonRegisterConnector(config={})
    doc = DocMeta(
        source=connector.name,
        collection_id="25-01",
        doc_id="25-01-031",
        title="Sample",
        submitter="Agency",
        submitter_type="agency",
        org="Agency",
        submitted_at=None,
        language="en",
        urls={"pdf": "https://example.test/file.pdf"},
        extra={},
    )

    monkeypatch.setattr(
        connector,
        "_download_file",
        lambda url: b"PDF",
    )

    artifacts = connector.fetch(doc, tmp_path)
    saved = Path(artifacts["pdf"])
    assert saved.read_bytes() == b"PDF"
