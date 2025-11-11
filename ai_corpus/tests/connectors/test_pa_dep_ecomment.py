from __future__ import annotations

from pathlib import Path

from ai_corpus.connectors.base import DocMeta
from ai_corpus.connectors.pa_dep_ecomment import CommentRow, PaDepEcommentConnector

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "pennsylvania_ecomment"


def test_discover_parses_listing(monkeypatch):
    connector = PaDepEcommentConnector(config={})
    html = (FIXTURES / "home.html").read_text(encoding="utf-8")
    monkeypatch.setattr(connector, "_get_page", lambda _: html)
    collections = list(connector.discover())
    assert collections
    assert collections[0].collection_id


def test_list_documents_builds_docmeta(monkeypatch):
    connector = PaDepEcommentConnector(config={})
    rows = [
        CommentRow(
            index=1,
            first_name="Clifford",
            last_name="Lau",
            affiliation="BCMAC",
            city="Coraopolis",
            state="PA",
            country="US",
            method="Online",
            received="11/03/2025",
            button_name="ctl00$ContentPlaceHolder1$CommentGrid$ctl02$ViewButton",
            attachments=["https://example.test/file.pdf"],
        )
    ]
    monkeypatch.setattr(connector, "_fetch_comment_rows", lambda enc: rows)
    docs = list(connector.list_documents("token"))
    assert len(docs) == 1
    assert docs[0].extra["attachments"]


def test_fetch_writes_text(tmp_path, monkeypatch):
    connector = PaDepEcommentConnector(config={})
    doc = DocMeta(
        source=connector.name,
        collection_id="token",
        doc_id="token-1",
        title="Comment",
        submitter="Clifford Lau",
        submitter_type=None,
        org=None,
        submitted_at=None,
        language="en",
        urls={},
        extra={
            "enc": "token",
            "button_name": "ctl00$ContentPlaceHolder1$CommentGrid$ctl02$ViewButton",
            "attachments": [],
        },
    )

    class DummySession:
        def get(self, *_, **__):
            raise AssertionError("No attachments requested")

    monkeypatch.setattr(
        connector,
        "_fetch_comment_text",
        lambda enc, button: ("Sample text", DummySession()),
    )

    artifacts = connector.fetch(doc, tmp_path)
    text_path = Path(artifacts["text"])
    assert text_path.read_text(encoding="utf-8") == "Sample text"
