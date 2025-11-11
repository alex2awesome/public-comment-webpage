from __future__ import annotations

import json
from pathlib import Path

from ai_corpus.connectors.base import DocMeta
from ai_corpus.connectors.utah_bulletin import BulletinPost, UtahBulletinConnector

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "utah_bulletin" / "posts.json"


def load_posts():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    posts = []
    for item in data:
        posts.append(
            BulletinPost(
                post_id=item["id"],
                title=item["title"]["rendered"],
                link=item["link"],
                pdf_url="https://rules.utah.gov/wp-content/uploads/example.pdf",
                published_at=item["date"],
            )
        )
    return posts


def test_discover_returns_posts(monkeypatch):
    connector = UtahBulletinConnector(config={"per_page": 2})
    monkeypatch.setattr(connector, "_fetch_posts", load_posts)
    collections = list(connector.discover())
    assert collections
    assert collections[0].extra["pdf_url"]


def test_list_documents_creates_docmeta(monkeypatch):
    connector = UtahBulletinConnector(config={"per_page": 2})
    posts = load_posts()
    monkeypatch.setattr(connector, "_fetch_posts", lambda: posts)
    docs = list(connector.list_documents(str(posts[0].post_id)))
    assert len(docs) == 1
    assert docs[0].urls["pdf"].endswith(".pdf")


def test_fetch_saves_pdf(tmp_path, monkeypatch):
    connector = UtahBulletinConnector(config={})
    doc = DocMeta(
        source=connector.name,
        collection_id="123",
        doc_id="utah_bulletin_123",
        title="Sample",
        submitter="Utah Office of Administrative Rules",
        submitter_type="agency",
        org="Utah Office of Administrative Rules",
        submitted_at=None,
        language="en",
        urls={"pdf": "https://rules.utah.gov/wp-content/uploads/example.pdf"},
        extra={},
    )

    class DummyResponse:
        status_code = 200
        content = b"%PDF-1.4"
        text = ""

    monkeypatch.setattr(
        "ai_corpus.connectors.utah_bulletin.backoff_get",
        lambda *args, **kwargs: DummyResponse(),
    )

    artifacts = connector.fetch(doc, tmp_path)
    saved = Path(artifacts["pdf"])
    assert saved.exists()
    assert saved.read_bytes().startswith(b"%PDF")
