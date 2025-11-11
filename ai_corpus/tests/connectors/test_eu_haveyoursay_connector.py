from __future__ import annotations

from pathlib import Path

from ai_corpus.connectors.base import Collection, DocMeta
from ai_corpus.connectors.eu_haveyoursay_keyword import (
    EuHaveYourSayKeywordConnector,
    InitiativeContext,
)


class _FakePage:
    def __init__(self, html: str):
        self._html = html
        self.last_url: str | None = None

    def goto(self, url: str, **_kwargs) -> None:
        self.last_url = url

    def wait_for_load_state(self, *_args, **_kwargs) -> None:
        return None

    def wait_for_timeout(self, *_args, **_kwargs) -> None:
        return None

    def content(self) -> str:
        return self._html

    def locator(self, *_args, **_kwargs):  # noqa: ANN001
        return _FakeLocator()


class _FakeLocator:
    def count(self) -> int:
        return 0


class _FakeBrowser:
    def __init__(self, page: _FakePage):
        self._page = page
        self.closed = False

    def new_page(self) -> _FakePage:
        return self._page

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser):
        self._browser = browser

    def launch(self, **_kwargs) -> _FakeBrowser:
        return self._browser


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser):
        self.chromium = _FakeChromium(browser)


class _FakeManager:
    def __init__(self, page_html: str):
        self._page = _FakePage(page_html)
        self._browser = _FakeBrowser(self._page)

    def __enter__(self) -> _FakePlaywright:
        return _FakePlaywright(self._browser)

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401, ANN001
        self._browser.close()
        return False


def test_eu_keyword_connector_io(monkeypatch, tmp_path: Path) -> None:
    connector = EuHaveYourSayKeywordConnector(
        config={"keywords": ["Artificial intelligence act adoption"]},
        global_config={"user_agent": "pytest"},
    )

    collection_id = "info_law_better-regulation_have-your-say_initiatives_12427-Artificial-intelligence-act-adoption-of-the-commission-proposal"
    collection = Collection(
        source=connector.name,
        collection_id=collection_id,
        title="Artificial intelligence act adoption",
        url="https://example.com/initiative",
        jurisdiction="EU",
        topic="AI",
        extra={},
    )
    doc = DocMeta(
        source=connector.name,
        collection_id=collection_id,
        doc_id="F1234_en",
        title="Sample feedback",
        submitter="Example Submitter",
        submitter_type="Business",
        org="Example Org",
        submitted_at="2024-01-01",
        language="en",
        urls={"html": "https://example.com/feedback/F1234_en"},
        extra={
            "attachment_present": False,
            "keyword": "Artificial intelligence act adoption",
            "metadata_table": {"Feedback reference": "F1234_en"},
        },
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake_html = "<html><body><p>Feedback content</p></body></html>"
    cached_html_path = cache_dir / "F1234_en.html"
    cached_html_path.write_text(fake_html, encoding="utf-8")
    doc.extra["cached_html_path"] = str(cached_html_path)
    doc.extra["expected_attachment_name"] = "F1234_en.pdf"

    context = InitiativeContext(
        keyword="Artificial intelligence act adoption",
        matched_text="Artificial intelligence act adoption",
        initiative_url=collection.url,
        initiative_path_fragment="info/law/better-regulation/have-your-say/initiatives_12427-Artificial-intelligence-act-adoption-of-the-commission-proposal",
        feedback_links=["https://example.com/initiative/all-feedback"],
        collection=collection,
    )
    connector._keyword_cache[context.keyword] = context
    connector._collection_cache[collection_id] = context

    def fake_harvest(self, ctx):  # noqa: D401, ANN001
        assert ctx.collection.collection_id == collection_id
        return [doc]

    monkeypatch.setattr(
        EuHaveYourSayKeywordConnector,
        "_harvest_documents_for_context",
        fake_harvest,
    )

    discovered = list(connector.discover())
    assert discovered and discovered[0].collection_id == collection_id

    docs = list(connector.list_documents(collection_id))
    assert docs and docs[0].doc_id == "F1234_en"

    fetch_result = connector.fetch(doc, tmp_path)
    html_path = Path(fetch_result["html"])
    assert html_path.exists()
    assert html_path.read_text(encoding="utf-8").strip() == fake_html
    assert "attachments" not in fetch_result
