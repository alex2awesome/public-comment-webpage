from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ai_corpus.connectors.nist_airmf import NistAirmfConnector


class FakeResponse:
    def __init__(self, text: str):
        self.status_code = 200
        self._text = text

    @property
    def text(self) -> str:  # type: ignore[misc]
        return self._text


@pytest.fixture
def nist_fixture() -> str:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "nist_ai_rmf_comments.html"
    return fixture.read_text()


def test_nist_list_complete(monkeypatch, nist_fixture):
    soup = BeautifulSoup(nist_fixture, "html.parser")
    expected_urls = set()
    for anchor in soup.select('a[data-file-url], a[href$=".pdf"]'):
        file_url = anchor.get("data-file-url") or anchor.get("href")
        if not file_url or not file_url.lower().endswith(".pdf"):
            continue
        if file_url.startswith("http"):
            expected_urls.add(file_url)
        else:
            expected_urls.add(f"https://www.nist.gov{file_url}")

    def fake_get(url, **kwargs):  # noqa: ANN001
        return FakeResponse(nist_fixture)

    monkeypatch.setattr("ai_corpus.connectors.nist_airmf.backoff_get", fake_get)
    connector = NistAirmfConnector(
        config={"index_url": "https://example.com/index", "base_url": "https://www.nist.gov"},
        global_config={"user_agent": "pytest"},
    )
    collections = list(connector.discover())
    assert collections[0].collection_id == "AI-RMF-2ND-DRAFT-2022"
    docs = list(connector.list_documents(collection_id="AI-RMF-2ND-DRAFT-2022"))
    urls = {doc.urls["pdf"] for doc in docs}
    assert len(docs) == len(expected_urls)
    assert expected_urls.issubset(urls)
