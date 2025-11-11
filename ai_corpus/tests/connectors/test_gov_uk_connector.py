import json
from pathlib import Path

import pytest

from ai_corpus.connectors.base import DocMeta
from ai_corpus.connectors.gov_uk import GovUkConnector


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def search_payload(fixture_path: Path) -> dict:
    return json.loads((fixture_path / "govuk_search.json").read_text())


@pytest.fixture
def content_payload(fixture_path: Path) -> dict:
    return json.loads((fixture_path / "govuk_content.json").read_text())


@pytest.fixture
def consultation_html(fixture_path: Path) -> str:
    return (fixture_path / "govuk_consultation_page.html").read_text()


def test_discover_uses_search_api(monkeypatch, search_payload):
    calls = []

    def fake_get_json(self, url, params=None):  # noqa: ANN001
        calls.append((url, params))
        return search_payload

    monkeypatch.setattr(GovUkConnector, "_get_json", fake_get_json)
    connector = GovUkConnector(config={}, global_config={"user_agent": "pytest-agent"})

    collections = list(connector.discover(max_results=10))

    assert len(collections) == len(search_payload["results"])
    first = collections[0]
    assert first.collection_id == "/government/consultations/ai-governance-consultation"
    assert first.topic == "AI"
    assert calls  # ensure API was invoked
    _, params = calls[0]
    assert params["filter_format"] == "consultation"
    assert params["order"] == "-public_timestamp"


def test_list_documents_combines_json_and_html(monkeypatch, content_payload, consultation_html):
    def fake_get_json(self, url, params=None):  # noqa: ANN001
        assert "/api/content/" in url
        return content_payload

    def fake_get_html(self, url):  # noqa: ANN001
        assert url.startswith("https://www.gov.uk/")
        return consultation_html

    monkeypatch.setattr(GovUkConnector, "_get_json", fake_get_json)
    monkeypatch.setattr(GovUkConnector, "_get_html", fake_get_html)

    connector = GovUkConnector(config={}, global_config={"user_agent": "pytest-agent"})
    docs = list(connector.list_documents("/government/consultations/ai-governance-consultation"))

    assert docs
    assert docs[0].doc_id.startswith("page:")
    attachment_docs = docs[1:]
    expected_urls = {
        "https://assets.publishing.service.gov.uk/media/consultation-document.pdf",
        "https://assets.publishing.service.gov.uk/media/impact-assessment.pdf",
        "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/0001/summary.pdf",
        "https://example.gov.uk/data.csv",
    }
    extracted_urls = {doc.urls["file"] for doc in attachment_docs}
    assert extracted_urls == expected_urls


class FakeResponse:
    def __init__(self, *, status_code=200, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self._text = text

    @property
    def text(self):  # type: ignore[misc]
        return self._text


def test_fetch_downloads_html_and_file(tmp_path, monkeypatch):
    connector = GovUkConnector(config={}, global_config={"user_agent": "pytest-agent"})

    monkeypatch.setattr(connector, "_get_html", lambda url: "<html>consultation</html>")

    def fake_backoff_get(url, **kwargs):  # noqa: ANN001
        assert url.endswith(".pdf")
        return FakeResponse(content=b"%PDF-1.4 test pdf", status_code=200)

    monkeypatch.setattr("ai_corpus.connectors.gov_uk.backoff_get", fake_backoff_get)

    doc = DocMeta(
        source="gov_uk",
        collection_id="/government/consultations/ai-governance-consultation",
        doc_id="attachment:/government/consultations/ai-governance-consultation:consultation-document.pdf",
        title="Consultation document",
        submitter="Department for Science, Innovation and Technology",
        submitter_type="government",
        org="Department for Science, Innovation and Technology",
        submitted_at="2024-04-02T08:15:00Z",
        language="en",
        urls={
            "html": "https://www.gov.uk/government/consultations/ai-governance-consultation",
            "file": "https://assets.publishing.service.gov.uk/media/consultation-document.pdf",
        },
    )

    artifacts = connector.fetch(doc, tmp_path)
    assert "html" in artifacts
    html_path = Path(artifacts["html"])
    assert html_path.exists()
    assert html_path.read_text() == "<html>consultation</html>"

    assert "file" in artifacts
    file_path = Path(artifacts["file"])
    assert file_path.exists()
    assert file_path.read_bytes() == b"%PDF-1.4 test pdf"
