from pathlib import Path

import pytest

from ai_corpus.connectors.cppa_admt import CppaAdmtConnector, PagePairJudgment


class FakeResponse:
    def __init__(self, text: str | None = None, content: bytes | None = None):
        self.status_code = 200
        self._text = text
        self.content = content or b"PDF"

    @property
    def text(self) -> str:  # type: ignore[misc]
        return self._text or ""


@pytest.fixture
def index_html() -> str:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "cppa_index.html"
    return fixture.read_text()


@pytest.fixture
def bundle_text() -> str:
    first_letter = (
        "ACME CORPORATION\n\nJanuary 5, 2024\n\n"
        + "We appreciate the opportunity to comment. " * 20
    )
    second_letter = (
        "\n\nBETA ASSOCIATION\n\nFebruary 1, 2024\n\n"
        + "We share similar concerns. " * 20
    )
    return first_letter + second_letter


@pytest.fixture
def cppa_fixture() -> str:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "cppa_updates.html"
    return fixture.read_text()


def test_cppa_list_all_links(monkeypatch, cppa_fixture):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(cppa_fixture, 'html.parser')
    assert soup.select('a[href$=".pdf"]'), "Fixture should contain PDF links"
    def fake_get(url, **kwargs):  # noqa: ANN001
        return FakeResponse(text=cppa_fixture)

    monkeypatch.setattr("ai_corpus.connectors.cppa_admt.backoff_get", fake_get)
    connector = CppaAdmtConnector(
        config={"index_url": "https://cppa.ca.gov/regulations/ccpa_updates.html", "base_url": "https://cppa.ca.gov"},
        global_config={"user_agent": "pytest"},
    )
    collections = list(connector.discover())
    assert collections[0].collection_id == "PR-02-2023"
    docs = list(connector.list_documents(collection_id="PR-02-2023"))

    assert docs, "Expected at least one aggregated comments PDF"
    for doc in docs:
        assert doc.extra.get("kind") == "aggregated_comments"
        assert doc.urls["pdf"].endswith(".pdf")


def test_cppa_bundle(monkeypatch, tmp_path, index_html, bundle_text):
    def fake_get(url, **kwargs):  # noqa: ANN001
        if url.endswith('.pdf'):
            return FakeResponse(content=b"%PDF")
        return FakeResponse(text=index_html)

    monkeypatch.setattr("ai_corpus.connectors.cppa_admt.backoff_get", fake_get)
    monkeypatch.setattr(
        "ai_corpus.connectors.cppa_admt.extract_pages",
        lambda path, parser="auto": (
            [
                "ACME CORPORATION\n\nJanuary 5, 2024\n\n" + "We appreciate the opportunity to comment. " * 20,
                "BETA ASSOCIATION\n\nFebruary 1, 2024\n\n" + "We share similar concerns. " * 20,
            ],
            {"total_pages": 2},
        ),
    )

    monkeypatch.setattr(
        "ai_corpus.connectors.cppa_admt.request_structured_response",
        lambda client, model, prompt_text, schema, max_output_tokens=96, temperature=None: PagePairJudgment(
            label="0", a_is_error=False, b_is_error=False
        ),
    )

    connector = CppaAdmtConnector(
        config={"index_url": "https://cppa.ca.gov/regulations/ccpa_updates.html"},
        global_config={"user_agent": "pytest"},
    )
    docs = list(connector.list_documents(collection_id="PR-02-2023"))
    assert len(docs) == 2
    bundle = next(d for d in docs if d.extra.get("is_bundle"))
    result = connector.fetch(bundle, tmp_path)
    assert "letters" in result
    assert len(result["letters"]) == 2
    assert result["letters"][0]["submitter_name"].startswith("ACME")
