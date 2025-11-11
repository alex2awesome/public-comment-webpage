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

    assert docs, "Expected at least one PDF in the docket listing"
    bundles = [doc for doc in docs if doc.extra.get("is_bundle")]
    government_docs = [doc for doc in docs if doc.kind == "call"]
    assert bundles, "Should expose aggregated comment bundles"
    assert government_docs, "Should expose government-authored PDFs"
    for doc in bundles + government_docs:
        assert doc.urls["pdf"].endswith(".pdf")
    for doc in government_docs:
        assert doc.submitter == "CPPA"
        assert doc.extra.get("document_role") == "call"


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
    class _DummyClient:
        def __init__(self) -> None:
            self.sync = object()

    monkeypatch.setattr(
        "ai_corpus.connectors.cppa_admt.init_backend",
        lambda backend_name: _DummyClient(),
    )

    connector = CppaAdmtConnector(
        config={"index_url": "https://cppa.ca.gov/regulations/ccpa_updates.html"},
        global_config={"user_agent": "pytest"},
    )
    collection_id = list(connector.discover())[0].collection_id
    docs = list(connector.list_documents(collection_id=collection_id))
    bundles = [d for d in docs if d.extra.get("is_bundle")]
    assert len(bundles) == 2
    bundle = bundles[0]
    result = connector.fetch(bundle, tmp_path)
    assert "letters" in result
    assert len(result["letters"]) == 2
    assert result["letters"][0]["submitter_name"].startswith("ACME")
