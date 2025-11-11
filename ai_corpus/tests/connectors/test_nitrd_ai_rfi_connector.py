from pathlib import Path

import pytest

from ai_corpus.connectors.nitrd_ai_rfi import NitrdAiRfiConnector


class FakeResponse:
    def __init__(self, text: str, content: bytes | None = None):
        self.status_code = 200
        self._text = text
        self.content = content or b"test"

    @property
    def text(self) -> str:  # type: ignore[misc]
        return self._text


@pytest.fixture
def index_text() -> str:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "nitrd_index.html"
    return fixture.read_text()


def test_nitrd(monkeypatch, tmp_path, index_text):
    def fake_get(url, **kwargs):  # noqa: ANN001
        if url.endswith('.pdf') or url.endswith('.docx'):
            return FakeResponse(index_text, content=b'%PDF')
        return FakeResponse(index_text)

    monkeypatch.setattr("ai_corpus.connectors.nitrd_ai_rfi.backoff_get", fake_get)
    connector = NitrdAiRfiConnector(
        config={"base_url": "https://files.nitrd.gov/90-fr-9088/"},
        global_config={"user_agent": "pytest"},
    )
    collections = list(connector.discover())
    assert collections[0].collection_id == "90-FR-9088"
    docs = list(connector.list_documents(collection_id="90-FR-9088"))
    doc_ids = {doc.doc_id for doc in docs}
    assert doc_ids == {"MITRE-AI-RFI-2025.pdf", "CAIDP-AI-RFI-2025.pdf", "OpenAI-AI-RFI-2025.docx"}
    for doc in docs:
        result = connector.fetch(doc, tmp_path)
        if 'pdf' in result:
            assert Path(result['pdf']).exists()

