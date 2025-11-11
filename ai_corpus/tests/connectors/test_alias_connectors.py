import json
from pathlib import Path

import pytest

from ai_corpus.connectors.regulations_gov import RegulationsGovConnector


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code=200):
        self.status_code = status_code
        self._json = json_data
        self.content = content

    def json(self):
        if self._json is not None:
            return self._json
        return {}

    @property
    def text(self):  # type: ignore[misc]
        if self._json is not None:
            import json as _json

            return _json.dumps(self._json)
        return ""


@pytest.fixture
def regs_payload(tmp_path: Path) -> dict:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "regulations_comments_page1.json"
    return json.loads(fixture.read_text())


def _monkeypatch_regulations(monkeypatch, payload):
    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")

    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        if path == "dockets":
            params = kwargs.get("params", {})
            docket_id = params.get("filter[docketId]") or "NTIA-2023-0005"
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": docket_id,
                            "attributes": {"title": docket_id, "lastModifiedDate": "2023-05-01"},
                        }
                    ],
                    "meta": {"hasNextPage": False, "totalPages": 1},
                }
            )
        return FakeResponse(payload)

    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    def fake_download(url, **kwargs):  # noqa: ANN001
        if url.endswith(".pdf") or "attachment" in url:
            return FakeResponse({}, content=b"%PDF-1.4")
        return FakeResponse(payload)

    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.backoff_get", fake_download)


def test_ntia_open_models_docket(monkeypatch, regs_payload, tmp_path: Path):
    _monkeypatch_regulations(monkeypatch, regs_payload)
    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest"},
    )
    collections = list(connector.discover(docket_ids=["NTIA-2023-0009"]))
    assert len(collections) == 1
    assert collections[0].collection_id == "NTIA-2023-0009"

    docs = list(connector.list_documents(collection_id="NTIA-2023-0009"))
    assert docs and docs[0].source == connector.name
    result = connector.fetch(docs[0], tmp_path)
    assert "doc_id" in result


def test_omb_ai_memo_docket(monkeypatch, regs_payload, tmp_path: Path):
    _monkeypatch_regulations(monkeypatch, regs_payload)
    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest"},
    )
    collections = list(connector.discover(docket_ids=["OMB-2023-0020"]))
    assert collections[0].collection_id == "OMB-2023-0020"

    docs = list(connector.list_documents(collection_id="OMB-2023-0020"))
    assert docs and docs[0].source == connector.name
    result = connector.fetch(docs[0], tmp_path)
    assert "doc_id" in result
