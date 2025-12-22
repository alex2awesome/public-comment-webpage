import json
from pathlib import Path

import pytest

from ai_corpus.connectors.regulations_gov import RegulationsGovConnector


class FakeResponse:
    def __init__(self, status_code=200, *, text=None, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._text = text
        self._json = json_data
        self.content = content
        self.headers = headers or {}

    def json(self):
        if self._json is not None:
            return self._json
        raise ValueError("No JSON payload")

    @property
    def text(self):  # type: ignore[misc]
        if self._text is not None:
            return self._text
        if self._json is not None:
            return json.dumps(self._json)
        return ""


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def regs_payload(fixture_path: Path) -> dict:
    return json.loads((fixture_path / "regulations_comments_page1.json").read_text())


@pytest.fixture
def regs_multi_pages(fixture_path: Path) -> tuple[dict, dict]:
    page1 = json.loads((fixture_path / "regulations_ntia_2023_0005_page1.json").read_text())
    page2 = json.loads((fixture_path / "regulations_ntia_2023_0005_page2.json").read_text())
    return page1, page2


@pytest.fixture
def regs_dockets_page(fixture_path: Path) -> dict:
    return json.loads((fixture_path / "regulations_dockets_2023_page1.json").read_text())


def test_list_and_fetch(monkeypatch, tmp_path, regs_payload):
    calls = []

    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        calls.append(path)
        if path == "documents":
            return FakeResponse(
                json_data={
                    "data": [
                        {
                            "id": "DOC-0001",
                            "attributes": {
                                "title": "Notice of Request for Information",
                                "postedDate": "2024-01-01T00:00:00Z",
                                "documentType": "Notice",
                                "documentSubtype": "Request for Information",
                                "fileFormats": [],
                                "agencyId": "NTIA",
                            },
                            "relationships": {},
                        }
                    ],
                    "meta": {"page": {"hasNextPage": False, "totalPages": 1}},
                }
            )
        if path == "documents/DOC-0001":
            return FakeResponse(
                json_data={
                    "data": {
                        "attributes": {
                            "fileFormats": [
                                {
                                    "format": "PDF",
                                    "fileUrl": "https://downloads.regulations.gov/DOC-0001/content.pdf",
                                }
                            ]
                        }
                    },
                    "included": [],
                }
            )
        return FakeResponse(json_data=regs_payload)

    def fake_download(url, **kwargs):  # noqa: ANN001
        if url.endswith("attachment_1.pdf"):
            return FakeResponse(content=b"%PDF-1.4 test pdf")
        if url.endswith("content.pdf"):
            return FakeResponse(content=b"%PDF-1.4 content pdf")
        return FakeResponse(status_code=404)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.backoff_get", fake_download)

    connector = RegulationsGovConnector(
        config={"corpora_seeds": [{"docket_id": "NTIA-2023-0005"}]},
        global_config={"user_agent": "pytest-agent"},
    )

    collections = list(connector.discover())
    assert collections and collections[0].collection_id == "NTIA-2023-0005"

    docs = list(connector.list_documents(collection_id="NTIA-2023-0005"))
    assert len(docs) == 2
    comment = next(doc for doc in docs if doc.kind == "response")
    assert comment.doc_id == "NTIA-2023-0005-1000"
    assert comment.submitter == "OpenAI"
    government_docs = [doc for doc in docs if doc.kind == "call"]
    assert government_docs and government_docs[0].doc_id == "DOC-0001"
    assert government_docs[0].extra.get("government_document") is True
    result = connector.fetch(comment, tmp_path)
    assert "html" in result
    assert Path(result["html"]).exists()
    assert "attachments" in result and len(result["attachments"]) == 1
    assert Path(result["attachments"][0]).suffix == ".pdf"
    assert calls.count("comments") == 1
    assert calls.count("documents") == 1


def test_list_documents_paginates_all_pages(monkeypatch, regs_multi_pages):
    page1, page2 = regs_multi_pages

    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        params = kwargs.get("params", {})
        page_number = params.get("page[number]", 1)
        if page_number == 1:
            return FakeResponse(json_data=page1)
        if page_number == 2:
            return FakeResponse(json_data=page2)
        return FakeResponse(json_data={"data": [], "meta": {"hasNextPage": False, "totalPages": 2}})

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)
    monkeypatch.setattr(
        "ai_corpus.connectors.regulations_gov.backoff_get",
        lambda *args, **kw: FakeResponse(content=b"%PDF"),
    )

    connector = RegulationsGovConnector(
        config={"corpora_seeds": [{"docket_id": "NTIA-2023-0005"}]},
        global_config={"user_agent": "pytest-agent"},
    )

    collections = list(connector.discover())
    assert collections and collections[0].collection_id == "NTIA-2023-0005"

    docs = list(connector.list_documents(collection_id="NTIA-2023-0005", document_type="comments"))
    expected = len(page1.get("data", [])) + len(page2.get("data", []))
    assert len(docs) == expected
    assert len({doc.doc_id for doc in docs}) == expected


def test_discover_date_filtered(monkeypatch, regs_dockets_page):
    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        assert path == "dockets"
        params = kwargs.get("params", {})
        assert params.get("filter[lastModifiedDate][ge]") == "2023-01-01 00:00:00"
        assert params.get("filter[lastModifiedDate][le]") == "2023-12-31 23:59:59"
        payload = json.loads(json.dumps(regs_dockets_page))
        for idx, entry in enumerate(payload.get("data", []), start=1):
            attrs = entry.setdefault("attributes", {})
            attrs["numberOfDocuments"] = idx
            attrs["commentCount"] = idx * 10
        payload.setdefault("meta", {}).update({"hasNextPage": False, "totalPages": 1})
        return FakeResponse(json_data=payload)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest-agent"},
    )

    collections = list(
        connector.discover(start_date="2023-01-01", end_date="2023-12-31", page_size=25)
    )
    assert len(collections) == len(regs_dockets_page.get("data", []))
    first_attrs = regs_dockets_page["data"][0]["attributes"]
    assert collections[0].collection_id == regs_dockets_page["data"][0]["id"]
    assert collections[0].extra.get("last_modified") == first_attrs.get("lastModifiedDate")
    assert collections[0].extra.get("document_count") == 1
    assert collections[0].extra.get("comment_count") == 10


def test_discover_applies_min_update_filter(monkeypatch, regs_dockets_page):
    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        payload = json.loads(json.dumps(regs_dockets_page))
        for idx, entry in enumerate(payload.get("data", [])):
            entry.setdefault("attributes", {})["numberOfDocuments"] = idx
        payload.setdefault("meta", {}).update({"hasNextPage": False, "totalPages": 1})
        return FakeResponse(json_data=payload)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest-agent", "regulations_min_updates": 3},
    )

    collections = list(
        connector.discover(start_date="2023-01-01", end_date="2023-12-31", page_size=25)
    )
    assert collections, "Expected some dockets to meet the min-update threshold"
    for coll in collections:
        assert coll.extra.get("document_count") is not None
        assert coll.extra["document_count"] >= 3


def test_discover_applies_min_comment_filter(monkeypatch, regs_dockets_page):
    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        payload = json.loads(json.dumps(regs_dockets_page))
        for idx, entry in enumerate(payload.get("data", [])):
            entry.setdefault("attributes", {})["commentCount"] = idx
        payload.setdefault("meta", {}).update({"hasNextPage": False, "totalPages": 1})
        return FakeResponse(json_data=payload)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest-agent", "regulations_min_comments": 5},
    )

    collections = list(
        connector.discover(start_date="2023-01-01", end_date="2023-12-31", page_size=25)
    )
    assert collections, "Expected some dockets to meet the min-comment threshold"
    for coll in collections:
        assert coll.extra.get("comment_count") is not None
        assert coll.extra["comment_count"] >= 5


def test_discover_drops_null_comment_counts(monkeypatch, regs_dockets_page):
    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        payload = json.loads(json.dumps(regs_dockets_page))
        payload.setdefault("meta", {}).update({"hasNextPage": False, "totalPages": 1})
        return FakeResponse(json_data=payload)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest-agent", "regulations_drop_null_comments": True},
    )

    collections = list(
        connector.discover(start_date="2023-01-01", end_date="2023-12-31", page_size=25)
    )
    assert collections == []


def test_discover_drops_null_comment_counts_when_requested(monkeypatch, regs_dockets_page):
    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        payload = json.loads(json.dumps(regs_dockets_page))
        # Leave commentCount missing for all entries.
        payload.setdefault("meta", {}).update({"hasNextPage": False, "totalPages": 1})
        return FakeResponse(json_data=payload)

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(
        config={},
        global_config={"user_agent": "pytest-agent", "regulations_drop_null_comments": True},
    )

    collections = list(
        connector.discover(start_date="2023-01-01", end_date="2023-12-31", page_size=25)
    )
    assert collections == []


def test_get_call_document_prioritises_notices(monkeypatch):
    documents_payload = {
        "data": [
            {
                "id": "DOC-1",
                "attributes": {
                    "title": "Supporting Material",
                    "postedDate": "2024-01-02T00:00:00Z",
                    "documentType": "Supporting & Related Material",
                    "documentSubtype": "FAQ",
                    "fileFormats": [],
                },
                "relationships": {},
            },
            {
                "id": "DOC-2",
                "attributes": {
                    "title": "Notice of Request for Information",
                    "postedDate": "2024-01-01T00:00:00Z",
                    "documentType": "Notice",
                    "documentSubtype": "Request for Information",
                    "fileFormats": [],
                },
                "relationships": {},
            },
        ],
        "meta": {"page": {"hasNextPage": False, "totalPages": 1}},
    }

    def fake_regs_call(path, **kwargs):  # noqa: ANN001
        if path == "documents":
            return FakeResponse(json_data=documents_payload)
        if path == "documents/DOC-2":
            return FakeResponse(
                json_data={
                    "data": {
                        "attributes": {
                            "fileFormats": [
                                {
                                    "format": "PDF",
                                    "fileUrl": "https://downloads.regulations.gov/DOC-2/content.pdf",
                                }
                            ]
                        }
                    },
                    "included": [],
                }
            )
        if path == "documents/DOC-1":
            return FakeResponse(json_data={"data": {"attributes": {}}, "included": []})
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr("ai_corpus.connectors.regulations_gov.regs_backoff_get", fake_regs_call)

    connector = RegulationsGovConnector(config={}, global_config={"user_agent": "pytest-agent"})

    doc = connector.get_call_document("NTIA-2023-0005")
    assert doc is not None
    assert doc.doc_id == "DOC-2"
    assert doc.kind == "call"
    assert doc.urls.get("pdf") == "https://downloads.regulations.gov/DOC-2/content.pdf"


def test_iter_rule_versions_builds_rows(monkeypatch):
    fr_doc = {
        "document_number": "2024-12345",
        "title": "Request for Information on Testing",
        "type": "NOTICE",
        "publication_date": "2024-01-15",
        "comments_close_on": "2024-02-15",
        "html_url": "https://federalregister.gov/documents/2024-12345",
        "regulations_dot_gov_url": "https://www.regulations.gov/document/NTIA-2023-0005-0001",
        "regulations_dot_gov_info": {"document_id": "NTIA-2023-0005-0001", "object_id": "0900006488abcdef"},
        "docket_id": "NTIA-2023-0005",
        "docket_ids": ["NTIA-2023-0005"],
        "abstract": "Summary text",
        "action": "Notice of request for information.",
        "dates": "Comments must be received by February 15, 2024.",
        "agency_names": ["NTIA"],
        "full_text_xml_url": "https://example.com/doc.xml",
    }

    def fake_fetch(self, docket_id, include_types):  # noqa: ANN001
        assert docket_id == "NTIA-2023-0005"
        return [fr_doc]

    def fake_xml(self, url):  # noqa: ANN001
        assert url == "https://example.com/doc.xml"
        return ({"xml": True}, "Response to Comment 5 is provided in section II.")

    monkeypatch.setenv("REGS_GOV_API_KEY", "dummy")
    monkeypatch.setattr(RegulationsGovConnector, "_fr_fetch_docket_documents", fake_fetch)
    monkeypatch.setattr(RegulationsGovConnector, "_fr_extract_supplementary", fake_xml)

    connector = RegulationsGovConnector(config={}, global_config={"user_agent": "pytest-agent"})
    rows = list(connector.iter_rule_versions("NTIA-2023-0005"))
    assert len(rows) == 1
    row = rows[0]
    assert row["fr_document_number"] == "2024-12345"
    assert row["mentions_comment_response"] is True
    assert row["comment_citation_snippet"].startswith("Response to Comment")
