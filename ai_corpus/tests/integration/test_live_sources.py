from pathlib import Path

import json
import pytest

from ai_corpus.connectors.nist_airmf import NistAirmfConnector
from ai_corpus.connectors.nitrd_ai_rfi import NitrdAiRfiConnector
from ai_corpus.connectors.regulations_gov import RegulationsGovConnector


LIVE_USER_AGENT = "ai-policy-corpus-live-tests/1.0 (+contact: you@example.org)"


@pytest.mark.live
def test_regulations_gov_live(tmp_path: Path):
    connector = RegulationsGovConnector(
        config={
            "base_url": "https://api.regulations.gov",
            "version": "v4",
            "auth": {
                "value": "QeHJqzCTU5j8leqEarXoaXtEBmhYWqWdSoU4EVXe",
            },
        },
        global_config={"user_agent": LIVE_USER_AGENT},
    )
    docs = list(connector.list_documents(collection_id="NTIA-2023-0005"))
    expected_ids = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures" / "regulations_ntia_2023_0005_doc_ids.json").read_text()
    )
    assert len(docs) == len(expected_ids)
    assert {doc.doc_id for doc in docs} == set(expected_ids)
    sample = docs[0]
    fetch_result = connector.fetch(sample, tmp_path)
    assert fetch_result.get("attachments") or fetch_result.get("html") or fetch_result.get("pdf")


@pytest.mark.live
def test_nist_airmf_live(tmp_path: Path):
    connector = NistAirmfConnector(
        config={
            "index_url": "https://www.nist.gov/itl/ai-risk-management-framework/comments-2nd-draft-ai-risk-management-framework",
            "base_url": "https://www.nist.gov",
        },
        global_config={"user_agent": LIVE_USER_AGENT},
    )
    docs = list(connector.list_documents(collection_id="AI-RMF-2ND-DRAFT-2022"))
    from bs4 import BeautifulSoup

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "nist_ai_rmf_comments.html"
    soup = BeautifulSoup(fixture_path.read_text(), "html.parser")
    expected_urls = set()
    for anchor in soup.select('a[data-file-url], a[href$=".pdf"]'):
        file_url = anchor.get("data-file-url") or anchor.get("href")
        if not file_url or not file_url.lower().endswith(".pdf"):
            continue
        if file_url.startswith("http"):
            expected_urls.add(file_url)
        else:
            expected_urls.add(f"https://www.nist.gov{file_url}")
    urls = {doc.urls["pdf"] for doc in docs}
    assert len(docs) == len(expected_urls)
    assert expected_urls.issubset(urls)
    sample = docs[0]
    fetch_result = connector.fetch(sample, tmp_path)
    assert Path(fetch_result["pdf"]).exists()


@pytest.mark.live
def test_nitrd_ai_rfi_live(tmp_path: Path):
    connector = NitrdAiRfiConnector(
        config={
            "base_url": "https://files.nitrd.gov/90-fr-9088/",
            "seed_files": ["CAIDP-AI-RFI-2025.pdf", "MITRE-AI-RFI-2025.pdf"],
        },
        global_config={"user_agent": LIVE_USER_AGENT},
    )
    docs = list(connector.list_documents(collection_id="90-FR-9088"))
    doc_ids = {doc.doc_id for doc in docs}
    for seed in ["CAIDP-AI-RFI-2025.pdf", "MITRE-AI-RFI-2025.pdf"]:
        assert seed in doc_ids
    sample = next(doc for doc in docs if doc.doc_id.endswith(".pdf"))
    fetch_result = connector.fetch(sample, tmp_path)
    assert Path(fetch_result["pdf"]).exists()



