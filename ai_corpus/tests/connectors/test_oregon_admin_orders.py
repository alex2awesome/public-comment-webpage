from __future__ import annotations

import json
from pathlib import Path

from ai_corpus.connectors.oregon_admin_orders import OregonAdminOrdersConnector
from ai_corpus.connectors.base import DocMeta

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "oregon_admin_orders" / "sample_orders.json"


def load_rows():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_discover_returns_dataset_collection():
    connector = OregonAdminOrdersConnector(config={})
    collections = list(connector.discover())
    assert len(collections) == 1
    assert collections[0].collection_id == "oregon_admin_orders"


def test_list_documents_converts_rows(monkeypatch):
    connector = OregonAdminOrdersConnector(config={})
    monkeypatch.setattr(connector, "_fetch_rows", load_rows)
    docs = list(connector.list_documents("oregon_admin_orders"))
    assert docs
    assert docs[0].collection_id == "oregon_admin_orders"


def test_fetch_writes_html(tmp_path, monkeypatch):
    connector = OregonAdminOrdersConnector(config={})
    doc = DocMeta(
        source=connector.name,
        collection_id="oregon_admin_orders",
        doc_id="ACCAA 1-2006",
        title="sample",
        submitter=None,
        submitter_type=None,
        org=None,
        submitted_at=None,
        language="en",
        urls={"html": "https://example.test/doc"},
        extra={},
    )

    class DummyResponse:
        status_code = 200
        text = "<html>doc</html>"
        encoding = "utf-8"

    monkeypatch.setattr(
        "ai_corpus.connectors.oregon_admin_orders.backoff_get",
        lambda *args, **kwargs: DummyResponse(),
    )

    artifacts = connector.fetch(doc, tmp_path)
    saved = Path(artifacts["html"])
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "<html>doc</html>"
