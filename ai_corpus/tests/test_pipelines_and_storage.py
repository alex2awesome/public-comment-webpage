from pathlib import Path

from ai_corpus.connectors.base import Collection, DocMeta
from ai_corpus.pipelines.discover import discover_collections
from ai_corpus.pipelines.download import download_documents
from ai_corpus.pipelines.extract import extract_text_from_artifacts
from ai_corpus.pipelines.harvest import harvest_documents
from ai_corpus.pipelines.normalize import normalize_doc
from ai_corpus.storage.db import Database, Document
from ai_corpus.storage.fs import BlobStore


class DummyConnector:
    name = "dummy"

    def __init__(self) -> None:
        self._collections = [
            Collection(
                source=self.name,
                collection_id="COLL-1",
                title="Dummy Collection",
                url="http://example.com",
                jurisdiction="Test",
                topic="AI",
                extra={},
            )
        ]
        self._docs = [
            DocMeta(
                source=self.name,
                collection_id="COLL-1",
                doc_id="DOC-1",
                title="Doc 1",
                submitter="Tester",
                submitter_type="individual",
                org=None,
                submitted_at="2024-01-01",
                language="en",
                urls={"html": "http://example.com/doc1"},
                extra={},
            )
        ]

    def discover(self, **_):
        yield from self._collections

    def list_documents(self, collection_id, **_):  # noqa: ANN001
        if collection_id == "COLL-1":
            yield from self._docs

    def fetch(self, doc, out_dir: Path, **_):  # noqa: ANN001
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{doc.doc_id}.html"
        html_path.write_text("<html><body>Hello World</body></html>", encoding="utf-8")
        return {"doc_id": doc.doc_id, "html": str(html_path)}


def test_pipelines_and_storage(tmp_path: Path):
    connector = DummyConnector()

    collections = discover_collections(connector)
    assert len(collections) == 1 and collections[0].collection_id == "COLL-1"

    docs = harvest_documents(connector, collection_id="COLL-1")
    assert len(docs) == 1 and docs[0].doc_id == "DOC-1"

    artifacts = download_documents(connector, docs, tmp_path / "downloads")
    assert artifacts and artifacts[0]["doc_id"] == "DOC-1"

    text = extract_text_from_artifacts(artifacts[0])
    assert "Hello World" in text

    blob_store = BlobStore(tmp_path / "blobs")
    blob_path = blob_store.store_bytes(text.encode("utf-8"), suffix="txt")
    assert blob_path.exists()

    normalized = normalize_doc(docs[0])
    normalized.text_path = str(blob_path)
    normalized.sha256_text = blob_path.name

    db = Database(f"sqlite:///{tmp_path/'test.db'}")
    db.upsert_documents([normalized])

    # Verify persistence
    db_check = Database(f"sqlite:///{tmp_path/'test.db'}")
    with db_check.engine.connect() as conn:
        rows = conn.execute(Document.__table__.select()).fetchall()
        assert len(rows) == 1
        assert rows[0].uid == normalized.uid
