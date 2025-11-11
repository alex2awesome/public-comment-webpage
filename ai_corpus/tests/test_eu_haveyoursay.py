
from ai_corpus.connectors.eu_haveyoursay_keyword import EuHaveYourSayKeywordConnector
from ai_corpus.config.loader import load_config
import pytest

@pytest.fixture
def config():
    return load_config()

def test_eu_haveyoursay_list_documents(config):
    connector = EuHaveYourSayKeywordConnector(config['eu_have_your_say_keyword'], config['global'])
    connector.headless = True
    collections = list(connector.discover())
    assert len(collections) > 0

    for collection in collections:
        documents = list(connector.list_documents(collection.collection_id, max_pages=1))
        assert len(documents) > 0
        for doc in documents:
            assert doc.doc_id is not None
            assert doc.urls['html'] is not None
