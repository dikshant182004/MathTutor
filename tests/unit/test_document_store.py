from pathlib import Path
from backend.v2.document_store import PersistentDocumentStore

def test_document_isolation_and_retrieval(tmp_path: Path):
    store = PersistentDocumentStore(str(tmp_path / "docs.json"))
    store.ingest("a", "doc1", "integration by parts formula and examples", "notes.pdf", 2)
    store.ingest("b", "doc2", "probability distribution examples", "other.pdf", 1)
    a = store.retrieve("a", "integration formula")
    assert a and a[0]["document_id"] == "doc1"
    assert store.retrieve("a", "probability") == []
    assert a[0]["metadata"]["page"] == 2
