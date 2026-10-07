from pathlib import Path
from backend.v2.document_store import PersistentDocumentStore

def test_document_isolation_and_retrieval(tmp_path: Path):
    store=PersistentDocumentStore(str(tmp_path/"docs.json"))
    store.ingest("a","doc1","integration by parts and examples","notes.pdf",2)
    store.ingest("b","doc2","probability distribution examples","other.pdf",1)
    hit=store.retrieve("a","integration")
    assert hit and hit[0]["document_id"]=="doc1"
    assert store.retrieve("a","probability")==[]
    assert hit[0]["metadata"]["page"]==2

def test_document_pages_append(tmp_path: Path):
    store=PersistentDocumentStore(str(tmp_path/"docs.json"))
    store.ingest("a","doc","first page","notes.pdf",1)
    store.ingest("a","doc","second page","notes.pdf",2)
    assert len(store.retrieve("a","page"))==2
