from __future__ import annotations
import json, os, re, time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
try:
    import redis
except Exception:
    redis=None

@dataclass
class DocumentChunk:
    chunk_id:str; document_id:str; student_id:str; text:str; source:str; page:int|None; created_at:float

class PersistentDocumentStore:
    def __init__(self,path=".data/v2_documents.json"):
        self.redis_url=os.getenv("REDIS_URL",""); self._redis=None; self._lock=RLock()
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if redis and self.redis_url:
            try:
                c=redis.from_url(self.redis_url,decode_responses=True); c.ping(); self._redis=c
            except Exception: pass
    def _key(self,student_id,document_id=None):
        if not student_id or ":" in student_id: raise ValueError("invalid student_id")
        return f"v2:docs:{student_id}" if not document_id else f"v2:docs:{student_id}:{document_id}"
    def _load(self):
        if not self.path.exists(): return {}
        try: return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError,ValueError): return {}
    def _save(self,data):
        tmp=self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(data,indent=2),encoding="utf-8"); tmp.replace(self.path)
    @staticmethod
    def _chunks(text,size=1200,overlap=180):
        words=text.split(); out=[]; step=max(1,size-overlap)
        for i in range(0,len(words),step):
            chunk=" ".join(words[i:i+size]).strip()
            if chunk: out.append(chunk)
            if i+size>=len(words): break
        return out
    def ingest(self,student_id,document_id,text,source,page=None):
        chunks=self._chunks(text); now=time.time(); key=self._key(student_id,document_id)
        if self._redis:
            raw=self._redis.get(key); docs=json.loads(raw) if raw else []
        else:
            docs=self._load().get(student_id,{}).get(document_id,[])
        start=len(docs)
        docs.extend(asdict(DocumentChunk(f"{document_id}:{start+i}",document_id,student_id,chunk,source,page,now)) for i,chunk in enumerate(chunks))
        if self._redis: self._redis.set(key,json.dumps(docs))
        else:
            with self._lock:
                data=self._load(); data.setdefault(student_id,{})[document_id]=docs; self._save(data)
        return {"document_id":document_id,"chunks":len(chunks),"total_chunks":len(docs),"source":source}
    def list_documents(self,student_id):
        if self._redis:
            return [{"document_id":k.rsplit(":",1)[-1]} for k in self._redis.keys(self._key(student_id)+":*")]
        return [{"document_id":k} for k in self._load().get(student_id,{})]
    def _all(self,student_id):
        if self._redis:
            docs=[]
            for item in self.list_documents(student_id):
                raw=self._redis.get(self._key(student_id,item["document_id"]))
                if raw: docs.extend(json.loads(raw))
            return docs
        return [c for d in self._load().get(student_id,{}).values() for c in d]
    def retrieve(self,student_id,query,top_k=5):
        chunks=self._all(student_id); q=set(_tokens(query)); scored=[]
        for c in chunks:
            toks=_tokens(c["text"]); overlap=sum(1 for t in toks if t in q)
            if overlap: scored.append((overlap/(1+len(set(toks))),c))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [{"chunk_id":c["chunk_id"],"document_id":c["document_id"],"text":c["text"],"score":round(score,6),"metadata":{"source":c["source"],"page":c["page"]}} for score,c in scored[:max(1,min(top_k,20))]]
def _tokens(text): return re.findall(r"[a-z0-9_]+",text.lower())
document_store=PersistentDocumentStore()
