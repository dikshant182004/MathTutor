from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Iterator
from uuid import uuid4

@dataclass
class RunMetrics:
    trace_id: str
    started_at: float = field(default_factory=time.perf_counter)
    node_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "running"
    error: str | None = None

    @property
    def latency_ms(self):
        return (time.perf_counter() - self.started_at) * 1000

    def finish(self, status="ok", error=None):
        self.status=status; self.error=error
        return {"trace_id":self.trace_id,"latency_ms":round(self.latency_ms,2),
                "node_ms":self.node_ms,"llm_calls":self.llm_calls,"tool_calls":self.tool_calls,
                "input_tokens":self.input_tokens,"output_tokens":self.output_tokens,
                "status":self.status,"error":self.error}

@contextmanager
def measure_node(metrics: RunMetrics, node: str) -> Iterator[None]:
    start=time.perf_counter()
    try: yield
    finally: metrics.node_ms[node]=metrics.node_ms.get(node,0.0)+(time.perf_counter()-start)*1000

class TraceStore:
    def __init__(self, path=".data/v2_traces.jsonl"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._lock=RLock()

    def append(self, trace: dict):
        line=json.dumps(trace, separators=(",",":"))
        with self._lock:
            with self.path.open("a",encoding="utf-8") as f: f.write(line+"\n")

    def recent(self, limit=50):
        if not self.path.exists(): return []
        with self._lock:
            lines=self.path.read_text(encoding="utf-8").splitlines()[-max(1,min(limit,500)):]
        out=[]
        for line in reversed(lines):
            try: out.append(json.loads(line))
            except ValueError: pass
        return out

def new_trace() -> RunMetrics:
    return RunMetrics(trace_id=str(uuid4()))

trace_store=TraceStore()
