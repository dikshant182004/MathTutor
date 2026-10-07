from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterator
from uuid import uuid4

_current_trace: ContextVar["RunMetrics|None"] = ContextVar("mathtutor_trace", default=None)

@dataclass
class RunMetrics:
    trace_id: str
    started_at: float = field(default_factory=time.perf_counter)
    node_ms: dict[str,float] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "running"
    error: str | None = None

    @property
    def latency_ms(self): return (time.perf_counter()-self.started_at)*1000
    def finish(self,status="ok",error=None):
        self.status=status; self.error=error
        return {"trace_id":self.trace_id,"latency_ms":round(self.latency_ms,2),
                "node_ms":self.node_ms,"llm_calls":self.llm_calls,"tool_calls":self.tool_calls,
                "input_tokens":self.input_tokens,"output_tokens":self.output_tokens,
                "status":self.status,"error":self.error}

@contextmanager
def active_trace(metrics: RunMetrics) -> Iterator[RunMetrics]:
    token=_current_trace.set(metrics)
    try: yield metrics
    finally: _current_trace.reset(token)

def current_trace(): return _current_trace.get()

@contextmanager
def measure_node(metrics: RunMetrics,node: str)->Iterator[None]:
    start=time.perf_counter()
    try: yield
    finally: metrics.node_ms[node]=metrics.node_ms.get(node,0.0)+(time.perf_counter()-start)*1000

class TraceCallback:
    """Small callback bridge; it records only real provider metadata."""

    def on_llm_start(self,*_args,**_kwargs):
        m=current_trace()
        if m: m.llm_calls+=1

    def on_llm_end(self,response,**_kwargs):
        m=current_trace()
        if not m: return
        usage=getattr(response,"llm_output",None) or {}
        usage=usage.get("token_usage") or {}
        m.input_tokens += int(usage.get("prompt_tokens",0) or usage.get("input_tokens",0) or 0)
        m.output_tokens += int(usage.get("completion_tokens",0) or usage.get("output_tokens",0) or 0)
        if not usage:
            for gen in getattr(response,"generations",[]) or []:
                for item in gen or []:
                    meta=getattr(getattr(item,"message",None),"usage_metadata",None) or {}
                    m.input_tokens += int(meta.get("input_tokens",0) or 0)
                    m.output_tokens += int(meta.get("output_tokens",0) or 0)

    def on_tool_start(self,*_args,**_kwargs):
        m=current_trace()
        if m: m.tool_calls+=1

def callback_handler():
    return TraceCallback()

class TraceStore:
    def __init__(self,path=".data/v2_traces.jsonl"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._lock=RLock()
    def append(self,trace):
        with self._lock:
            with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(trace,separators=(",",":"))+"\n")
    def recent(self,limit=50):
        if not self.path.exists(): return []
        with self._lock: lines=self.path.read_text(encoding="utf-8").splitlines()[-max(1,min(limit,500)):]
        out=[]
        for line in reversed(lines):
            try: out.append(json.loads(line))
            except ValueError: pass
        return out

def new_trace(): return RunMetrics(trace_id=str(uuid4()))
trace_store=TraceStore()
