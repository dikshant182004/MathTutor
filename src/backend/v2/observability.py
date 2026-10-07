from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class RunMetrics:
    trace_id: str
    started_at: float = field(default_factory=perf_counter)
    node_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def latency_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000


@contextmanager
def measure_node(metrics: RunMetrics, node: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        metrics.node_ms[node] = metrics.node_ms.get(node, 0.0) + (perf_counter() - start) * 1000
