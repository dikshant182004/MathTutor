from backend.v2.observability import RunMetrics, measure_node


def test_node_timer_records_latency():
    metrics = RunMetrics(trace_id="test")
    with measure_node(metrics, "solver"):
        pass
    assert "solver" in metrics.node_ms
    assert metrics.node_ms["solver"] >= 0
