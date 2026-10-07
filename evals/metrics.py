from __future__ import annotations
from math import log2

def accuracy(values):
    values=list(values)
    return sum(bool(v) for v in values)/len(values) if values else 0.0

def recall_at_k(relevant_ids, retrieved_ids, k=5):
    relevant=set(relevant_ids)
    if not relevant:return 0.0
    return len(relevant.intersection(retrieved_ids[:k]))/len(relevant)

def reciprocal_rank(relevant_ids, retrieved_ids):
    relevant=set(relevant_ids)
    for rank,item in enumerate(retrieved_ids,1):
        if item in relevant:return 1.0/rank
    return 0.0

def ndcg_at_k(relevance, k=5):
    vals=list(relevance)[:k]
    dcg=sum((2**rel-1)/log2(i+2) for i,rel in enumerate(vals))
    ideal=sorted(vals,reverse=True)
    idcg=sum((2**rel-1)/log2(i+2) for i,rel in enumerate(ideal))
    return dcg/idcg if idcg else 0.0

def verification_catch_rate(results):
    return accuracy(caught for was_wrong,caught in results if was_wrong)

def memory_isolation(results):
    return accuracy(a and b for a,b in results)

def latency_percentile(values, percentile=0.95):
    values=sorted(values)
    if not values:return 0.0
    idx=max(0,min(len(values)-1,round((len(values)-1)*percentile)))
    return float(values[idx])

def summarize_traces(traces):
    latencies=[t.get("latency_ms",0) for t in traces]
    return {"requests":len(traces),"p50_latency_ms":latency_percentile(latencies,.50),"p95_latency_ms":latency_percentile(latencies,.95),"llm_calls":sum(t.get("llm_calls",0) for t in traces),"tool_calls":sum(t.get("tool_calls",0) for t in traces),"input_tokens":sum(t.get("input_tokens",0) for t in traces),"output_tokens":sum(t.get("output_tokens",0) for t in traces)}
