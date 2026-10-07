from evals.metrics import accuracy, ndcg_at_k, recall_at_k, reciprocal_rank

def test_eval_metrics():
    assert accuracy([True, True, False]) == 2/3
    assert recall_at_k(["a","b"], ["x","b","a"], 2) == 0.5
    assert reciprocal_rank(["a"], ["x","a"]) == 0.5
    assert ndcg_at_k([3,2,0], 3) > 0
