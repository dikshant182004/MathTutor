from backend.v2.knowledge_graph import prerequisite_edges
from backend.v2.practice import generate_problem

def test_prerequisite_graph():
    assert {"source":"skill:algebra","target":"skill:calculus","type":"prerequisite"} in prerequisite_edges("calculus")

def test_practice_generation_is_deterministic():
    a=generate_problem("algebra","easy",seed=7); b=generate_problem("algebra","easy",seed=7)
    assert a==b and a.prompt and a.answer
