from __future__ import annotations
import pytest
from backend.agents.routing_policy import route, route_with_reason
@pytest.mark.parametrize(("state","target","reason"),[
({"guardrail_passed":False},"END","input_blocked"),
({"prefilter_passed":False},"END","input_blocked"),
({"hitl_required":True},"hitl_node","clarification_required"),
({"intake_output":{"needs_clarification":True}},"hitl_node","clarification_required"),
({"solution_plan":{"intent_type":"explain"}},"direct_response_node","direct_intent:explain"),
({"solution_plan":{"intent_type":"research"}},"direct_response_node","direct_intent:research"),
({"solution_plan":{"intent_type":"generate"}},"direct_response_node","direct_intent:generate"),
({"solution_plan":{"intent_type":"formula_lookup"}},"direct_response_node","direct_intent:formula_lookup"),
({"solution_plan":{"intent_type":"hint"}},"solver_agent","hint_mode"),
({"solution_plan":{"intent_type":"solve","confidence":.9},"complexity":"trivial"},"solver_agent","trivial_solve_skip_retrieval"),
({"solution_plan":{"intent_type":"solve","confidence":.9},"complexity":"standard"},"retrieve_ltm","standard_solve_retrieval"),
({"solution_plan":{"intent_type":"solve","confidence":.9},"complexity":"hard"},"retrieve_ltm","hard_solve_retrieval_web_allowed"),
({"solution_plan":{"intent_type":"solve","confidence":.4},"complexity":"standard"},"retrieve_ltm","hard_solve_retrieval_web_allowed"),
({"solution_plan":{"intent_type":"solve","confidence":.4},"complexity":"trivial"},"retrieve_ltm","standard_solve_retrieval"),
({"solution_plan":{"intent_type":"unknown","confidence":.9},"complexity":"standard"},"retrieve_ltm","standard_solve_retrieval"),
({"solution_plan":{"intent_type":"solve","confidence":"not-a-number"},"complexity":"standard"},"retrieve_ltm","hard_solve_retrieval_web_allowed"),
])
def test_every_routing_policy_branch(state,target,reason):
    before=dict(state)
    assert route(state)==target
    assert route_with_reason(state)==(target,reason)
    assert state==before
