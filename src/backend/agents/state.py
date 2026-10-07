from backend.agents import Annotated, List, Optional, TypedDict, BaseMessage
from langgraph.graph.message import add_messages
from operator import add


class AgentState(TypedDict):
    student_id: Optional[str]
    input_mode: str
    raw_text: Optional[str]
    image_path: Optional[str]
    audio_path: Optional[str]
    thread_id: Optional[str]

    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    transcript: Optional[str]
    asr_confidence: Optional[float]
    user_corrected_text: Optional[str]

    parsed_data: Optional[dict]
    execution_plan: Optional[dict]
    solution_plan: Optional[dict]
    retrieved_context: Optional[str]
    rag_citations: Optional[List[dict]]
    solver_output: Optional[dict]
    verifier_output: Optional[dict]
    safety_passed: Optional[bool]
    safety_reason: Optional[str]
    explainer_output: Optional[dict]
    solve_iterations: int

    agent_payload_log: Optional[List[dict]]
    direct_response_tool_calls: Optional[list]
    conversation_log: Annotated[List[str], add]
    final_response: Optional[str]

    hitl_required: bool
    hitl_reason: Optional[str]
    hitl_type: Optional[str]
    hitl_interrupt: Optional[dict]
    human_feedback: Optional[str]
    student_satisfied: Optional[bool]
    follow_up_question: Optional[str]

    guardrail_passed: Optional[bool]
    guardrail_reason: Optional[str]

    ltm_mode: Optional[str]
    ltm_context: Optional[dict]
    ltm_stored: Optional[bool]

    messages: Annotated[list[BaseMessage], add_messages]


def make_initial_state(
    student_id: str,
    thread_id: str,
    raw_text: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> AgentState:
    return AgentState({
        "student_id": student_id,
        "thread_id": thread_id,
        "raw_text": raw_text,
        "image_path": image_path,
        "audio_path": audio_path,
        "input_mode": None,
        "ocr_text": None,
        "ocr_confidence": None,
        "transcript": None,
        "asr_confidence": None,
        "user_corrected_text": None,
        "parsed_data": None,
        "execution_plan": None,
        "solution_plan": None,
        "retrieved_context": None,
        "rag_citations": [],
        "messages": [],
        "solve_iterations": 0,
        "solver_output": None,
        "verifier_output": None,
        "safety_passed": None,
        "safety_reason": None,
        "explainer_output": None,
        "final_response": None,
        "hitl_required": False,
        "hitl_type": None,
        "hitl_reason": None,
        "hitl_interrupt": None,
        "user_corrected_text": None,
        "human_feedback": None,
        "student_satisfied": None,
        "follow_up_question": None,
        "guardrail_passed": None,
        "guardrail_reason": None,
        "ltm_mode": None,
        "ltm_context": None,
        "ltm_stored": None,
        "agent_payload_log": [],
        "conversation_log": [],
        "direct_response_tool_calls": [],
    })
