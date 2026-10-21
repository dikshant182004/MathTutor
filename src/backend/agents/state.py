from backend.agents import Annotated, List, Optional, TypedDict, BaseMessage
from langgraph.graph.message import add_messages
from operator import add


class AgentState(TypedDict):

    student_id: Optional[str]

    input_mode:  str
    raw_text:    Optional[str]
    image_path:  Optional[str]
    audio_path:  Optional[str]
    thread_id:   Optional[str]

    ocr_text:        Optional[str]
    ocr_confidence:  Optional[float] 
    transcript:      Optional[str]
    asr_confidence:  Optional[float] 
    user_corrected_text: Optional[str]

    prefilter_passed: Optional[bool]
    cleaned_problem: Optional[str]
    complexity: Optional[str]
    intake_output: Optional[dict]
    route: Optional[str]
    route_reason: Optional[str]
    parsed_data:        Optional[dict]           # ParserOutput
    solution_plan:      Optional[dict]           # IntentRouterOutput
    retrieved_context:  Optional[str]
    retrieved_chunks: Optional[dict]
    solver_output:      Optional[dict]           # SolverOutput
    verifier_output:    Optional[dict]           # VerifierOutput
    safety_passed: Optional[bool]
    safety_reason: Optional[str]
    # Structured deterministic composition replaces the legacy explainer output.
    solution_struct: Optional[dict]
    solve_iterations:   int                      # solver↔verifier loop counter

    agent_payload_log: Optional[List[dict]]
    direct_response_tool_calls: Optional[list]   # web search signals from DirectResponseAgent
    conversation_log: Annotated[List[str], add]
    final_response: Optional[str]

    hitl_required        : bool
    hitl_reason          : Optional[str]
    hitl_type            : Optional[str]         
    hitl_interrupt       : Optional[dict]        
    human_feedback       : Optional[str]         
    student_satisfied    : Optional[bool]        
    follow_up_question   : Optional[str]         

    guardrail_passed: Optional[bool]
    guardrail_reason: Optional[str]
    
    ltm_mode: Optional[str]
    ltm_context: Optional[dict]
    ltm_stored: Optional[bool]

    messages: Annotated[list[BaseMessage], add_messages]

    # Per-request Phase C observability and bounded execution.
    node_timings_ms: Optional[dict]
    max_llm_calls: Optional[int]
    max_tool_calls: Optional[int]
    llm_calls_used: int
    tool_calls_used: int
    budget_exhausted: Optional[bool]


def make_initial_state(
    student_id:  str,
    thread_id:   str,
    raw_text:    str | None = None,
    image_path:  str | None = None,
    audio_path:  str | None = None,
) -> AgentState:
    """
    Builds the initial AgentState for a new problem-solving session.
    Call this in your UI layer when the student submits a new problem:
    """
    return AgentState({
        # Identity
        "student_id":          student_id,
        "thread_id":           thread_id,
 
        # Raw input (exactly one should be non-None)
        "raw_text":            raw_text,
        "image_path":          image_path,
        "audio_path":          audio_path,
 
        # Input processing
        "input_mode":          None,
        "ocr_text":            None,
        "ocr_confidence":      None,
        "transcript":          None,
        "asr_confidence":      None,
 
        # HITL
        "hitl_required":       False,
        "hitl_type":           None,
        "hitl_reason":         None,
        "hitl_interrupt":      None,
        "user_corrected_text": None,
        "human_feedback":      None,
        "student_satisfied":   None,
        "follow_up_question":  None,
 
        # Guardrail
        "guardrail_passed":    None,
        "guardrail_reason":    None,
 
        # Parsing & routing
        "prefilter_passed":    None,
        "cleaned_problem":     None,
        "complexity":          None,
        "intake_output":       None,
        "route":               None,
        "route_reason":        None,
        "parsed_data":         None,
        "solution_plan":       None,
        "retrieved_context":   None,
        "retrieved_chunks":    None, 
 
        # Solver
        "messages":            [],
        "solve_iterations":    0,
        "solver_output":       None,
 
        # Verification & safety
        "verifier_output":     None,
        "safety_passed":       None,
        "safety_reason":       None,
 
        # Explanation
        "solution_struct":     None,
        "final_response":      None,
 
        # Memory
        "ltm_mode":            None,
        "ltm_context":         None,
        "ltm_stored":          None,

        "agent_payload_log":   [],
        "conversation_log":    [],

        # Request-local budgets/telemetry. Defaults are bounded and overridable.
        "node_timings_ms":     {},
        # None means complexity-specific defaults: trivial=2, standard=4, hard=6.
        "max_llm_calls":       None,
        "max_tool_calls":      3,
        "llm_calls_used":      0,
        "tool_calls_used":     0,
        "budget_exhausted":    False,
    })
 