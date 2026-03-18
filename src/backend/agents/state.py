from backend.agents import Annotated, List, Optional, TypedDict, BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    student_id: Optional[str]

    input_mode:  str
    raw_text:    Optional[str]
    image_path:  Optional[str]
    audio_path:  Optional[str]
    thread_id:   Optional[str]

    ocr_text:        Optional[str]
    transcript:      Optional[str]
    user_corrected_text: Optional[str]

    parsed_data:        Optional[dict]           # ParserOutput
    solution_plan:      Optional[dict]           # IntentRouterOutput
    retrieved_context:  Optional[str]
    solver_output:      Optional[dict]           # SolverOutput
    verifier_output:    Optional[dict]           # VerifierOutput
    safety_passed: Optional[bool]
    safety_reason: Optional[str]
    explainer_output:   Optional[dict]           # ExplainerOutput
    solve_iterations:   int                      # solver↔verifier loop counter
    manim_video_path:   Optional[str]            # path returned by MCP server

    agent_payload_log: Optional[List[dict]]
    conversation_log: Optional[List[str]]
    final_response: Optional[str]

    hitl_required        : bool
    hitl_reason          : Optional[str]
    hitl_type            : Optional[str]         
    hitl_interrupt       : Optional[dict]        
    user_corrected_text  : Optional[str]         
    human_feedback       : Optional[str]         
    student_satisfied    : Optional[bool]        
    follow_up_question   : Optional[str]         

    guardrail_passed: Optional[bool]
    guardrail_reason: Optional[str]
    
    ltm_mode: Optional[str]
    ltm_context: Optional[dict]
    ltm_stored: Optional[bool]

    messages: Annotated[list[BaseMessage], add_messages]


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
 
        from agents.state import make_initial_state
        from nodes.memory_manager import create_thread
 
        thread_id = create_thread(student_id)
        config    = {"configurable": {"thread_id": thread_id}}
        state     = make_initial_state(
            student_id = student_id,
            thread_id  = thread_id,
            raw_text   = user_input,   # or image_path / audio_path
        )
        result = graph.invoke(state, config)
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
        "parsed_data":         None,
        "solution_plan":       None,
 
        # Solver
        "messages":            [],
        "solve_iterations":    0,
        "solver_output":       None,
 
        # Verification & safety
        "verifier_output":     None,
        "safety_passed":       None,
        "safety_reason":       None,
 
        # Explanation
        "explainer_output":    None,
        "final_response":      None,
        "manim_scene_code":    None,
 
        # Memory
        "ltm_mode":            None,
        "ltm_context":         None,
        "ltm_stored":          None,
    })
 