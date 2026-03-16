from backend.agents import Annotated, List, Optional, TypedDict, BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
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

    parsed_data:        Optional[dict]           # ParserOutput
    solution_plan:      Optional[dict]           # IntentRouterOutput
    retrieved_context:  Optional[str]
    solver_output:      Optional[dict]           # SolverOutput
    verifier_output:    Optional[dict]           # VerifierOutput
    explainer_output:   Optional[dict]           # ExplainerOutput
    solve_iterations:   int                      # solver↔verifier loop counter
    manim_video_path:   Optional[str]            # path returned by MCP server

    agent_payload_log: Optional[List[dict]]
    conversation_log: Optional[List[str]]
    final_response: Optional[str]

    hitl_required: bool
    hitl_reason:   Optional[str]
    human_feedback: Optional[str]                # answer injected after interrupt

    messages: Annotated[list[BaseMessage], add_messages]
