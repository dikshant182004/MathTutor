from backend.agents import Agent_Exception, logger, sys
from backend.agents.nodes import AgentState
from backend.agents.utils.helper import MediaProcessor

from langchain_core.messages import RemoveMessage

def detect_input_type(state: AgentState) -> AgentState:
    try:
        # Clear all per-problem state
        state["solve_iterations"] = 0
        state["hitl_required"]    = False
        state["hitl_type"]        = None
        state["hitl_reason"]      = None
        state["solver_output"]    = None
        state["verifier_output"]  = None
        state["explainer_output"] = None
        state["solution_plan"]    = None
        state["parsed_data"]      = None
        state["safety_passed"]    = None
        state["safety_reason"]    = None
        state["student_satisfied"]   = None     # reset satisfaction/follow-up fields
        state["follow_up_question"]  = None
        state["human_feedback"]      = None
        state["user_corrected_text"] = None
        state["agent_payload_log"]= None
        state["direct_response_tool_calls"] = None 

        # Properly clear messages using RemoveMessage
        existing_messages = state.get("messages") or []
        if existing_messages:
            removes = [RemoveMessage(id=m.id) for m in existing_messages if getattr(m, "id", None)]
            if removes:
                state["messages"] = removes

        if state.get("raw_text"):
            state["input_mode"] = "text"
        elif state.get("image_path"):
            state["input_mode"] = "image"
        elif state.get("audio_path"):
            state["input_mode"] = "audio"
        else:
            state["hitl_required"] = True
            state["hitl_reason"]   = "No valid input detected — please provide text, image, or audio."

        return state
    except Exception as e:
        raise Agent_Exception(e, sys)
    

def ocr_node(state: AgentState) -> AgentState:

    try:
        processor = MediaProcessor()
        path = state.get("image_path")

        if not path:
            return state
        
        text, conf = processor.process_image(path)
        state["ocr_text"] = text
        state["ocr_confidence"] = conf

        logger.info(f"[OCR] conf={conf:.2f} text_len={len(text)}")
        return state
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise Agent_Exception(e, sys)


def asr_node(state: AgentState) -> AgentState:
    
    try:
        processor = MediaProcessor()
        path = state.get("audio_path")

        if not path:
            return state
        transcript, conf  = processor.process_audio(path)
        state["transcript"] = transcript
        state["asr_confidence"] = conf

        logger.info(f"[ASR] conf={conf:.2f} transcript_len={len(transcript)}")
        return state
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        raise Agent_Exception(e, sys)
