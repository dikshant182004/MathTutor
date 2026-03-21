from backend.agents import Agent_Exception, logger, sys
from backend.agents.nodes import AgentState
from backend.agents.utils.helper import MediaProcessor


def detect_input_type(state: AgentState) -> AgentState:
    try:
        if state.get("raw_text"):
            state["input_mode"] = "text"
        elif state.get("image_path"):
            state["input_mode"] = "image"
        elif state.get("audio_path"):
            state["input_mode"] = "audio"
        else:
            state["hitl_required"] = True
            state["hitl_reason"]   = "No valid input detected — please provide text, image, or audio."

        state.setdefault("solve_iterations", 0)
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
