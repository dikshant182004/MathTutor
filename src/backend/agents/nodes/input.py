
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

