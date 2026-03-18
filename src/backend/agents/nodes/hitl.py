from agents import *
from agents.nodes import *
from langgraph.types import interrupt


class HITLAgent(BaseAgent):
    """
    Human-in-the-Loop node — the single suspension point in the graph.

    Four HITL scenarios:

    ┌─────────────────┬──────────────────────────────┬────────────────────────────┐
    │ hitl_type       │ Triggered by                 │ Human action               │
    ├─────────────────┼──────────────────────────────┼────────────────────────────┤
    │ bad_input       │ ocr_node / asr_node          │ Re-upload image/audio      │
    │                 │ conf < 0.5 or empty text     │ or type text directly      │
    ├─────────────────┼──────────────────────────────┼────────────────────────────┤
    │ clarification   │ parser_agent                 │ Rephrase / complete the    │
    │                 │ needs_clarification = True   │ problem                    │
    ├─────────────────┼──────────────────────────────┼────────────────────────────┤
    │ verification    │ verifier_agent               │ Expert marks correct /     │
    │                 │ status = "needs_human"       │ incorrect + optional hint  │
    ├─────────────────┼──────────────────────────────┼────────────────────────────┤
    │ satisfaction    │ explainer_agent (always)     │ Student thumbs up/down     │
    │                 │ after explanation delivered  │ + optional follow-up       │
    └─────────────────┴──────────────────────────────┴────────────────────────────┘

    How interrupt() works (LangGraph >= 0.2):
        interrupt(payload) suspends the graph at this node, checkpoints state,
        and returns control to the caller.  Unlike the old NodeInterrupt
        exception, interrupt() is a regular function call that RESUMES in the
        same node and returns the human's response as its return value.

        Flow:
            human_response = interrupt(payload)   ← suspends here, UI sees payload
            # resumes here when graph.invoke(Command(resume=response), config) is called
            return process(human_response)        ← state update returned normally

        On resume from UI:
            from langgraph.types import Command
            graph.invoke(Command(resume={"corrected_text": "..."}), config)

    IMPORTANT: Requires a checkpointer in your compiled graph:
        from langgraph.checkpoint.memory import MemorySaver
        graph = workflow.compile(checkpointer=MemorySaver())
    """

    def hitl_node(self, state: AgentState) -> dict:
        try:
            reason    = state.get("hitl_reason", "Human review required.")
            hitl_type = state.get("hitl_type")

            # ── Auto-detect type if not explicitly set by triggering node ─────
            if not hitl_type:
                if state.get("input_mode") in ("image", "audio") and not state.get("parsed_data"):
                    hitl_type = "bad_input"
                elif state.get("parsed_data") and not state.get("solution_plan"):
                    hitl_type = "clarification"
                elif state.get("verifier_output"):
                    hitl_type = "verification"
                elif state.get("final_response"):
                    hitl_type = "satisfaction"
                else:
                    hitl_type = "clarification"

            # ── Build scenario-specific interrupt payload ─────────────────────
            if hitl_type == "bad_input":
                interrupt_payload = _build_bad_input_interrupt(state, reason)

            elif hitl_type == "clarification":
                interrupt_payload = _build_clarification_interrupt(state, reason)

            elif hitl_type == "verification":
                interrupt_payload = _build_verification_interrupt(state, reason)

            elif hitl_type == "satisfaction":
                interrupt_payload = _build_satisfaction_interrupt(state)

            else:
                interrupt_payload = {
                    "hitl_type": hitl_type,
                    "message":   reason,
                    "prompt":    "Please provide the required input.",
                }

            payload(
                state, "hitl_node",
                summary = f"INTERRUPT [{hitl_type.upper()}] — waiting for human",
                fields  = {"Type": hitl_type, "Reason": reason[:120]},
            )
            logger.info(
                f"[HITL] Suspending graph | type={hitl_type} | reason={reason[:80]}"
            )

            # ── Suspend and wait for human response ───────────────────────────
            # interrupt() checkpoints state and pauses execution here.
            # When the graph is resumed via Command(resume=...), execution
            # continues from this exact line and human_response holds the
            # dict passed inside Command(resume=...).
            human_response: dict = interrupt(interrupt_payload)

            logger.info(f"[HITL] Resumed | type={hitl_type} | response={human_response}")

            # ── Process human response into state updates ─────────────────────
            if hitl_type == "bad_input":
                return _process_bad_input_response(human_response)

            elif hitl_type == "clarification":
                return _process_clarification_response(human_response)

            elif hitl_type == "verification":
                return _process_verification_response(human_response)

            elif hitl_type == "satisfaction":
                return _process_satisfaction_response(human_response)

            else:
                return {"hitl_required": False, "hitl_type": None, "hitl_interrupt": None}

        except Exception as e:
            logger.error(f"[HITL] failed: {e}")
            raise Agent_Exception(e, sys)


# ──────────────────────────────────────────────────────────────────────────────
# INTERRUPT PAYLOAD BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_bad_input_interrupt(state: AgentState, reason: str) -> dict:
    """
    OCR / ASR confidence too low or empty text.
    UI: re-upload widget + 'type it instead' fallback.

    Expected resume dict keys:
        new_image_path : str | None
        new_audio_path : str | None
        raw_text       : str | None
    """
    input_mode = state.get("input_mode", "unknown")
    conf_key   = "ocr_confidence" if input_mode == "image" else "asr_confidence"
    conf       = state.get(conf_key)

    return {
        "hitl_type":  "bad_input",
        "message":    reason,
        "input_mode": input_mode,
        "confidence": round(conf, 2) if conf is not None else None,
        "prompt": (
            "We couldn't read your image clearly. "
            "Please upload a sharper photo, or type the problem directly."
            if input_mode == "image"
            else
            "We couldn't transcribe your audio clearly. "
            "Please re-record in a quieter environment, or type the problem directly."
        ),
    }


def _build_clarification_interrupt(state: AgentState, reason: str) -> dict:
    """
    Parser decided the problem is ambiguous or incomplete.
    UI: text-area pre-filled with current problem text.

    Expected resume dict keys:
        corrected_text : str
    """
    parsed       = state.get("parsed_data") or {}
    problem_text = (
        parsed.get("problem_text")
        or state.get("user_corrected_text")
        or state.get("ocr_text")
        or state.get("transcript")
        or state.get("raw_text")
        or ""
    )

    return {
        "hitl_type":    "clarification",
        "message":      reason,
        "problem_text": problem_text,
        "prompt": (
            "This problem seems incomplete or ambiguous. "
            "Please clarify or retype it below so we can solve it correctly."
        ),
    }


def _build_verification_interrupt(state: AgentState, reason: str) -> dict:
    """
    Verifier cannot determine correctness — needs expert review.
    UI: show problem + full solver working.

    Expected resume dict keys:
        is_correct : bool
        fix_hint   : str  (optional)
    """
    parsed       = state.get("parsed_data") or {}
    solver_out   = state.get("solver_output") or {}
    verifier_out = state.get("verifier_output") or {}

    return {
        "hitl_type":        "verification",
        "message":          reason,
        "problem_text":     parsed.get("problem_text", ""),
        "solution":         solver_out.get("solution", ""),
        "final_answer":     solver_out.get("final_answer", ""),
        "verifier_verdict": verifier_out.get("verdict", ""),
        "solve_attempts":   state.get("solve_iterations", 1),
        "prompt": (
            "The automated verifier could not determine whether this solution is correct. "
            "Please review the working and mark it as correct or incorrect. "
            "If incorrect, briefly describe what went wrong."
        ),
    }


def _build_satisfaction_interrupt(state: AgentState) -> dict:
    """
    Shown after the explanation is delivered to the student.
    UI: thumbs-up / thumbs-down + optional follow-up text-area.

    Expected resume dict keys:
        satisfied  : bool
        follow_up  : str  (optional)
    """
    return {
        "hitl_type": "satisfaction",
        "message":   "Was this explanation helpful?",
        "prompt": (
            "Did this explanation make sense? "
            "If not, tell us what was unclear and we will re-explain it."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE PROCESSORS
# Called after interrupt() returns with the human's response dict.
# Each returns the state update dict that hitl_node returns to the graph.
# NOTE: hitl_type is intentionally kept in state — route_after_hitl reads it.
# ──────────────────────────────────────────────────────────────────────────────

def _process_bad_input_response(r: dict) -> dict:
    """Clears stale media outputs so ocr_node / asr_node re-run cleanly."""
    update: dict = {
        "hitl_required":  False,
        "hitl_interrupt": None,
        "hitl_reason":    None,
        "ocr_text":       None,
        "transcript":     None,
        "ocr_confidence": None,
        "asr_confidence": None,
    }
    if r.get("new_image_path"):
        update["image_path"] = r["new_image_path"]
        update["input_mode"] = "image"
    if r.get("new_audio_path"):
        update["audio_path"] = r["new_audio_path"]
        update["input_mode"] = "audio"
    if r.get("raw_text"):
        update["raw_text"]   = r["raw_text"].strip()
        update["input_mode"] = "text"
    return update


def _process_clarification_response(r: dict) -> dict:
    """Stores corrected text and clears parsed_data to force parser re-run."""
    return {
        "user_corrected_text": r.get("corrected_text", "").strip(),
        "hitl_required":       False,
        "hitl_interrupt":      None,
        "hitl_reason":         None,
        "parsed_data":         None,
    }


def _process_verification_response(r: dict) -> dict:
    """Overwrites verifier_output with the human verdict."""
    is_correct = bool(r.get("is_correct", False))
    fix_hint   = r.get("fix_hint", "").strip()

    verifier_override: dict = {
        "status":        "correct" if is_correct else "incorrect",
        "verdict":       (
            "Marked correct by human reviewer."
            if is_correct
            else f"Marked incorrect by human reviewer. Hint: {fix_hint}".rstrip(". ") + "."
        ),
        "confidence":    1.0,
        "suggested_fix": fix_hint if (not is_correct and fix_hint) else None,
        "hitl_reason":   None,
    }
    return {
        "verifier_output": verifier_override,
        "human_feedback":  fix_hint if fix_hint else None,
        "hitl_required":   False,
        "hitl_interrupt":  None,
        "hitl_reason":     None,
    }


def _process_satisfaction_response(r: dict) -> dict:
    """Stores satisfaction result; route_after_hitl sends to END or explainer."""
    return {
        "student_satisfied":  bool(r.get("satisfied", False)),
        "follow_up_question": r.get("follow_up", "").strip() or None,
        "hitl_required":      False,
        "hitl_interrupt":     None,
        "hitl_reason":        None,
    }

