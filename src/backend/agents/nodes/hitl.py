from backend.agents import *
from backend.agents.nodes import *
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt


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
    """

    def hitl_node(self, state: AgentState) -> dict:
        try:
            reason    = state.get("hitl_reason") or "Human review required."
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

            rich_prompt = interrupt_payload.get("prompt") or reason
            interrupt_payload["hitl_reason"] = rich_prompt

            payload(
                state, "hitl_node",
                summary = f"INTERRUPT [{hitl_type.upper()}] — waiting for human",
                fields = {
                    "Type":   hitl_type,
                    "Reason": (rich_prompt or "")[:280],
                },
            )
            logger.info(
                f"[HITL] Suspending graph | type={hitl_type} | "
                f"reason={rich_prompt[:80]}"
            )

            try:
                human_response: dict = interrupt(interrupt_payload)
            except GraphInterrupt:
                raise

            logger.info(f"[HITL] Resumed | type={hitl_type} | response={human_response}")

            # ── Process human response into state updates ─────────────────────
            if hitl_type == "bad_input":
                return _process_bad_input_response(human_response)

            elif hitl_type == "clarification":
                return _process_clarification_response(human_response, state)

            elif hitl_type == "verification":
                return _process_verification_response(human_response)

            elif hitl_type == "satisfaction":
                return _process_satisfaction_response(human_response, state)

            else:
                return {"hitl_required": False, "hitl_type": None, "hitl_interrupt": None}

        except GraphInterrupt:
            raise

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
    """
    parsed       = state.get("parsed_data") or {}
    ocr_text     = (state.get("ocr_text") or "").strip()
    problem_text = (
        parsed.get("problem_text")
        or state.get("user_corrected_text")
        or ocr_text
        or state.get("transcript")
        or state.get("raw_text")
        or ""
    )

    if ocr_text and ocr_text != problem_text:
        llm_reason = (
            f"{reason}\n\n"
            f"We extracted this from your image:\n\"{ocr_text}\"\n\n"
            f"Please type the complete problem with all values filled in."
        )
    else:
        llm_reason = reason or "This problem seems incomplete or ambiguous."

    return {
        "hitl_type":    "clarification",
        "message":      reason,
        "problem_text": problem_text,
        "prompt":       llm_reason,
    }


def _build_verification_interrupt(state: AgentState, reason: str) -> dict:
    """
    Verifier cannot determine correctness — needs expert review.
    UI: show problem + full solver working.
    """
    parsed       = state.get("parsed_data") or {}
    solver_out   = state.get("solver_output") or {}
    verifier_out = state.get("verifier_output") or {}

    verdict  = verifier_out.get("verdict", "")
    fix_hint = verifier_out.get("suggested_fix", "")

    llm_message = verdict or reason
    if fix_hint:
        llm_message = f"{llm_message}\n\nSuggested issue: {fix_hint}"

    return {
        "hitl_type":        "verification",
        "message":          reason,
        "problem_text":     parsed.get("problem_text", ""),
        "solution":         solver_out.get("solution", ""),
        "final_answer":     solver_out.get("final_answer", ""),
        "verifier_verdict": verifier_out.get("verdict", ""),
        "solve_attempts":   state.get("solve_iterations", 1),
        "prompt":           llm_message,
    }


def _build_satisfaction_interrupt(state: AgentState) -> dict:
    """
    Shown after the explanation is delivered to the student.
    UI: thumbs-up / thumbs-down + optional follow-up text-area.
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
# ──────────────────────────────────────────────────────────────────────────────

def _process_bad_input_response(r: dict) -> dict:
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


def _process_clarification_response(r: dict, state: AgentState) -> dict:
    corrected   = (r.get("corrected_text") or "").strip()
    parsed      = state.get("parsed_data") or {}
    original    = (
        parsed.get("problem_text")
        or state.get("raw_text")
        or ""
    ).strip()
 
    if not corrected:
        # User submitted empty form — keep original
        merged = original
    elif original and corrected.lower() not in original.lower() and original.lower() not in corrected.lower():
        # Clarification adds new info — merge them
        merged = f"{original}\n[Student clarification: {corrected}]"
        logger.info(f"[HITL] Merged clarification: original='{original[:50]}' + clarification='{corrected[:50]}'")
    else:
        # Clarification IS the full question (user retyped it) — use as-is
        merged = corrected
        logger.info(f"[HITL] Using clarification as-is: '{corrected[:80]}'")
 
    return {
        "user_corrected_text": merged,
        "hitl_required":       False,
        "hitl_interrupt":      None,
        "hitl_reason":         None,
        "parsed_data":         None,
        "ltm_context":         state.get("ltm_context"),   # FIX: preserve LTM
        "solution_plan":       None,   # re-route after re-parse
        "solver_output":       None,
        "verifier_output":     None,
    }


def _process_verification_response(r: dict) -> dict:
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



def _process_satisfaction_response(r: dict, state: AgentState) -> dict:
    """
    Stores satisfaction result; route_after_hitl sends to END or re-explains.
    """
    satisfied    = bool(r.get("satisfied", False))
    follow_up    = (r.get("follow_up") or "").strip()
 
    update: dict = {
        "student_satisfied":  satisfied,
        "follow_up_question": follow_up or None,
        "hitl_required":      False,
        "hitl_interrupt":     None,
        "hitl_reason":        None,
        # "final_response":     None,   # clear so UI shows new response
    }
 
    if not satisfied and follow_up:
        # Inject the follow-up into the problem context for re-run
        parsed          = state.get("parsed_data") or {}
        original_text   = parsed.get("problem_text") or state.get("raw_text") or ""
        injected_text   = (
            f"{original_text}\n\n"
            f"[Student follow-up question: {follow_up}]"
        ).strip()
        # Use user_corrected_text as the injection vector — parser checks this first
        update["user_corrected_text"] = injected_text
        if parsed:
            updated_parsed = dict(parsed)
            updated_parsed["problem_text"] = injected_text
            update["parsed_data"] = updated_parsed
        logger.info(
            f"[HITL] Follow-up injected into problem context: {follow_up[:80]}"
        )
 
    return update