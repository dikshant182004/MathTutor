from backend.agents import List, Optional, Union
from pydantic import BaseModel, Field, model_validator
from backend.agents.utils.helper import _coerce_bools


class ParserOutput(BaseModel):

    problem_text:         str           = Field(...,  description="Cleaned, complete problem statement ready for solving")
    topic:                Optional[str] = Field(None, description="Detected math topic (algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory)")
    variables:            List[str]     = Field(default_factory=list, description="All variables present in the problem, e.g. ['x', 'y', 'n']")
    constraints:          List[str]     = Field(default_factory=list, description="Explicit constraints or given conditions, e.g. ['x > 0', 'n is a positive integer']")
    needs_clarification:  Union[bool, str]  = Field(False, description="True only when the problem is genuinely ambiguous or critically incomplete")
    clarification_reason: Optional[str] = Field(None, description="Why clarification is needed — must be set when needs_clarification=True")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data, bool_fields={"needs_clarification"})


class IntentRouterOutput(BaseModel):
    
    topic:           str           = Field(..., description="algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory")
    difficulty:      str           = Field(..., description="easy | medium | hard")
    solver_strategy: str           = Field(..., description="Best high-level strategy to solve this problem in 1-3 sentences")

    intent_type: str = Field(
        "solve",
        description=(
            "The student's intent. One of:\n"
            "  solve          — full solve + verify pipeline (default for all problems)\n"
            "  explain        — student wants a concept explained, no new calculation needed\n"
            "  hint           — student wants a nudge, not a full solution\n"
            "  formula_lookup — student is asking for a formula or theorem statement only\n"
            "  research       — student wants info about recent/general math topics (no problem to solve)\n"
            "  generate       — student wants practice problems, examples, or content generated"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data, bool_fields=set())
    

class VerifierOutput(BaseModel):

    status: str = Field(..., description="correct | incorrect | partially_correct | needs_human")
    verdict: str = Field(..., description=(
        "One concise paragraph covering all three checks: "
        "(1) step-by-step correctness — cite step numbers for any error, "
        "(2) units and domain/range validity, "
        "(3) edge cases checked (division by zero, undefined log, empty set, etc.). "
        "If correct, state what was verified. If wrong, state exactly what failed."),
    )
    suggested_fix: Optional[str] = Field(None, description=(
        "Required when status != correct. "
        "Name the exact step that is wrong and what the solver must do differently."),
    )
    hitl_reason: Optional[str] = Field(None, description="Required when status == needs_human.")
    confidence: float = Field(..., description="Your confidence in this verdict, 0.0–1.0.")

    # enforce suggested_fix is present when status != "correct"
    @model_validator(mode="after")
    def _require_fix_when_wrong(self) -> "VerifierOutput":
        if self.status not in ("correct",) and not self.suggested_fix:
            object.__setattr__(
                self, "suggested_fix",
                "No specific fix provided by verifier — review the working manually."
            )
        if self.status == "needs_human" and not self.hitl_reason:
            object.__setattr__(
                self, "hitl_reason",
                "Verifier marked as needs_human but provided no reason."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data, bool_fields=set())


class SolutionStep(BaseModel):

    step_number:    Union[int, str]            = Field(..., description="Sequential step number starting from 1.")
    heading:        str            = Field(..., description="Name the technique applied — one phrase.")
    working:        str            = Field(..., description="Complete line-by-line algebraic working.")
    result:         str            = Field(..., description="The expression or value this step reduces to.")
    why:            Optional[str]  = Field(None, description="One sentence explaining WHY this step is taken — only when non-obvious.")
    inline_diagram: Optional[str]  = Field(None, description="Optional ASCII/Unicode diagram for this step.")

    
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        # Coerce step_number string to int
        if "step_number" in data and not isinstance(data["step_number"], int):
            try:
                data["step_number"] = int(data["step_number"])
            except (ValueError, TypeError):
                data["step_number"] = 0
        # Coerce "None" strings to actual None
        for field in ("why", "inline_diagram"):
            if data.get(field) in ("None", "none", "null", ""):
                data[field] = None
        return data


class ExplainerOutput(BaseModel):

    approach_summary:  str        = Field(..., description="2-3 sentence conceptual overview of the method and why it is best.")
    steps:             List[SolutionStep] = Field(..., description="Ordered steps — no skipped algebra.")
    final_answer:      str        = Field(..., description="Boxed final answer in exact form.")
    key_formulae:      List[str]  = Field(default_factory=list)
    key_concepts:      List[str]  = Field(default_factory=list)
    common_mistakes:   List[str]  = Field(default_factory=list)
    difficulty_rating: str        = Field(..., description="easy | medium | hard")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        data = _coerce_bools(data, bool_fields={"needs_diagram"})
        # Coerce string fields — Groq sometimes returns 0/False for zero answers
        for str_field in ("final_answer", "approach_summary", "difficulty_rating"):
            if str_field in data and data[str_field] is not None:
                val = data[str_field]
                if isinstance(val, bool):
                    data[str_field] = "0" if not val else "1"
                elif not isinstance(val, str):
                    data[str_field] = str(val)
        return data


class GuardrailOutput(BaseModel):

    passed:       Union[bool, str]   = Field(..., description="True if input is safe and on-topic.")
    topic:        Optional[str]  = Field(None)
    block_reason: Optional[str]  = Field(None, description="off_topic | prompt_injection | pii | harmful_content")
    message:      Optional[str]  = Field(None, description="Student-facing explanation when passed=False.")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data, bool_fields={"passed"})


class SafetyOutput(BaseModel):
    
    passed:         Union[bool, str]  = Field(..., description="True if the output is safe to show the student.")
    violation_type: Optional[str] = Field(None, description="harmful_content | policy_violation | hallucinated_pii")
    reason:         Optional[str] = Field(None)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data, bool_fields={"passed"})