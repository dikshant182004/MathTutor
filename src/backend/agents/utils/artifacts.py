from backend.agents import List, Optional
from pydantic import BaseModel, Field, model_validator
from utils.helper import _coerce_bools

class ParserOutput(BaseModel):
    """
    Output of the Parser Agent.
    Converts raw text / OCR / ASR input into a clean, structured problem.
    """

    problem_text:         str           = Field(...,  description="Cleaned, complete problem statement ready for solving")
    topic:                Optional[str] = Field(None, description="Detected math topic (algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory)")
    variables:            List[str]     = Field(default_factory=list, description="All variables present in the problem, e.g. ['x', 'y', 'n']")
    constraints:          List[str]     = Field(default_factory=list, description="Explicit constraints or given conditions, e.g. ['x > 0', 'n is a positive integer']")
    needs_clarification:  bool          = Field(False, description="True only when the problem is genuinely ambiguous or critically incomplete")
    clarification_reason: Optional[str] = Field(None, description="Why clarification is needed — must be set when needs_clarification=True")
    
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)


class IntentRouterOutput(BaseModel):
    """
    Output of the Intent Router Agent.
    Classifies the problem and plans the solving strategy.
    """

    topic:               str           = Field(...,  description="algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory")
    difficulty:          str           = Field(...,  description="easy | medium | hard")
    solver_strategy:     str           = Field(...,  description="Best high-level strategy to solve this problem in 1-3 sentences")
 
 
class VerifierOutput(BaseModel):
    """
    Focused verdict from the verifier.
    Three checks: correctness, units/domain, edge cases.
    Routes back to solver (retry), forward to explainer (correct),
    or to HITL (genuinely uncertain).
    """

    status: str = Field(..., description="correct | incorrect | partially_correct | needs_human",)
    verdict: str = Field(..., description=(
            "One concise paragraph covering all three checks: "
            "(1) step-by-step correctness — cite step numbers for any error, "
            "(2) units and domain/range validity, "
            "(3) edge cases checked (division by zero, undefined log, empty set, etc.). "
            "If correct, state what was verified. If wrong, state exactly what failed."),
    )
    suggested_fix: Optional[str] = Field(None, description=(
            "Required when status != correct. "
            "Name the exact step that is wrong and what the solver must do differently. "
            "Be specific — e.g. 'Step 3: the substitution u = sin x gives du = cos x dx, "
            "not dx = du' — not 'check your substitution'."),
    )
    hitl_reason: Optional[str] = Field(None, description="Required when status == needs_human. Why you cannot determine correctness.",)
    confidence: float = Field(..., description="Your confidence in this verdict, 0.0–1.0.",)
 
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)
 

class SolutionStep(BaseModel):
    """
    One step of the JEE answer sheet — mirrors how a teacher writes on a whiteboard.
 
    Notation contract (enforced at schema level):
    - Use EXACT variable names from the problem — never rename.
    - Integrals   : ∫ f(x) dx  — always include the differential.
    - Fractions   : (numerator)/(denominator) — fully bracketed.
    - Powers      : x^2 or x² — match the problem's form throughout.
    - Roots       : √(expr) — never mix with sqrt() in the same solution.
    - Limits      : lim_{x → a}  with arrow, not ->
    - Summations  : Σ_{k=1}^{n} with explicit bounds.
    - Exact form  : fractions/surds/π/e/ln — decimals only when asked.
    - Vectors     : →a, magnitude |→a|.
    """

    step_number:  int = Field(..., description="Sequential step number starting from 1.")
    heading: str = Field(..., description=(
            "Name the technique applied — e.g. 'Let u = x² + 1 (substitution)', "
            "'Apply integration by parts', 'Use Bayes theorem', "
            "'Row-reduce the augmented matrix'. One phrase."),
    )
    working: str = Field(..., description=(
            "Complete line-by-line algebraic working. "
            "Every manipulation on its own line ending in = the new expression. "
            "Follow the notation contract above. "
            "Write exactly as it appears on a JEE answer sheet."),
    )
    result: str = Field(..., description=(
            "The expression or value this step reduces to — exactly as it would be "
            "circled on the answer sheet. Same notation as working. Math only, no prose."),
    )
    why: Optional[str] = Field(None, description=(
            "One sentence explaining WHY this step is taken — only when non-obvious. "
            "e.g. 'We multiply by the conjugate to rationalise the denominator.' "
            "Skip for standard mechanical steps like expanding brackets."),
    )
    inline_diagram: Optional[str] = Field(None, description=(
            "Optional ASCII/Unicode diagram for this step — use ONLY when a visual "
            "genuinely aids understanding of THIS step. Examples: a number line for "
            "inequality solutions, a right triangle for trig ratios, a Venn diagram "
            "for probability, a unit circle for trig values. Keep under 10 lines. "
            "Leave None for purely algebraic steps."),
    )
 
 
class ExplainerOutput(BaseModel):
    """
    Complete JEE-style explanation produced after the verifier confirms correctness.
    Structured like a model answer from a coaching institute.
    """

    approach_summary: str = Field(..., description=(
            "2-3 sentence conceptual overview: what method is used and WHY it is "
            "the best approach for this problem type. Written for a student who got "
            "the answer wrong — help them understand the thinking, not just the steps."),
    )
    steps: List[SolutionStep] = Field(..., description=(
            "Ordered steps — each builds directly on the previous result. "
            "No skipped algebra. No jumps. Every substitution shown explicitly."),
    )
    final_answer: str = Field(..., description=(
            "Boxed final answer in exact form — same notation as working. "
            "Include unit inline if applicable e.g. '(π/4) sq. units'. "
            "If the answer is a set, write it as a set."),
    )
    key_formulae: List[str] = Field(default_factory=list, description=(
            "Every formula or identity applied in the solution, written in proper "
            "notation e.g. ['∫u dv = uv − ∫v du', 'sin²θ + cos²θ = 1', "
            "'P(A|B) = P(A∩B)/P(B)']. Student can use this as a quick reference."),
    )
    key_concepts: List[str] = Field(default_factory=list, description=(
            "Core mathematical concepts the student must understand to solve this "
            "problem type — not just what was used, but the underlying idea. "
            "e.g. ['Substitution reduces a composite integrand to a standard form', "
            "'The determinant gives the scaling factor of the linear transformation']."),
    )
    common_mistakes:  List[str] = Field( default_factory=list,
        description=(
            "2-4 specific mistakes students make on THIS problem type — with the "
            "correct alternative. e.g. 'Forgetting to change limits of integration "
            "after substitution — always convert x-limits to u-limits immediately'."),
    )
    needs_diagram: bool = Field(False, description=(
            "True when the problem involves geometry, vectors, coordinate geometry, "
            "trigonometric graphs, probability trees, or any concept that is genuinely "
            "easier to understand with a picture. False for pure algebra/number theory."),
    )
    manim_hint: Optional[str]  = Field(None, description=(
            "Required when needs_diagram=True. "
            "Describe EXACTLY what the Manim animation should show — one sentence per "
            "visual element. e.g. 'Draw the unit circle, mark angle θ in standard "
            "position, animate the point (cos θ, sin θ) as θ increases from 0 to 2π, "
            "trace the sin and cos curves simultaneously below the circle'."),
    )
    difficulty_rating: str = Field(..., description="easy | medium | hard — from the student's perspective at JEE level.",)
 
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)
 
 
class GuardrailOutput(BaseModel):
    """
    Output of the guardrail agent.
    Checks raw user input before any processing.
    route_after_guardrail reads: passed, block_reason.
    """
    
    passed: bool = Field(..., description="True if input is safe and on-topic. False if it must be blocked.",)
    topic: Optional[str] = Field(None, description="Detected math topic when passed=True e.g. 'calculus', 'probability'.",)
    block_reason: Optional[str] = Field(None, description=(
            "Required when passed=False. "
            "One of: off_topic | prompt_injection | pii | harmful_content."),
    )
    message: Optional[str] = Field(None,
        description=(
            "Student-facing explanation when passed=False — polite, one sentence. "
            "e.g. 'I can only help with JEE mathematics problems.'"
        ),
    )
 
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)
 
  
class SafetyOutput(BaseModel):
    """
    Output of the safety agent.
    Checks solver output before it reaches the student.
    route_after_safety reads: passed, violation_type.
    """

    passed: bool = Field(..., description="True if the output is safe to show the student.",)
    violation_type: Optional[str] = Field(None, description=(
            "Required when passed=False. "
            "One of: harmful_content | policy_violation | hallucinated_pii."),
    )
    reason: Optional[str] = Field(None, description="Internal note on why output was blocked — not shown to student.",)
 
    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)
 