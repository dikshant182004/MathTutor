from agents import List, Optional
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
    needs_visualization: bool          = Field(False, description="Use JSON boolean true or false — NOT a string. True if a Manim animation would meaningfully aid student understanding")
    visualization_hint:  Optional[str] = Field(None, description="What the animation should show, if needs_visualization is true")
    requires_calculator: bool          = Field(False, description="Use JSON boolean true or false — NOT a string. True if non-trivial numerical computation is needed")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)

class SolverOutput(BaseModel):
    """
    Output of the Solver Agent (ReAct loop — may be built across multiple turns).
    Captures the complete solution after all tool calls are resolved.
    """
    solution:         str       = Field(...,  description="Complete solution with every working step shown")
    final_answer:     str       = Field(...,  description="Concise final answer only (number, expression, or statement)")
    steps:            List[str] = Field(default_factory=list, description="Ordered list of solution steps in plain language")
    formulas_used:    List[str] = Field(default_factory=list, description="Mathematical formulas or theorems applied")
    rag_context_used: bool      = Field(False, description="Use JSON boolean true or false — NOT a string. True if rag_tool results were used")
    calculator_used:  bool      = Field(False, description="Use JSON boolean true or false — NOT a string. True if python_calculator_tool was called")
    confidence_score: float     = Field(1.0,  description="Solver's confidence in the answer, 0.0 to 1.0")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)

class VerifierOutput(BaseModel):
    """
    Output of the Verifier Agent.
    Critically assesses the solver's solution and decides next action.
    """
    status:            str           = Field(...,  description="correct | incorrect | partially_correct | needs_human")
    is_correct:        bool          = Field(...,  description="Use JSON boolean true or false — NOT a string. True only when the solution is fully correct")
    correctness_notes: str           = Field(...,  description="Detailed step-by-step assessment of correctness")
    unit_domain_check: str           = Field(...,  description="Assessment of units, domain restrictions, and range validity")
    edge_case_notes:   str           = Field(...,  description="Edge cases considered and whether they were handled correctly")
    suggested_fix:     Optional[str] = Field(None, description="Concrete, actionable fix for the solver — required when status != correct")
    hitl_reason:       Optional[str] = Field(None, description="Why human review is needed — required when status == needs_human")
    confidence_score:  float         = Field(1.0,  description="Verifier's confidence in its own assessment, 0.0 to 1.0")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: dict) -> dict:
        return _coerce_bools(data)

class ExplainerOutput(BaseModel):
    """
    Output of the Explainer Agent.
    Student-friendly explanation with optional Manim visualization code.
    """
    explanation:             str           = Field(...,  description="Full student-friendly explanation of the solution")
    step_by_step:            List[str]     = Field(default_factory=list, description="Ordered explanation steps written in plain language a student can follow")
    key_concepts:            List[str]     = Field(default_factory=list, description="Core mathematical concepts the student must understand")
    common_mistakes:         List[str]     = Field(default_factory=list, description="Typical errors students make on this type of problem")
    manim_scene_code:        Optional[str] = Field(None, description="Complete, self-contained Manim Community Edition Python scene — None if visualization not needed")
    manim_scene_description: Optional[str] = Field(None, description="Plain-English description of what the animation shows")
    difficulty_rating:       str           = Field(...,  description="easy | medium | hard — from the student's perspective")
