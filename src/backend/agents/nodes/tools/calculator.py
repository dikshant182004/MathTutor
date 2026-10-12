"""Safe bounded calculator evaluator; kept separate for isolated security tests."""
from __future__ import annotations
import sympy as sp
_CALCULATOR_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float,
    "Rational": sp.Rational, "Add": sp.Add, "Mul": sp.Mul, "Pow": sp.Pow,
    "Tuple": sp.Tuple, "Matrix": sp.Matrix, "ImmutableDenseMatrix": sp.ImmutableDenseMatrix,
    "factorial": sp.factorial, "binomial": sp.binomial, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "log": sp.log, "exp": sp.exp, "Abs": sp.Abs,
    "Min": sp.Min, "Max": sp.Max, "pi": sp.pi, "E": sp.E,
}
_CALCULATOR_NAMES = set(_CALCULATOR_GLOBALS) - {"__builtins__", "Symbol", "Integer", "Float", "Rational", "Add", "Mul", "Pow", "Tuple", "ImmutableDenseMatrix"}
_CALCULATOR_SYMBOLS = {"x", "y", "z", "n", "a", "b", "c", "t", "r", "theta"}

def _calculator_worker(expression: str, result_queue) -> None:
    """Evaluate in a child process so expensive symbolic work can be terminated."""
    try:
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, implicit_multiplication_application
        local = {name: value for name, value in _CALCULATOR_GLOBALS.items() if name != "__builtins__"}
        for name in _CALCULATOR_SYMBOLS:
            local[name] = sp.Symbol(name)
        parsed = parse_expr(expression, local_dict=local, global_dict=_CALCULATOR_GLOBALS,
            transformations=standard_transformations + (convert_xor, implicit_multiplication_application), evaluate=True)
        result_queue.put(("ok", str(parsed) if isinstance(parsed, sp.MatrixBase) else str(sp.N(parsed))))
    except BaseException as exc:
        result_queue.put(("error", f"Invalid or unsupported expression ({type(exc).__name__})."))

def _safe_calculator_expression(expression: str, timeout_seconds: float = 5.0) -> str:
    """Validate a small allowlisted expression language and enforce a hard timeout."""
    import multiprocessing
    import re
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    expression = expression.strip()
    if len(expression) > 500:
        raise ValueError("Expression exceeds the 500-character limit.")
    if "__" in expression or re.search(r"[{};\\`]", expression):
        raise ValueError("Expression contains disallowed syntax.")
    if "." in expression and not re.search(r"\)\.(?:det|inv|eigenvals)\(\)\s*$", expression):
        raise ValueError("Attribute access is not allowed.")
    names = set(re.findall(r"[A-Za-z_]\w*", expression))
    allowed = _CALCULATOR_NAMES | _CALCULATOR_SYMBOLS | {"Matrix", "det", "inv", "eigenvals"}
    if names - allowed:
        raise ValueError("Expression contains an unknown function or name.")
    if re.search(r"\.(?!det\(\)|inv\(\)|eigenvals\(\))", expression):
        raise ValueError("Attribute access is not allowed.")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_calculator_worker, args=(expression, result_queue), daemon=True)
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise ValueError("Calculation exceeded the 5-second time limit.")
    if result_queue.empty():
        raise ValueError("Calculator process failed without a result.")
    status, value = result_queue.get()
    result_queue.close()
    if status != "ok":
        raise ValueError(value)
    return value

