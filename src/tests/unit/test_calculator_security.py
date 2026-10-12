from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
SRC = Path(__file__).resolve().parents[2]
PATH = SRC / "backend/agents/nodes/tools/calculator.py"
spec = importlib.util.spec_from_file_location("calculator_under_test", PATH)
calculator = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(calculator)
_safe_calculator_expression = calculator._safe_calculator_expression
@pytest.mark.parametrize(("expression","expected"), [("2 + 2","4.00000000000000"),("factorial(5)","120.000000000000"),("x + 1","x + 1")])
def test_calculator_accepts_whitelisted_expressions(expression, expected):
    assert _safe_calculator_expression(expression) == expected
@pytest.mark.parametrize("expression", ["__import__('os').system('echo pwned')", "x.__class__", "open('/tmp/pwned','w')", "eval('1+1')"])
def test_calculator_rejects_code_execution_and_attribute_access(expression):
    with pytest.raises(ValueError): _safe_calculator_expression(expression)
def test_calculator_rejects_oversized_expression():
    with pytest.raises(ValueError): _safe_calculator_expression("1+" + "1+" * 300)
