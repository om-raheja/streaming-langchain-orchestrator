"""Sandboxed calculator tool: correctness, extraction and abuse resistance."""

from __future__ import annotations

import pytest

from app.orchestrator.math_tool import (
    MathToolError,
    evaluate,
    extract_expression,
    format_number,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("(12.5 * 4) + 2**10", 1074),
        ("2 ^ 10", 1024),
        ("10 / 4", 2.5),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("-3 + 10", 7),
        ("sqrt(144)", 12),
        ("max(3, 9, 4)", 9),
        ("abs(-42)", 42),
        ("round(3.14159, 2)", 3.14),
        ("2 ** (1 / 2)", 2**0.5),
    ],
)
def test_evaluate_expressions(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "[x for x in range(10)]",
        "(lambda: 1)()",
        "().__class__",
        "pow(2, 2, 2, 2)",
        "x",
        "'abc' + 'def'",
        "",
    ],
)
def test_evaluate_rejects_unsafe_input(expression):
    with pytest.raises(MathToolError):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 999",  # exponent bomb
        "1 / 0",  # division by zero
        "1e308 * 1e308",  # overflow
        "1 + " * 40 + "1",  # too many nodes
        "1" * 300,  # too long
    ],
)
def test_evaluate_enforces_resource_limits(expression):
    with pytest.raises(MathToolError):
        evaluate(expression)


@pytest.mark.parametrize(
    ("query", "expected_value"),
    [
        ("what is 15% of 240", 36),
        ("what is 50 percent of 80", 40),
        ("square root of 144", 12),
        ("12 plus 8", 20),
        ("12 times 4", 48),
        ("100 divided by 8", 12.5),
        ("what is 2 + 2", 4),
        ("(12.5 * 4) + 2^10", 1074),
        ("3 * 4 - 6 / 2", 9),
    ],
)
def test_extract_and_evaluate(query, expected_value):
    expression = extract_expression(query)
    assert expression is not None, query
    assert evaluate(expression) == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "query",
    [
        "hello world",
        "write a poem about the sea",
        "compare nodejs and go",
        "I have 3 apples and 4 oranges",
    ],
)
def test_extract_expression_returns_none_for_prose(query):
    assert extract_expression(query) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4.0, "4"),
        (0.1 + 0.2, "0.3"),
        (1 / 3, "0.333333333333"),
        (-2.0, "-2"),
        (1075.0, "1075"),
    ],
)
def test_format_number(value, expected):
    assert format_number(value) == expected
