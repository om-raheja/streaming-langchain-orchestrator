"""A sandboxed calculator "tool" plus the natural-language -> expression
extractor that feeds it.

Design notes
------------
* ``evaluate`` never uses ``eval``/``exec``.  It walks an AST and rejects any
  node that is not on a small allow-list (arithmetic operators and a handful
  of :mod:`math` functions), so queries such as ``__import__("os")`` or
  ``open("/etc/passwd")`` fail closed.
* Resource guards (input length, node count, exponent magnitude, output
  magnitude) keep a hostile query from turning into a CPU/memory bomb.
* ``extract_expression`` handles both literal expressions (``2 + 2``) and a
  small set of natural-language forms (``15% of 240``, ``square root of 9``).
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable

MAX_EXPRESSION_CHARS = 120
MAX_AST_NODES = 80
MAX_EXPONENT = 64
MAX_RESULT = 1e15


class MathToolError(ValueError):
    """Raised when an expression is invalid, unsafe or unrepresentable."""


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
}

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    *_BIN_OPS,
    *_UNARY_OPS,
)


def _check_node(node: ast.AST) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise MathToolError(f"disallowed syntax: {type(node).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise MathToolError("unknown function")
        if node.keywords:
            raise MathToolError("keyword arguments are not allowed")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise MathToolError("only numeric literals are allowed")
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        _check_power(node.left, node.right)


def _check_power(left: ast.AST, right: ast.AST) -> None:
    if (
        isinstance(right, ast.Constant)
        and isinstance(right.value, (int, float))
        and abs(right.value) > MAX_EXPONENT
    ):
        raise MathToolError("exponent too large")
    if (
        isinstance(left, ast.Constant)
        and isinstance(left.value, (int, float))
        and abs(left.value) > 1e6
    ):
        raise MathToolError("base too large")
    if (
        isinstance(right, ast.Constant)
        and isinstance(right.value, float)
        and not right.value.is_integer()
        and isinstance(left, ast.Constant)
        and left.value < 0
    ):
        raise MathToolError("negative base with fractional exponent")


class _Evaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left, right = self.visit(node.left), self.visit(node.right)
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:  # pragma: no cover - guarded by _check_node
            raise MathToolError("unsupported operator")
        try:
            result = fn(left, right)
        except ZeroDivisionError as exc:
            raise MathToolError("division by zero") from exc
        except (OverflowError, ValueError) as exc:
            raise MathToolError(f"math error: {exc}") from exc
        if isinstance(result, complex):  # pragma: no cover - guarded upstream
            raise MathToolError("complex result")
        if not math.isfinite(result):
            raise MathToolError("result is not finite")
        if abs(result) > MAX_RESULT:
            raise MathToolError("result magnitude too large")
        return float(result)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:  # pragma: no cover
            raise MathToolError("unsupported unary operator")
        return float(fn(operand))

    def visit_Call(self, node: ast.Call) -> float:
        assert isinstance(node.func, ast.Name)
        fn = _FUNCTIONS[node.func.id]
        args = [self.visit(arg) for arg in node.args]
        try:
            result = self._call_function(fn, args)
        except ZeroDivisionError as exc:
            raise MathToolError("division by zero") from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise MathToolError(f"math error: {exc}") from exc
        if not math.isfinite(result):
            raise MathToolError("result is not finite")
        if abs(result) > MAX_RESULT:
            raise MathToolError("result magnitude too large")
        return result

    @staticmethod
    def _call_function(fn: Callable[..., float], args: list[float]) -> float:
        try:
            return float(fn(*args))
        except TypeError:
            # Functions like round(x, ndigits) need an int; retry once with
            # whole floats coerced (round(3.14, 2.0) -> round(3.14, 2)).
            coerced: list[float | int] = []
            changed = False
            for arg in args:
                if isinstance(arg, float) and arg.is_integer():
                    coerced.append(int(arg))
                    changed = True
                else:
                    coerced.append(arg)
            if not changed:
                raise
            return float(fn(*coerced))

    def visit_Name(self, node: ast.Name) -> float:
        raise MathToolError("unknown symbol")  # only function names appear

    def visit_Constant(self, node: ast.Constant) -> float:
        return float(node.value)


def evaluate(expression: str) -> float:
    """Safely evaluate an arithmetic expression and return a float."""
    expression = expression.strip().replace("^", "**")  # calculators use ^
    if not expression:
        raise MathToolError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise MathToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MathToolError("could not parse expression") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise MathToolError("expression too complex")
    for node in nodes:
        _check_node(node)

    return float(_Evaluator().visit(tree))


def format_number(value: float) -> str:
    """Render floats the way a human would say them (no ``2.0`` for ints)."""
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    rounded = float(f"{value:.12g}")
    return str(rounded)


# --------------------------------------------------------------------------
# Natural language -> expression
# --------------------------------------------------------------------------

_PERCENT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_SQRT_RE = re.compile(r"square\s+root\s+of\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)

_WORD_OPS: dict[str, str] = {
    "plus": "+",
    "added to": "+",
    "minus": "-",
    "subtracted from": "-",
    "times": "*",
    "multiplied by": "*",
    "divided by": "/",
    "over": "/",
}
_WORD_OP_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+("
    + "|".join(re.escape(w) for w in _WORD_OPS)
    + r")\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_RAW_CHUNK_RE = re.compile(r"[0-9+\-*/().,%^ ]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_OPERATOR_RE = re.compile(r"[*/+^%]|-(?=\S)")


def _from_words(text: str) -> str | None:
    match = _WORD_OP_RE.search(text)
    if not match:
        return None
    left, word, right = match.group(1), match.group(2).lower(), match.group(3)
    return f"{left} {_WORD_OPS[word]} {right}"


def _from_percent(text: str) -> str | None:
    match = _PERCENT_RE.search(text)
    if not match:
        return None
    return f"({match.group(1)} / 100) * {match.group(2)}"


def _from_sqrt(text: str) -> str | None:
    match = _SQRT_RE.search(text)
    if not match:
        return None
    return f"sqrt({match.group(1)})"


def _from_raw(text: str) -> str | None:
    """Pull the densest run of arithmetic characters that is a real expression."""
    best: str | None = None
    for match in _RAW_CHUNK_RE.finditer(text):
        candidate = match.group(0).strip()
        if len(candidate) < 3:
            continue
        # Must contain >=2 numbers and a binary operator between them.
        if len(_NUMBER_RE.findall(candidate)) < 2:
            continue
        if not _OPERATOR_RE.search(candidate.lstrip("-+. ")):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    if best is None:
        return None
    return best.replace("^", "**").rstrip(".,").strip()


def extract_expression(query: str) -> str | None:
    """Best-effort conversion of a user query into a computable expression."""
    for extractor in (_from_percent, _from_sqrt, _from_words, _from_raw):
        expression = extractor(query)
        if expression:
            return expression
    return None
