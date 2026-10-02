import ast
import operator
import os
import re
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    raise ValueError("Unsupported math expression")


def _extract_expression(query: str) -> str:
    parts = [p.strip() for p in re.findall(r"[\d\s\+\-\*\/\(\)\.\%\^]+", query) if p.strip()]
    return max(parts, key=len, default="")


def _math_handler(payload: dict) -> str:
    query = payload["query"]
    expression = _extract_expression(query).replace("^", "**")
    if not expression:
        return "Math result: please provide an arithmetic expression."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception:
        return "Math result: unable to parse the math expression."

    normalized = int(result) if result.is_integer() else result
    return f"Math result: {normalized}"


def _build_general_chain():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return RunnableLambda(
            lambda payload: "General response unavailable: OPENAI_API_KEY is not configured."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{query}")]
    )
    return prompt | llm | StrOutputParser()


general_chain = _build_general_chain()
math_chain = RunnableLambda(_math_handler)

router_chain = RunnableBranch(
    (lambda payload: "math" in payload["query"].lower(), math_chain),
    (lambda payload: "general" in payload["query"].lower(), general_chain),
    general_chain,
)


async def stream_query(query: str) -> AsyncIterator[str]:
    async for chunk in router_chain.astream({"query": query}):
        text = str(chunk)
        if text:
            yield text
