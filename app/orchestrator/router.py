"""LangChain routing.

The router is a single ``Runnable[str, RouteDecision]`` composed from two
stages:

1. **Heuristic fast-path** — a pure function that short-circuits obviously
   mathematical queries (an extractable arithmetic expression, or strong math
   vocabulary).  Zero network cost, sub-millisecond.
2. **Semantic classifier** — an LLM constrained with structured output that
   decides ambiguous queries.  It is wrapped in ``asyncio.wait_for`` so a slow
   model can never stall the request; on timeout/error it degrades to the
   ``general`` branch instead of failing the request.

Because both stages are LangChain Runnables, the whole route is observable
with LangSmith callbacks (``run_name`` is set on every stage) and testable in
isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda

from app.orchestrator.math_tool import extract_expression
from app.schemas import Route, RouteDecision
from app.security import safe_error_message

logger = logging.getLogger(__name__)

# Vocabulary that is unambiguous enough to bypass the semantic classifier.
_STRONG_MATH_WORDS = frozenset(
    {
        "math",
        "maths",
        "mathematics",
        "arithmetic",
        "algebra",
        "equation",
        "equations",
        "calculus",
        "integral",
        "integrals",
        "derivative",
        "derivatives",
        "geometry",
        "trigonometry",
        "factorial",
        "polynomial",
    }
)

# Verbs that only mean "math" when a number is also present
# ("calculate the median" vs "calculate 48 * 12").
_WEAK_MATH_WORDS = frozenset({"calculate", "compute", "evaluate", "solve", "workout"})

_DIGITS_RE = re.compile(r"\d")

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict query router for an AI gateway.\n"
                "Classify the user query as exactly one of:\n"
                "- 'math': the query is fundamentally a computation or "
                "mathematics problem (arithmetic, algebra, calculus, unit "
                "conversion, percentages, numeric reasoning).\n"
                "- 'general': everything else (chat, knowledge, code, writing, "
                "summarisation, opinion, instructions).\n"
                "Respond with a single JSON object and nothing else, matching:\n"
                "{format_instructions}"
            ),
        ),
        ("human", "{query}"),
    ]
)


def _has_digits(text: str) -> bool:
    return _DIGITS_RE.search(text) is not None


def _has_math_vocabulary(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & _STRONG_MATH_WORDS)


def _has_weak_math_vocabulary(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & _WEAK_MATH_WORDS)


def heuristic_route(query: str, *, classifier_available: bool) -> RouteDecision | None:
    """Stage 1: deterministic fast path. ``None`` means "ask the LLM"."""
    if extract_expression(query) is not None:
        return RouteDecision(
            route=Route.MATH,
            reason="arithmetic expression detected",
            source="heuristic",
        )
    if _has_math_vocabulary(query):
        return RouteDecision(
            route=Route.MATH, reason="math vocabulary detected", source="heuristic"
        )
    if _has_weak_math_vocabulary(query) and _has_digits(query):
        return RouteDecision(
            route=Route.MATH,
            reason="computation verb with numeric input",
            source="heuristic",
        )
    if not classifier_available:
        return RouteDecision(
            route=Route.GENERAL,
            reason="no semantic classifier configured",
            source="fallback",
        )
    return None


def build_classifier(llm: Any) -> Runnable[str, RouteDecision]:
    """Build the stage-2 semantic classifier.

    Uses the provider's native structured-output mode when available and
    falls back to a prompt + parser chain otherwise, so the same router works
    with any LangChain chat model (including test doubles).
    """
    try:
        structured = llm.with_structured_output(RouteDecision)
    except NotImplementedError:
        parser = PydanticOutputParser(pydantic_object=RouteDecision)
        chain = (
            CLASSIFIER_PROMPT.partial(
                format_instructions=parser.get_format_instructions()
            )
            | llm
            | parser
        )
    else:
        chain = (
            CLASSIFIER_PROMPT.partial(
                format_instructions=json.dumps(RouteDecision.model_json_schema())
            )
            | structured
        )

    chain = chain.with_config(run_name="semantic_classifier")

    def _to_decision(output: Any) -> RouteDecision:
        decision = (
            output
            if isinstance(output, RouteDecision)
            else RouteDecision.model_validate(output)
        )
        decision.source = "llm"
        return decision

    return (
        RunnableLambda(lambda query: {"query": query})
        | chain
        | RunnableLambda(_to_decision)
    )


def build_router(
    *, classifier: Runnable[str, RouteDecision] | None, timeout_s: float
) -> Runnable[str, RouteDecision]:
    """Compose both routing stages into one async-capable Runnable."""

    def _pre(query: str) -> dict[str, Any]:
        return {
            "query": query,
            "decision": heuristic_route(
                query, classifier_available=classifier is not None
            ),
        }

    async def _classify(payload: dict[str, Any]) -> RouteDecision:
        query = payload["query"]
        assert classifier is not None, "classifier must exist when stage-1 abstains"
        try:
            decision = await asyncio.wait_for(classifier.ainvoke(query), timeout_s)
        except TimeoutError:
            logger.warning(
                "router: semantic classifier timed out after %.2fs", timeout_s
            )
            return RouteDecision(
                route=Route.GENERAL,
                reason=f"classifier timed out after {timeout_s:.1f}s",
                source="fallback",
            )
        except Exception as exc:  # noqa: BLE001 - routing must never fail a request
            logger.warning("router: classifier degraded: %s", safe_error_message(exc))
            return RouteDecision(
                route=Route.GENERAL,
                reason="classifier unavailable",
                source="fallback",
            )
        if isinstance(decision, dict):
            decision = RouteDecision.model_validate(decision)
        decision.source = "llm"
        logger.info(
            "router: classified query as %s (%s)", decision.route, decision.reason
        )
        return decision

    return (
        RunnableLambda(_pre).with_config(run_name="route_heuristic")
        | RunnableBranch(
            (
                lambda payload: payload["decision"] is not None,
                RunnableLambda(lambda payload: payload["decision"]),
            ),
            RunnableLambda(_classify).with_config(run_name="route_classify"),
        ).with_config(run_name="route_branch")
    ).with_config(run_name="query_router")
