"""Router unit tests: heuristic fast path, semantic classifier, degradation."""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeChatModel
from langchain_core.runnables import RunnableLambda

from app.orchestrator.router import (
    build_classifier,
    build_router,
    heuristic_route,
)
from app.schemas import Route

MATH_QUERIES = [
    "what is 12 * (4 + 3)",
    "calculate 48 * 12",
    "15% of 240",
    "square root of 144",
    "12 plus 8",
    "explain the quadratic equation",
    "solve for x: 3x + 2 = 11",
]

GENERAL_QUERIES = [
    "tell me a joke",
    "summarise this paragraph about penguins",
    "write a python function that reads a csv",
    "who won the world cup in 2018",
]


@pytest.mark.parametrize("query", MATH_QUERIES)
def test_heuristic_flags_math(query):
    decision = heuristic_route(query, classifier_available=True)
    assert decision is not None
    assert decision.route is Route.MATH
    assert decision.source == "heuristic"


@pytest.mark.parametrize("query", GENERAL_QUERIES)
def test_heuristic_defers_general_queries_to_classifier(query):
    decision = heuristic_route(query, classifier_available=True)
    assert decision is None


def test_heuristic_falls_back_to_general_without_classifier():
    decision = heuristic_route("tell me a joke", classifier_available=False)
    assert decision is not None
    assert decision.route is Route.GENERAL
    assert decision.source == "fallback"


async def test_router_uses_classifier_for_ambiguous_queries():
    async def classify(query: str):
        await asyncio.sleep(0)
        assert query == "tell me a joke"
        from app.schemas import RouteDecision

        return RouteDecision(route=Route.GENERAL, reason="chit-chat")

    router = build_router(classifier=RunnableLambda(classify), timeout_s=1.0)
    decision = await router.ainvoke("tell me a joke")
    assert decision.route is Route.GENERAL
    assert decision.source == "llm"


async def test_router_short_circuits_math_without_calling_classifier():
    called = False

    async def classify(query: str):
        nonlocal called
        called = True
        from app.schemas import RouteDecision

        return RouteDecision(route=Route.GENERAL, reason="never")

    router = build_router(classifier=RunnableLambda(classify), timeout_s=1.0)
    decision = await router.ainvoke("what is 2 + 2")
    assert decision.route is Route.MATH
    assert decision.source == "heuristic"
    assert called is False


async def test_router_times_out_and_degrades_to_general():
    async def slow(query: str):
        await asyncio.sleep(5)
        from app.schemas import RouteDecision

        return RouteDecision(route=Route.MATH, reason="too late")

    router = build_router(classifier=RunnableLambda(slow), timeout_s=0.05)
    decision = await router.ainvoke("tell me a joke")
    assert decision.route is Route.GENERAL
    assert decision.source == "fallback"
    assert "timed out" in decision.reason


async def test_router_survives_classifier_exception():
    async def broken(query: str):
        raise RuntimeError("sk-test-PLACEHOLDER-not-a-real-key")

    router = build_router(classifier=RunnableLambda(broken), timeout_s=0.5)
    decision = await router.ainvoke("tell me a joke")
    assert decision.route is Route.GENERAL
    assert decision.source == "fallback"
    assert "sk-" not in decision.reason


def test_build_classifier_falls_back_to_prompt_parser_for_plain_models():
    llm = FakeChatModel(text='{"route": "math", "reason": "arithmetic"}')
    classifier = build_classifier(llm)
    assert classifier is not None


async def test_build_classifier_end_to_end_with_plain_model():
    llm = FakeChatModel(text='{"route": "math", "reason": "arithmetic"}')
    classifier = build_classifier(llm)
    router = build_router(classifier=classifier, timeout_s=1.0)
    decision = await router.ainvoke("tell me a joke")
    assert decision.route is Route.MATH
    assert decision.source == "llm"
