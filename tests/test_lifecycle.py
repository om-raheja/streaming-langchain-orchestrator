"""Lifecycle tests: cancellation, offline degradation, config."""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeChatModel, offline_settings
from helpers import make_client, parse_sse

from app.orchestrator.pipeline import (
    OFFLINE_GENERAL_TEXT,
    StreamStallError,
    iter_text_chunks,
)


async def test_stream_cancels_cleanly_when_client_disconnects(make_app):
    _, orchestrator = make_app(
        general_llm=FakeChatModel(text="one two three four five", delay=0.01)
    )
    stream = orchestrator.stream("tell me about streaming", request_id="req-1")

    first = await stream.__anext__()
    assert first.event == "meta"
    await stream.__anext__()  # route
    await stream.aclose()

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_offline_deployment_still_streams(make_app):
    app, _ = make_app()  # no LLM configured anywhere
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "explain quantum computing"}
        ) as response,
    ):
        events = await parse_sse(response)

    names = [e["event"] for e in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert events[1]["data"]["source"] == "fallback"
    body = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
    assert body == OFFLINE_GENERAL_TEXT


async def test_offline_math_with_no_expression_explains_itself(make_app):
    app, _ = make_app()
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "explain how to find a derivative"}
        ) as response,
    ):
        events = await parse_sse(response)

    assert events[1]["data"]["route"] == "math"
    tool = next(e for e in events if e["event"] == "tool")["data"]
    assert tool["ok"] is False
    assert tool["error"] == "no_arithmetic_expression"
    assert event_names_ends_with_done(events)


def event_names_ends_with_done(events) -> bool:
    return events[-1]["event"] == "done"


async def test_stall_timeout_surfaces_as_error():
    settings = offline_settings(stall_timeout_s=0.05)

    async def hanging():
        await asyncio.sleep(10)
        yield "never"

    from app.orchestrator.pipeline import Orchestrator
    from app.orchestrator.router import build_router

    orchestrator = Orchestrator(
        router=build_router(classifier=None, timeout_s=0.1), settings=settings
    )
    with pytest.raises(StreamStallError):
        async for _ in orchestrator._timed_stream(hanging()):
            pass


async def test_iter_text_chunks_is_incremental():
    seen = [
        chunk async for chunk in iter_text_chunks("abcdefghijklmnopqrstuvwxyz", size=10)
    ]
    assert seen == ["abcdefghij", "klmnopqrst", "uvwxyz"]


async def test_router_failure_degrades_to_general_decision():
    """Even a broken router must produce a usable decision, never an exception."""
    from langchain_core.runnables import RunnableLambda

    from app.orchestrator.pipeline import Orchestrator

    async def explode(query: str) -> None:
        raise RuntimeError("router exploded")

    settings = offline_settings()
    orchestrator = Orchestrator(
        router=RunnableLambda(explode),
        settings=settings,  # type: ignore[arg-type]
    )
    decision = await orchestrator.route("hello")
    assert decision.route.value == "general"
    assert decision.source == "fallback"


async def test_narrator_stall_degrades_to_calculator_answer(make_app):
    """A hung narrator must not lose an answer the tool already produced."""
    from langchain_core.runnables import RunnableLambda

    app, orchestrator = make_app(settings=offline_settings(stall_timeout_s=0.05))

    async def hang(payload: dict) -> str:
        await asyncio.sleep(5)
        return "never"

    orchestrator._math_narrator = RunnableLambda(hang)  # type: ignore[assignment]

    async with (
        make_client(app) as client,
        client.stream("POST", "/ask", json={"query": "what is 12 * 12"}) as response,
    ):
        events = await parse_sse(response)

    assert events[-1]["event"] == "done"
    body = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
    assert "144" in body


async def test_aclose_is_safe(make_app):
    _, orchestrator = make_app(general_llm=FakeChatModel())
    await orchestrator.aclose()
