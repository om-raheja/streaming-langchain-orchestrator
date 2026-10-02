"""End-to-end tests for the POST /ask SSE endpoint."""

from __future__ import annotations

import pytest
from conftest import FakeChatModel
from helpers import event_names, joined_tokens, make_client, parse_sse


async def test_general_route_streams_chunked_sse(make_app):
    app, _ = make_app(general_llm=FakeChatModel(text="Paris is the capital of France."))
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "What is the capital of France?"}
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"].startswith("no-cache")
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers.get("x-request-id")
        events = await parse_sse(response)

    names = event_names(events)
    assert names[0] == "meta"
    assert names[1] == "route"
    assert names[-1] == "done"
    assert "error" not in names

    route = events[1]["data"]
    assert route["route"] == "general"
    assert route["source"] in {"heuristic", "llm", "fallback"}

    tokens = [e for e in events if e["event"] == "token"]
    # Chunked: several frames, not one buffered blob.
    assert len(tokens) > 1
    assert joined_tokens(events).strip() == "Paris is the capital of France."

    done = events[-1]["data"]
    assert done["chunks"] == len(tokens)
    assert done["first_token_ms"] is not None
    assert done["duration_ms"] >= 0

    # SSE ids are strictly increasing so clients can resume.
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


async def test_math_route_runs_calculator_tool(make_app):
    app, _ = make_app()
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "what is 12 * (4 + 3)"}
        ) as response,
    ):
        events = await parse_sse(response)

    assert events[1]["data"]["route"] == "math"
    assert events[1]["data"]["source"] == "heuristic"

    tool = next(e for e in events if e["event"] == "tool")["data"]
    assert tool["name"] == "calculator"
    assert tool["ok"] is True
    assert tool["result"] == "84"
    assert "84" in joined_tokens(events)
    assert event_names(events)[-1] == "done"


async def test_math_route_streams_narrator_when_llm_present(make_app):
    app, _ = make_app(math_llm=FakeChatModel(text="The answer is 84."))
    async with (
        make_client(app) as client,
        client.stream("POST", "/ask", json={"query": "calculate 12 * 7"}) as response,
    ):
        events = await parse_sse(response)

    tool = next(e for e in events if e["event"] == "tool")["data"]
    assert tool["result"] == "84"
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) > 1
    assert joined_tokens(events).strip() == "The answer is 84."


async def test_semantic_classifier_routes_chit_chat(make_app):
    app, _ = make_app(
        route_llm=FakeChatModel(
            text='{"route": "general", "reason": "not a computation"}'
        ),
        general_llm=FakeChatModel(text="Here is a joke."),
    )
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "tell me something interesting"}
        ) as response,
    ):
        events = await parse_sse(response)

    assert events[1]["data"]["route"] == "general"
    assert events[1]["data"]["source"] == "llm"
    assert joined_tokens(events).strip() == "Here is a joke."


async def test_model_failure_emits_redacted_error_frame(make_app):
    app, _ = make_app(general_llm=FakeChatModel(fail=True))
    async with (
        make_client(app) as client,
        client.stream("POST", "/ask", json={"query": "hello there"}) as response,
    ):
        events = await parse_sse(response)

    names = event_names(events)
    assert "error" in names
    assert "done" not in names
    error = next(e for e in events if e["event"] == "error")["data"]
    assert error["code"] == "RuntimeError"
    assert "Traceback" not in error["message"]
    assert "sk-" not in error["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "hi", "unexpected": True},
    ],
)
async def test_invalid_payloads_are_rejected(make_app, payload):
    app, _ = make_app()
    async with make_client(app) as client:
        response = await client.post("/ask", json=payload)
    assert response.status_code == 422


async def test_oversized_query_is_rejected(make_app):
    app, _ = make_app()  # offline settings cap queries at 500 chars
    async with make_client(app) as client:
        response = await client.post("/ask", json={"query": "a" * 501})
    assert response.status_code == 422


async def test_health_never_exposes_credentials(make_app):
    app, _ = make_app()
    async with make_client(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "openai"
    assert body["providers"] == {"active": "openai", "openai": False, "gemini": False}
    assert "sk-" not in response.text
    assert "AIza" not in response.text


async def test_concurrent_streams_are_isolated(make_app):
    import asyncio

    app, _ = make_app(general_llm=FakeChatModel(text="stream one two three four five"))
    async with make_client(app) as client:
        responses = await asyncio.gather(
            *[client.post("/ask", json={"query": f"question {i}"}) for i in range(8)]
        )
    for index, response in enumerate(responses):
        assert response.status_code == 200
        assert f"question {index}" not in response.text
        assert response.text.rstrip().endswith("}")
        assert '"event"' not in response.text  # SSE frames are not JSON envelopes
        assert "event: done" in response.text
        assert "event: error" not in response.text
