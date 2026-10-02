"""Integration: the real ``langchain-openai`` client against an in-process
OpenAI-compatible provider (``scripts/mock_openai.py``).

This proves the full path without any network access or credentials:

    POST /ask -> LangChain router (structured output) -> ChatOpenAI streaming
              -> SSE frames
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import socket
from pathlib import Path

import pytest
import uvicorn
from helpers import joined_tokens, make_client, parse_sse

from app.config import Settings
from app.main import create_app
from app.orchestrator.pipeline import Orchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_PATH = PROJECT_ROOT / "scripts" / "mock_openai.py"


def _load_mock_app():
    spec = importlib.util.spec_from_file_location("mock_openai", MOCK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def mock_provider_url():
    """Run the mock OpenAI provider inside the test's event loop."""
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(_load_mock_app(), host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    for _ in range(250):
        if server.started:
            break
        await asyncio.sleep(0.02)
    else:  # pragma: no cover
        server.should_exit = True
        await task
        pytest.fail("mock provider did not start")
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 3)


@pytest.fixture
def live_orchestrator(mock_provider_url) -> Orchestrator:
    settings = Settings(
        openai_api_key="mock-integration-key",  # dummy value, never a real key
        route_model="mock-router",
        general_model="mock-general",
        math_model="mock-math",
        router_timeout_s=5.0,
        stall_timeout_s=5.0,
        max_query_chars=500,
    )
    # ChatOpenAI reads the provider endpoint from the environment.
    import os

    os.environ["OPENAI_BASE_URL"] = mock_provider_url
    orchestrator = Orchestrator.from_settings(settings)
    yield orchestrator
    os.environ.pop("OPENAI_BASE_URL", None)


async def test_live_general_stream_from_provider(live_orchestrator):
    app = create_app(
        settings=live_orchestrator.settings, orchestrator=live_orchestrator
    )
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "tell me a joke about backpressure"}
        ) as response,
    ):
        assert response.status_code == 200
        events = await parse_sse(response)

    names = [e["event"] for e in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert events[1]["data"]["route"] == "general"
    assert events[1]["data"]["source"] == "llm"  # decided by the LLM classifier
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) > 5  # genuinely chunked, not one blob
    assert joined_tokens(events).startswith("Mock model answer")


async def test_live_math_route_uses_calculator_and_narrator(live_orchestrator):
    app = create_app(
        settings=live_orchestrator.settings, orchestrator=live_orchestrator
    )
    async with (
        make_client(app) as client,
        client.stream("POST", "/ask", json={"query": "calculate 48 * 12"}) as response,
    ):
        events = await parse_sse(response)

    assert events[1]["data"]["route"] == "math"
    tool = next(e for e in events if e["event"] == "tool")["data"]
    assert tool["ok"] is True
    assert tool["result"] == "576"
    assert len([e for e in events if e["event"] == "token"]) > 5
