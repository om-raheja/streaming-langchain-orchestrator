"""Gemini provider: configuration, model factory and full streaming path.

Offline by default — the integration cases run the **real**
``langchain-google-genai`` client against ``scripts/mock_gemini.py`` served
in-process, so no Google credential or network access is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from pathlib import Path

import pytest
from helpers import joined_tokens, make_client, parse_sse

from app.config import Settings, load_settings
from app.llm import build_chat_model
from app.main import create_app
from app.orchestrator.pipeline import Orchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_PATH = PROJECT_ROOT / "scripts" / "mock_gemini.py"

PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LLM_PROVIDER",
    "ROUTE_MODEL",
    "GENERAL_MODEL",
    "MATH_MODEL",
    "GOOGLE_BASE_URL",
)


# ---------------------------------------------------------------------------
# Configuration / credential handling
# ---------------------------------------------------------------------------
def test_auto_prefers_openai_then_gemini(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)

    assert load_settings().llm_provider == "openai"  # nothing configured

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-gemini-key")
    settings = load_settings()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key == "unit-test-gemini-key"
    assert settings.llm_configured is True
    assert settings.active_api_key == "unit-test-gemini-key"
    assert settings.providers["gemini"] is True
    # Provider-specific default model
    assert settings.general_model == "gemini-2.5-flash"
    assert settings.route_model == "gemini-2.5-flash"

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-openai-key")
    assert load_settings().llm_provider == "openai"  # auto prefers openai


def test_explicit_provider_and_google_api_key_alias(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "alias-key")
    assert load_settings().llm_provider == "gemini"  # alias accepted

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "some-openai-key")
    settings = load_settings()
    assert settings.llm_provider == "gemini"  # explicit beats auto
    assert settings.active_api_key == "alias-key"

    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    assert load_settings().llm_provider == "openai"  # unknown -> safe default


def test_openai_defaults_survive_when_openai_selected(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "some-openai-key")
    settings = load_settings()
    assert settings.general_model == "gpt-4o-mini"


def test_build_chat_model_returns_none_without_key():
    settings = Settings(openai_api_key=None, gemini_api_key=None, llm_provider="gemini")
    assert build_chat_model(settings, "gemini-2.5-flash") is None


def test_build_chat_model_builds_gemini_client(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        gemini_api_key="unit-test-gemini-key",
        llm_provider="gemini",
        general_model="gemini-2.5-flash",
        temperature=0.0,
    )
    model = build_chat_model(settings, settings.general_model, streaming=True)
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.streaming is True

    classifier = build_chat_model(settings, settings.route_model, streaming=False)
    assert isinstance(classifier, ChatGoogleGenerativeAI)
    assert classifier.streaming is False


def test_build_chat_model_still_builds_openai_client():
    from langchain_openai import ChatOpenAI

    settings = Settings(openai_api_key="unit-test-openai-key", llm_provider="openai")
    model = build_chat_model(settings, "gpt-4o-mini")
    assert isinstance(model, ChatOpenAI)


# ---------------------------------------------------------------------------
# Integration: real client -> mock Google API -> SSE endpoint
# ---------------------------------------------------------------------------
def _load_mock_app():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mock_gemini", MOCK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def mock_gemini_url():
    import uvicorn

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
        pytest.fail("mock gemini provider did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 3)


@pytest.fixture
def gemini_orchestrator(mock_gemini_url, monkeypatch) -> Orchestrator:
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")  # never a real key
    monkeypatch.setenv("GOOGLE_BASE_URL", mock_gemini_url)
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")

    settings = load_settings()
    assert settings.llm_provider == "gemini"
    orchestrator = Orchestrator.from_settings(settings)
    yield orchestrator
    os.environ.pop("GOOGLE_BASE_URL", None)


async def test_gemini_classifies_and_streams(gemini_orchestrator):
    app = create_app(
        settings=gemini_orchestrator.settings, orchestrator=gemini_orchestrator
    )
    async with (
        make_client(app) as client,
        client.stream(
            "POST", "/ask", json={"query": "tell me a joke about flash attention"}
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = await parse_sse(response)

    names = [e["event"] for e in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "error" not in names
    # Structured-output routing (responseMimeType=application/json) worked.
    assert events[1]["data"]["route"] == "general"
    assert events[1]["data"]["source"] == "llm"

    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) > 5  # chunked, not one blob
    assert joined_tokens(events).startswith("Mock Gemini answer")


async def test_gemini_math_route_runs_calculator_then_narrates(gemini_orchestrator):
    app = create_app(
        settings=gemini_orchestrator.settings, orchestrator=gemini_orchestrator
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
    assert events[-1]["event"] == "done"
    assert len([e for e in events if e["event"] == "token"]) > 5
