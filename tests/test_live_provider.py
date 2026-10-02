"""Live smoke tests against a real provider.

Skipped automatically unless a key is present in the environment, so CI and
offline runs never touch the network:

    GEMINI_API_KEY=AIza...   pytest tests/test_live_provider.py -v
    OPENAI_API_KEY=sk-...    pytest tests/test_live_provider.py -v

These are intentionally small (three queries) to keep quota use minimal.

Free-tier keys are throttled hard (e.g. 20 requests/day per model), so a
provider-side ``429 RESOURCE_EXHAUSTED``/``503`` skips the test with an
explanation instead of failing it — an exhausted quota is an environment
condition, not a regression in this codebase. Real assertion failures and
non-quota provider errors still fail loudly.
"""

from __future__ import annotations

import os

import pytest
from helpers import joined_tokens, make_client, parse_sse

from app.config import load_settings
from app.main import create_app
from app.orchestrator.pipeline import Orchestrator

_HAS_GEMINI = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
_HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not (_HAS_GEMINI or _HAS_OPENAI),
    reason="no provider key: set GEMINI_API_KEY or OPENAI_API_KEY to run live tests",
)


@pytest.fixture(scope="module")
def live_orchestrator() -> Orchestrator:
    settings = load_settings()
    if not settings.llm_configured:  # pragma: no cover - guarded by pytestmark
        pytest.skip("provider key not configured")
    return Orchestrator.from_settings(settings)


_QUOTA_MARKERS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "quota",
    "rate limit",
    "rate_limit",
    "overloaded",
)


def _quota_related(message: str) -> bool:
    """True when a provider error is throttling/availability, not a defect."""
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in _QUOTA_MARKERS)


async def _stream_events(orchestrator: Orchestrator, query: str) -> list[dict]:
    """POST ``query`` and parse the SSE stream.

    Skips the test if the provider throttles us (free-tier daily quota,
    transient overload) either by raising out of the stream or by ending it
    with an ``error`` frame; any other provider error is re-raised.
    """
    app = create_app(settings=orchestrator.settings, orchestrator=orchestrator)
    try:
        async with (
            make_client(app) as client,
            client.stream("POST", "/ask", json={"query": query}) as response,
        ):
            assert response.status_code == 200
            events = await parse_sse(response)
    except Exception as exc:
        if _quota_related(str(exc)):
            pytest.skip(f"provider throttled the request: {exc}")
        raise

    if events and events[-1]["event"] == "error":
        message = str(events[-1]["data"].get("message", ""))
        if _quota_related(message):
            pytest.skip(f"provider throttled the request: {message}")
    return events


async def test_live_health_reports_active_provider(live_orchestrator):
    app = create_app(
        settings=live_orchestrator.settings, orchestrator=live_orchestrator
    )
    async with make_client(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] in {"openai", "gemini"}
    assert body["providers"]["active"] == body["provider"]
    # The key itself must never appear in a response body.
    assert "AIza" not in response.text
    assert "sk-" not in response.text


async def test_live_math_route_is_deterministic(live_orchestrator):
    events = await _stream_events(live_orchestrator, "calculate 48 * 12")

    assert events[1]["data"]["route"] == "math"
    tool = next(e for e in events if e["event"] == "tool")["data"]
    assert tool["ok"] is True
    assert tool["result"] == "576"  # local calculator, provider-independent
    assert events[-1]["event"] == "done"
    assert len([e for e in events if e["event"] == "token"]) >= 1


async def test_live_general_route_streams_from_provider(live_orchestrator):
    events = await _stream_events(
        live_orchestrator, "In one sentence, what is backpressure?"
    )

    assert events[1]["data"]["route"] == "general"
    assert events[1]["data"]["source"] in {"heuristic", "llm", "fallback"}
    assert events[-1]["event"] == "done"
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) >= 1
    assert len(joined_tokens(events).strip()) > 10
