"""SSE parsing helpers for tests."""

from __future__ import annotations

import json
from typing import Any

import httpx


def make_client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def parse_sse(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse an SSE body into ``{event, data, id}`` dicts."""
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    async for line in response.aiter_lines():
        if line == "":
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.lstrip(" ")
        if field == "event":
            current["event"] = value
        elif field == "data":
            current["data"] = json.loads(value)
        elif field == "id":
            current["id"] = int(value)
    if current:
        events.append(current)
    return events


def joined_tokens(events: list[dict[str, Any]]) -> str:
    return "".join(e["data"]["delta"] for e in events if e["event"] == "token")


def event_names(events: list[dict[str, Any]]) -> list[str]:
    return [e["event"] for e in events]
