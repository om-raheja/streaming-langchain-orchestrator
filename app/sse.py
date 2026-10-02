"""Server-Sent Events framing.

The framing lives in one place so every event the API emits is guaranteed to
be spec compliant:

* a single-line ``data:`` payload (JSON never contains raw newlines),
* an ``event:`` name and monotonically increasing ``id:`` for reconnects,
* messages terminated by a blank line.
"""

from __future__ import annotations

import json
from typing import Any

from app.schemas import StreamEvent

SSE_MEDIA_TYPE = "text/event-stream"

# Responses must never be buffered by a proxy: nginx needs
# ``X-Accel-Buffering: no`` and the OS/uvicorn layer must not gzip or buffer.
STREAM_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": f"{SSE_MEDIA_TYPE}; charset=utf-8",
}


def _encode(data: Any) -> str:
    # ensure_ascii=False keeps unicode readable; json.dumps guarantees the
    # string is a single physical line, which SSE requires for ``data:``.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def sse_frame(event: str, data: dict[str, Any], *, seq: int | None = None) -> str:
    """Serialise one SSE message."""
    parts: list[str] = []
    if seq is not None:
        parts.append(f"id: {seq}")
    parts.append(f"event: {event}")
    parts.append(f"data: {_encode(data)}")
    return "\n".join(parts) + "\n\n"


def sse_from_event(event: StreamEvent) -> str:
    return sse_frame(event.event, event.data, seq=event.seq)


def sse_comment(text: str = "ping") -> str:
    """Keep-alive comment for long-running generations."""
    return f": {text}\n\n"
