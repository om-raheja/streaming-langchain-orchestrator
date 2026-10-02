"""Mock OpenAI-compatible provider for local demos and CI.

It implements just enough of ``POST /v1/chat/completions`` to exercise the
real ``langchain-openai`` client:

* non-streaming + ``tools``  -> structured-output routing calls
* non-streaming              -> plain completions
* ``stream: true``           -> chunked ``chat.completion.chunk`` SSE

Run it, then point the orchestrator at it:

    python scripts/mock_openai.py --port 8124 &
    OPENAI_API_KEY=mock-key OPENAI_BASE_URL=http://127.0.0.1:8124/v1 \
      uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.orchestrator.math_tool import evaluate, extract_expression

app = FastAPI(title="mock-openai")

# Per-word delay for streamed completions. 0.02s makes incremental delivery
# visible in the demo; 0 turns the mock into a throughput benchmark.
DELAY = float(os.getenv("MOCK_DELAY", "0.02"))

_SENTENCES = [
    "Here is a streamed answer that arrives token by token.",
    "Each chunk below is a separate server-sent event frame.",
    "This exercises the real ChatOpenAI streaming client end to end.",
    "No network access and no real credentials are required for the demo.",
]


def _last_user_text(body: dict) -> str:
    for message in reversed(body.get("messages", [])):
        if message.get("role") == "human" or message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return json.dumps(content)
    return ""


def _classify(text: str) -> tuple[str, str]:
    expression = extract_expression(text)
    if expression:
        try:
            value = evaluate(expression)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None:
            return "math", f"detected expression {expression}"
    lowered = text.lower()
    if any(word in lowered for word in ("calculate", "equation", "percent", "sqrt")):
        return "math", "math vocabulary detected"
    return "general", "no computation requested"


def _answer(text: str) -> str:
    head = re.sub(r"\s+", " ", text).strip()[:120]
    sentences = [f'Mock model answer to: "{head}".', *_SENTENCES]
    return " ".join(sentences)


def _completion(body: dict, content: str | None, tool_calls: list | None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 24, "total_tokens": 36},
    }


def _stream_options(body: dict, content: str):
    """OpenAI-style ``chat.completion.chunk`` SSE generator."""

    async def event_stream():
        for word in content.split(" "):
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if DELAY:
                await asyncio.sleep(DELAY)  # human-visible streaming cadence
        final = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if os.getenv("MOCK_DEBUG"):
        print(
            "MOCK REQ:",
            json.dumps(
                {
                    k: body.get(k)
                    for k in (
                        "model",
                        "stream",
                        "tools",
                        "tool_choice",
                        "response_format",
                    )
                },
                default=str,
            )[:1200],
            flush=True,
        )
    text = _last_user_text(body)
    tools = body.get("tools") or []
    response_format = body.get("response_format") or {}

    # Structured output: the json_schema flavour used by langchain-openai 1.x
    # (tool-calling flavour is handled below for older clients).
    if response_format:
        route, reason = _classify(text)
        payload = json.dumps(
            {"route": route, "reason": reason, "source": "llm"},
            separators=(",", ":"),
        )
        if body.get("stream"):
            return _stream_options(body, payload)
        return JSONResponse(_completion(body, payload, None))

    if tools:
        tool_name = tools[0].get("function", {}).get("name", "RouteDecision")
        route, reason = _classify(text)
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"route": route, "reason": reason}),
                },
            }
        ]
        return JSONResponse(_completion(body, None, tool_calls))

    if not body.get("stream"):
        return JSONResponse(_completion(body, _answer(text), None))

    return _stream_options(body, _answer(text))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument(
        "--delay", type=float, default=None, help="seconds between streamed words"
    )
    args = parser.parse_args()
    if args.delay is not None:
        DELAY = args.delay
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
