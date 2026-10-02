"""Mock Google Gemini API (``v1beta`` REST) for offline tests.

Implements just enough of the Generative Language API for the real
``langchain-google-genai`` client:

* ``POST /v1beta/models/{model}:generateContent``       -> JSON completion
* ``POST /v1beta/models/{model}:streamGenerateContent`` -> SSE chunks

Both honour ``generationConfig.response_mime_type == "application/json"``
(structured output), which is what the router's classifier uses.

    python scripts/mock_gemini.py --port 8200 &
    LLM_PROVIDER=gemini GEMINI_API_KEY=mock-key GOOGLE_BASE_URL=http://127.0.0.1:8200 \
      uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.orchestrator.math_tool import evaluate, extract_expression

app = FastAPI(title="mock-gemini")

SENTENCES = [
    "Here is a streamed Gemini-style answer that arrives token by token.",
    "Each chunk below is a separate server-sent event frame.",
    "No network access and no real credentials are needed for this demo.",
]


def _log(body: dict, model: str, streaming: bool) -> None:
    if os.getenv("MOCK_DEBUG"):
        config = body.get("generationConfig", {})
        print(
            "MOCK GEMINI REQ:",
            json.dumps(
                {
                    "model": model,
                    "streaming": streaming,
                    "generationConfig": config,
                    "tools": bool(body.get("tools")),
                    "top_keys": sorted(body.keys()),
                },
                default=str,
            )[:1500],
            flush=True,
        )


def _user_text(body: dict) -> str:
    chunks: list[str] = []
    for message in body.get("contents", []):
        if message.get("role") in (None, "user"):
            for part in message.get("parts", []):
                if "text" in part:
                    chunks.append(part["text"])
    return "\n".join(chunks)


def _classify(text: str) -> tuple[str, str]:
    expression = extract_expression(text)
    if expression:
        try:
            evaluate(expression)
        except Exception:  # noqa: BLE001 - mock only cares that it parses
            return "general", "expression could not be evaluated"
        return "math", f"detected expression {expression}"
    lowered = text.lower()
    if any(word in lowered for word in ("calculate", "equation", "percent", "sqrt")):
        return "math", "math vocabulary detected"
    return "general", "no computation requested"


def _answer(text: str) -> str:
    head = " ".join(text.split())[:120]
    return f'Mock Gemini answer to: "{head}". ' + " ".join(SENTENCES)


def _payload(model: str, text: str) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 24,
            "totalTokenCount": 36,
        },
        "modelVersion": model,
    }


def _completion_text(body: dict, text: str) -> str:
    config = body.get("generationConfig", {})
    # google-genai serialises config keys in camelCase for the REST wire format.
    mime = config.get("responseMimeType") or config.get("response_mime_type")
    if mime == "application/json":
        route, reason = _classify(text)
        return json.dumps(
            {"route": route, "reason": reason, "source": "llm"}, separators=(",", ":")
        )
    return _answer(text)


@app.post("/v1beta/models/{model_path}")
async def generate(model_path: str, request: Request):
    model, _, action = model_path.partition(":")
    body = await request.json()
    streaming = action.startswith("streamGenerateContent")
    _log(body, model, streaming)

    text = _completion_text(body, _user_text(body))

    if not streaming:
        return JSONResponse(_payload(model, text))

    async def event_stream():
        for word in text.split(" "):
            yield f"data: {json.dumps(_payload(model, word + ' '))}\n\n"
            if float(os.getenv("MOCK_DELAY", "0")):
                await asyncio.sleep(float(os.getenv("MOCK_DELAY", "0")))
        yield f"data: {json.dumps(_payload(model, ''))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
