"""The orchestration engine behind ``POST /ask``.

``Orchestrator.stream`` is an async generator of :class:`StreamEvent` frames.
It owns the full request lifecycle:

``meta -> route -> [tool] -> token* -> done``   (or ``error``)

The design goals are the ones that matter in production:

* **Time-to-first-byte is tiny** — the ``meta`` frame is emitted before any
  model call, so the client sees an open SSE stream immediately.
* **Bounded latency** — routing and every token await is wrapped in a
  timeout; a hung provider produces a clean ``error`` frame instead of a
  permanently open socket.
* **Cancellation-safe** — if the client disconnects, ``GeneratorExit`` /
  ``CancelledError`` is re-raised so uvicorn tears the task down instead of
  buffering tokens nobody will read.
* **Provider-optional** — with no API key the router and both chains degrade
  to deterministic behaviour, which keeps the service runnable (and testable)
  offline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.runnables import Runnable

from app.config import Settings
from app.llm import build_chat_model
from app.orchestrator import chains
from app.orchestrator.math_tool import (
    MathToolError,
    evaluate,
    extract_expression,
    format_number,
)
from app.orchestrator.router import build_classifier, build_router
from app.schemas import Route, RouteDecision, StreamEvent
from app.security import safe_error_message

logger = logging.getLogger(__name__)

SERVER_NAME = "mypip-orchestrator/1.0"

OFFLINE_GENERAL_TEXT = (
    "No LLM credentials are configured on this deployment (OPENAI_API_KEY "
    "is unset), so the general branch is running in offline mode. "
    "Configure a key in your environment to enable live model streaming."
)

OFFLINE_MATH_TEXT = (
    "I could not extract a computable arithmetic expression from that "
    'query. Try something like "(12.5 * 4) + 2^10" or "15% of 240".'
)


class StreamStallError(RuntimeError):
    """Raised when the upstream model stops producing tokens in time."""


def iter_text_chunks(text: str, *, size: int = 40) -> AsyncIterator[str]:
    """Slice text into fixed-width async deltas (keeps SSE frames small)."""

    async def _gen() -> AsyncIterator[str]:
        for start in range(0, len(text), size):
            await asyncio.sleep(0)  # yield to the event loop between frames
            yield text[start : start + size]

    return _gen()


class Orchestrator:
    """Routes, executes and streams one query at a time."""

    def __init__(
        self,
        *,
        router: Runnable[str, RouteDecision],
        settings: Settings,
        general_chain: Runnable[str, str] | None = None,
        math_narrator: Runnable[dict[str, str], str] | None = None,
    ) -> None:
        self._router = router
        self._settings = settings
        self._general_chain = general_chain
        self._math_narrator = math_narrator

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> Orchestrator:
        route_llm = build_chat_model(
            settings,
            settings.route_model,
            timeout_s=settings.router_timeout_s,
            streaming=False,  # one buffered round trip is cheapest for routing
        )
        general_llm = build_chat_model(settings, settings.general_model)
        math_llm = build_chat_model(settings, settings.math_model)

        classifier = build_classifier(route_llm) if route_llm is not None else None
        router = build_router(
            classifier=classifier, timeout_s=settings.router_timeout_s
        )
        return cls(
            router=router,
            settings=settings,
            general_chain=chains.build_general_chain(general_llm)
            if general_llm
            else None,
            math_narrator=chains.build_math_narrator(math_llm) if math_llm else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    async def route(self, query: str) -> RouteDecision:
        """Resolve the branch for ``query`` (never raises)."""
        try:
            return await self._router.ainvoke(query)
        except Exception as exc:  # noqa: BLE001 - routing is best-effort
            logger.warning("router: fell back to general: %s", safe_error_message(exc))
            return RouteDecision(
                route=Route.GENERAL, reason="router failure", source="fallback"
            )

    async def stream(
        self, query: str, *, request_id: str
    ) -> AsyncIterator[StreamEvent]:
        """Yield SSE frames for ``query`` until completion or disconnect."""
        started = time.perf_counter()
        seq = 0
        ctx: dict[str, Any] = {"chunks": 0, "first_token_ms": None}

        def _now_ms() -> float:
            return (time.perf_counter() - started) * 1000

        seq += 1
        yield StreamEvent(
            event="meta",
            seq=seq,
            data={
                "request_id": request_id,
                "server": SERVER_NAME,
                "protocol": "sse-1",
                "ts": time.time(),
            },
        )

        try:
            route_started = time.perf_counter()
            decision = await self.route(query)
            routing_ms = (time.perf_counter() - route_started) * 1000

            seq += 1
            yield StreamEvent(
                event="route",
                seq=seq,
                data={
                    "request_id": request_id,
                    "route": decision.route.value,
                    "reason": decision.reason,
                    "source": decision.source,
                    "routing_ms": round(routing_ms, 2),
                },
            )
            logger.info(
                "request=%s route=%s source=%s routing_ms=%.2f",
                request_id,
                decision.route.value,
                decision.source,
                routing_ms,
            )

            if decision.route is Route.MATH:
                async for event in self._math_stream(query, ctx, _now_ms):
                    seq += 1
                    event.seq = seq
                    yield event
            else:
                async for event in self._general_stream(query, ctx, _now_ms):
                    seq += 1
                    event.seq = seq
                    yield event

            seq += 1
            yield StreamEvent(
                event="done",
                seq=seq,
                data={
                    "request_id": request_id,
                    "route": decision.route.value,
                    "chunks": ctx["chunks"],
                    "first_token_ms": ctx["first_token_ms"],
                    "duration_ms": round(_now_ms(), 2),
                },
            )
            logger.info(
                "request=%s completed chunks=%s duration_ms=%.2f first_token_ms=%s",
                request_id,
                ctx["chunks"],
                _now_ms(),
                ctx["first_token_ms"],
            )
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected / server shutting down: propagate so the
            # underlying task is cancelled rather than leaking work.
            logger.info("request=%s cancelled after %.2fms", request_id, _now_ms())
            raise
        except Exception as exc:
            logger.exception("request=%s failed", request_id)
            seq += 1
            yield StreamEvent(
                event="error",
                seq=seq,
                data={
                    "request_id": request_id,
                    "code": type(exc).__name__,
                    "message": safe_error_message(exc),
                    "duration_ms": round(_now_ms(), 2),
                },
            )

    async def aclose(self) -> None:
        """Release provider HTTP clients (best effort)."""
        for runnable in (self._general_chain, self._math_narrator):
            if runnable is None:
                continue
            for node in getattr(runnable, "steps", []) or []:
                closer = getattr(node, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        logger.debug("failed closing model client", exc_info=True)

    # ------------------------------------------------------------------
    # Branch implementations
    # ------------------------------------------------------------------
    def _token(
        self, ctx: dict[str, Any], delta: str, source: str, now_ms: Callable[[], float]
    ) -> StreamEvent:
        if ctx["first_token_ms"] is None:
            ctx["first_token_ms"] = round(now_ms(), 2)
        ctx["chunks"] += 1
        return StreamEvent(
            event="token",
            seq=0,
            data={"delta": delta, "source": source, "index": ctx["chunks"] - 1},
        )

    async def _emit_text(
        self,
        text: str,
        source: str,
        ctx: dict[str, Any],
        now_ms: Callable[[], float],
        *,
        size: int = 40,
    ) -> AsyncIterator[StreamEvent]:
        async for delta in iter_text_chunks(text, size=size):
            yield self._token(ctx, delta, source, now_ms)

    async def _timed_stream(self, aiter: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Yield from ``aiter`` with a per-chunk stall timeout."""
        timeout = self._settings.stall_timeout_s
        while True:
            try:
                async with asyncio.timeout(timeout):
                    item = await aiter.__anext__()
            except TimeoutError as exc:
                raise StreamStallError(
                    f"upstream produced no token for {timeout:.0f}s"
                ) from exc
            except StopAsyncIteration:
                return
            yield item

    async def _general_stream(
        self,
        query: str,
        ctx: dict[str, Any],
        now_ms: Callable[[], float],
    ) -> AsyncIterator[StreamEvent]:
        if self._general_chain is None:
            async for event in self._emit_text(
                OFFLINE_GENERAL_TEXT, "general", ctx, now_ms
            ):
                yield event
            return

        async for delta in self._timed_stream(self._general_chain.astream(query)):
            if not delta:
                continue
            yield self._token(ctx, str(delta), "general", now_ms)

    async def _math_stream(
        self,
        query: str,
        ctx: dict[str, Any],
        now_ms: Callable[[], float],
    ) -> AsyncIterator[StreamEvent]:
        expression = extract_expression(query)
        tool_result: dict[str, Any] = {"name": "calculator"}
        summary = ""

        if expression is None:
            tool_result.update(ok=False, error="no_arithmetic_expression")
            answer = OFFLINE_MATH_TEXT
        else:
            try:
                value = evaluate(expression)
                pretty = format_number(value)
                tool_result.update(
                    ok=True, expression=expression, result=pretty, value=value
                )
                summary = f"{expression} = {pretty}"
                answer = summary
            except MathToolError as exc:
                tool_result.update(ok=False, expression=expression, error=str(exc))
                summary = f"{expression} failed: {exc}"
                answer = f"I could not evaluate `{expression}`: {exc}."

        yield StreamEvent(event="tool", seq=0, data=tool_result)

        if self._math_narrator is None:
            async for event in self._emit_text(answer, "math", ctx, now_ms):
                yield event
            return

        payload = {"query": query, "tool_result": summary or tool_result["error"]}
        try:
            async for delta in self._timed_stream(self._math_narrator.astream(payload)):
                if not delta:
                    continue
                yield self._token(ctx, str(delta), "math", now_ms)
        except StreamStallError:
            # The tool already produced a correct answer: degrade to the
            # deterministic text rather than failing a solved request.
            logger.warning("math narrator stalled; emitting tool answer only")
            async for event in self._emit_text(answer, "math", ctx, now_ms):
                yield event


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Factory used by the ASGI app."""
    return Orchestrator.from_settings(settings)
