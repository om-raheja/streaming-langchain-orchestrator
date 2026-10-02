"""FastAPI application: ``POST /ask`` SSE streaming endpoint."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.config import Settings, load_settings
from app.orchestrator.pipeline import Orchestrator, build_orchestrator
from app.schemas import AskRequest
from app.sse import SSE_MEDIA_TYPE, STREAM_HEADERS, sse_from_event

logger = logging.getLogger(__name__)

DESCRIPTION = """
High-speed streaming LangChain orchestrator.

`POST /ask` accepts `{"query": "..."}` and returns a **Server-Sent Events**
stream: `meta` -> `route` -> (`tool`)? -> `token`* -> `done` | `error`.
""".strip()


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(
    settings: Settings | None = None,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    """Application factory (dependency-injectable for tests)."""
    _configure_logging()
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        active = getattr(app.state, "orchestrator", None)
        if active is not None:
            await active.aclose()

    app = FastAPI(
        title="Streaming LangChain Orchestrator",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    # Set eagerly (not only inside lifespan) so the app also works under
    # bare ASGI test transports that do not run lifespan hooks.
    app.state.settings = resolved
    app.state.orchestrator = orchestrator

    def _orchestrator() -> Orchestrator:
        """Lazy singleton: keeps import side-effect free and env-overridable."""
        active = getattr(app.state, "orchestrator", None)
        if active is None:
            active = build_orchestrator(app.state.settings)
            app.state.orchestrator = active
        return active

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(
            round((time.perf_counter() - started) * 1000, 2)
        )
        return response

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        """Liveness + capability probe. Never exposes a credential."""
        settings: Settings = request.app.state.settings
        active: Orchestrator | None = getattr(request.app.state, "orchestrator", None)
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "provider": settings.llm_provider,
                "providers": settings.providers,
                "routes": ["math", "general"],
                "built": active is not None,
                "max_query_chars": settings.max_query_chars,
            }
        )

    @app.post(
        "/ask",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "SSE stream of orchestrator events.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            }
        },
    )
    async def ask(payload: AskRequest, request: Request) -> StreamingResponse:
        settings: Settings = request.app.state.settings
        if len(payload.query) > settings.max_query_chars:
            raise HTTPException(
                status_code=422,
                detail=f"query exceeds {settings.max_query_chars} characters",
            )

        request_id: str = getattr(request.state, "request_id", uuid.uuid4().hex)
        orchestrator = _orchestrator()

        async def event_stream() -> AsyncIterator[str]:
            async for event in orchestrator.stream(
                payload.query, request_id=request_id
            ):
                yield sse_from_event(event)

        headers = dict(STREAM_HEADERS)
        headers["X-Request-ID"] = request_id
        return StreamingResponse(
            event_stream(),
            media_type=SSE_MEDIA_TYPE,
            headers=headers,
        )

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        proxy_headers=True,
    )
