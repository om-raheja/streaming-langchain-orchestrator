"""Shared test doubles and fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from helpers import event_names, joined_tokens, make_client, parse_sse  # noqa: F401
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.config import Settings
from app.main import create_app
from app.orchestrator.pipeline import Orchestrator


class FakeChatModel(BaseChatModel):
    """Deterministic chat model double that streams word by word."""

    text: str = "Hello streaming world"
    fail: bool = False
    delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.text))]
        )

    async def _astream(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.fail:
            raise RuntimeError("fake model boom")
        for word in self.text.split(" "):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def offline_settings(**overrides: Any) -> Settings:
    """Settings with no provider key: exercises the degraded paths."""
    base = {
        "openai_api_key": None,
        "router_timeout_s": 0.5,
        "stall_timeout_s": 2.0,
        "max_query_chars": 500,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def offline() -> Settings:
    return offline_settings()


@pytest.fixture
def no_dotenv(monkeypatch, tmp_path):
    """Run ``load_settings`` as if no ``.env`` existed.

    Keeps credential tests hermetic: a developer's local ``.env`` (or CI
    secret) must never change what these assertions expect.
    """
    from app import config as app_config

    monkeypatch.setattr(app_config, "DOTENV_PATH", tmp_path / "missing.env")


@pytest.fixture
def make_app(offline: Settings):
    """Build an ASGI app wired to an orchestrator that uses fake models."""

    def _make(
        *,
        general_llm: FakeChatModel | None = None,
        math_llm: FakeChatModel | None = None,
        route_llm: FakeChatModel | None = None,
        settings: Settings | None = None,
    ) -> tuple[Any, Orchestrator]:
        from app.orchestrator.chains import build_general_chain, build_math_narrator
        from app.orchestrator.router import build_classifier, build_router

        settings = settings or offline
        classifier = build_classifier(route_llm) if route_llm else None
        orchestrator = Orchestrator(
            router=build_router(
                classifier=classifier, timeout_s=settings.router_timeout_s
            ),
            settings=settings,
            general_chain=build_general_chain(general_llm) if general_llm else None,
            math_narrator=build_math_narrator(math_llm) if math_llm else None,
        )
        application = create_app(settings=settings, orchestrator=orchestrator)
        return application, orchestrator

    return _make
