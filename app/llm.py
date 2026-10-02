"""Chat-model construction for the configured provider.

The factory is the *only* place in the codebase that touches a provider
credential, and it reads both from the environment — never from a literal:

* ``LLM_PROVIDER=openai`` (or ``auto`` with ``OPENAI_API_KEY`` set) →
  ``langchain-openai.ChatOpenAI``
* ``LLM_PROVIDER=gemini`` (or ``auto`` with ``GEMINI_API_KEY`` set) →
  ``langchain-google-genai.ChatGoogleGenerativeAI``

When the active provider has no key the factory returns ``None`` and the
orchestrator degrades to deterministic (LLM-free) behaviour instead of
crashing.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings

logger = logging.getLogger(__name__)


def build_chat_model(
    settings: Settings,
    model_name: str,
    *,
    timeout_s: float = 20.0,
    max_retries: int = 1,
    streaming: bool = True,
) -> BaseChatModel | None:
    """Return a chat model for the active provider, or ``None`` if unkeyed.

    ``streaming=False`` is used for request/response calls (the routing
    classifier) where a single buffered round trip is cheaper than an SSE
    session; ``streaming=True`` is used wherever tokens reach the client.
    """
    if not settings.llm_configured:
        return None

    if settings.llm_provider == "gemini":
        return _build_gemini(settings, model_name, timeout_s, max_retries, streaming)
    return _build_openai(settings, model_name, timeout_s, max_retries, streaming)


def _build_openai(
    settings: Settings,
    model_name: str,
    timeout_s: float,
    max_retries: int,
    streaming: bool,
) -> BaseChatModel:
    # Imported lazily so the app (and its tests) can run without the provider
    # SDK being importable when no key is configured.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,  # sourced from os.getenv
        temperature=settings.temperature,
        timeout=timeout_s,
        max_retries=max_retries,
        streaming=streaming,
    )


def _build_gemini(
    settings: Settings,
    model_name: str,
    timeout_s: float,
    max_retries: int,
    streaming: bool,
) -> BaseChatModel | None:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:  # pragma: no cover - dependency listed in requirements
        logger.error(
            "LLM_PROVIDER=gemini but langchain-google-genai is not installed; "
            "run `pip install langchain-google-genai`"
        )
        return None

    kwargs: dict = {
        "model": model_name,
        "google_api_key": settings.gemini_api_key,  # sourced from os.getenv
        "temperature": settings.temperature,
        "timeout": timeout_s,
        "max_retries": max_retries,
        "streaming": streaming,
    }
    if settings.google_base_url:
        # Pointed at a proxy or the bundled mock Gemini server for tests.
        kwargs["base_url"] = settings.google_base_url
    return ChatGoogleGenerativeAI(**kwargs)
