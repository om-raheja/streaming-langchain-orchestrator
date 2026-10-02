"""Environment-driven configuration.

Every secret and tunable is read from the process environment (optionally
seeded from a local ``.env`` file that is *never* committed).  Nothing in this
module hardcodes a credential: :func:`os.getenv` is the single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the ``app`` package.  Loading the .env relative to
# the code (not the CWD) keeps behaviour identical under uvicorn, gunicorn,
# systemd and pytest.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration resolved from the environment."""

    # --- secrets / providers -------------------------------------------------
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    llm_provider: str = "openai"
    """Active provider: ``openai`` or ``gemini`` (resolved from ``LLM_PROVIDER``)."""

    google_base_url: str | None = None
    """Optional override for the Google Generative Language API (proxies/mocks)."""

    # --- models --------------------------------------------------------------
    route_model: str = "gpt-4o-mini"
    general_model: str = "gpt-4o-mini"
    math_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # --- routing -------------------------------------------------------------
    router_timeout_s: float = 2.5
    """Hard ceiling for the semantic (LLM) router so classification latency
    can never dominate the request budget."""

    # --- streaming -----------------------------------------------------------
    stall_timeout_s: float = 30.0
    """Max seconds to wait for the *next* chunk before the stream errors out."""

    # --- limits --------------------------------------------------------------
    max_query_chars: int = 4000

    @property
    def active_api_key(self) -> str | None:
        """Key for the active provider (never logged, never returned to users)."""
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return self.openai_api_key

    @property
    def llm_configured(self) -> bool:
        return bool(self.active_api_key)

    @property
    def providers(self) -> dict[str, object]:
        """Which downstream capabilities are live (never exposes a key)."""
        return {
            "active": self.llm_provider,
            "openai": bool(self.openai_api_key),
            "gemini": bool(self.gemini_api_key),
        }


SUPPORTED_PROVIDERS = ("openai", "gemini")
_DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash"}


def _resolve_provider(
    requested: str, openai_key: str | None, gemini_key: str | None
) -> str:
    """Pick the active provider.

    ``auto`` prefers OpenAI when both keys exist (preserves existing
    deployments) and falls back to whichever key is present.
    """
    requested = (requested or "auto").lower()
    if requested == "auto":
        if openai_key:
            return "openai"
        if gemini_key:
            return "gemini"
        return "openai"  # unconfigured: both branches degrade gracefully
    if requested not in SUPPORTED_PROVIDERS:
        return "openai"
    return requested


def load_settings() -> Settings:
    """Resolve settings from the environment.

    ``load_dotenv`` never overrides variables that are already set, so real
    process-level env vars (Docker/K8s secrets) always win over a local file.
    """
    load_dotenv(DOTENV_PATH, override=False)

    # Credentials: the only point where a key enters the process.
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    gemini_api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()  # common alias
        or None
    )
    provider = _resolve_provider(
        _env("LLM_PROVIDER", "auto") or "auto", openai_api_key, gemini_api_key
    )
    default_model = _DEFAULT_MODELS[provider]

    return Settings(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        llm_provider=provider,
        google_base_url=_env("GOOGLE_BASE_URL"),
        route_model=_env("ROUTE_MODEL", default_model) or default_model,
        general_model=_env("GENERAL_MODEL", default_model) or default_model,
        math_model=_env("MATH_MODEL", default_model) or default_model,
        temperature=_env_float("MODEL_TEMPERATURE", 0.0),
        router_timeout_s=_env_float("ROUTER_TIMEOUT_S", 2.5),
        stall_timeout_s=_env_float("STREAM_STALL_TIMEOUT_S", 30.0),
        max_query_chars=_env_int("MAX_QUERY_CHARS", 4000),
    )
