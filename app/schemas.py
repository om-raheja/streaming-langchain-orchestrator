"""Wire-format schemas (request/response/event payloads)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Route(str, Enum):
    """The two branches the LangChain router can dispatch to."""

    MATH = "math"
    GENERAL = "general"


class RouteDecision(BaseModel):
    """Structured output produced by the semantic (LLM) classifier.

    This model doubles as the JSON-schema LangChain uses for structured
    output, so it stays intentionally flat.
    """

    model_config = ConfigDict(extra="forbid")

    route: Route
    reason: str = Field(default="", max_length=300)
    source: Literal["heuristic", "llm", "fallback"] = "llm"


class AskRequest(BaseModel):
    """POST /ask payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural-language user query")

    @field_validator("query")
    @classmethod
    def _normalise(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class StreamEvent(BaseModel):
    """One SSE frame emitted by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["meta", "route", "tool", "token", "done", "error"]
    data: dict[str, Any]
    seq: int
