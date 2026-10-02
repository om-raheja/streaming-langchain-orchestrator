"""Secret hygiene helpers.

Two jobs:

1. Make sure that if a secret ever lands in a log line, exception message or
   SSE error payload, it is scrubbed before it leaves the process.
2. Provide the predicates used by the test-suite to prove no key was ever
   hardcoded in this repository.
"""

from __future__ import annotations

import re

# Common provider key shapes (OpenAI, Anthropic, HuggingFace, generic bearer
# tokens).  Deliberately broad: false-positive redaction is harmless, the
# opposite is not.
_SECRET_RE = re.compile(
    r"""
    (?:
        sk-[A-Za-z0-9_\-]{8,}              # OpenAI / Anthropic style keys
      | sk-ant-[A-Za-z0-9_\-]{8,}
      | hf_[A-Za-z0-9_\-]{8,}
      | AIza[0-9A-Za-z_\-]{20,}             # Google API keys
      | xox[baprs]-[A-Za-z0-9\-]{10,}       # Slack tokens
      | Bearer\s+[A-Za-z0-9._\-]{16,}
      | (?i:api[_-]?key)\s*[=:]\s*['"][A-Za-z0-9._\-]{16,}['"]
    )
    """,
    re.VERBOSE,
)

REDACTED = "[REDACTED]"


def redact(text: object) -> str:
    """Return ``text`` with anything secret-looking replaced."""
    if not isinstance(text, str):
        text = str(text)
    return _SECRET_RE.sub(REDACTED, text)


def contains_secret(text: str) -> bool:
    """True when ``text`` appears to embed a credential."""
    return _SECRET_RE.search(text) is not None


def safe_error_message(exc: BaseException, *, limit: int = 300) -> str:
    """A short, redacted description of an exception safe to send to clients."""
    message = redact(f"{type(exc).__name__}: {exc}")
    if len(message) > limit:
        message = message[: limit - 1] + "…"
    return message
