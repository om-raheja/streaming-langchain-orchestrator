"""Security acceptance criteria: zero exposed API keys, secrets redacted."""

from __future__ import annotations

from pathlib import Path

from app.config import load_settings
from app.security import contains_secret, redact, safe_error_message

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT / "app"
SKIP_DIRS = {"__pycache__", ".venv", "node_modules"}


def _production_sources() -> list[Path]:
    return [
        path
        for path in APP_ROOT.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.parts)
    ]


def test_no_hardcoded_credentials_in_production_code():
    """Strict acceptance criterion: ZERO exposed API keys in the code."""
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _production_sources()
        if contains_secret(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"possible hardcoded secrets in: {offenders}"


def test_api_key_is_read_from_environment():
    config = (APP_ROOT / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("OPENAI_API_KEY"' in config
    # The credential only ever exists as an env lookup, never a literal.
    for path in _production_sources():
        text = path.read_text(encoding="utf-8")
        assert not contains_secret(text), path


def test_env_example_contains_placeholder_only():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in example
    assert not contains_secret(example)


def test_gitignore_excludes_env_file():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore


def test_settings_load_without_key_by_default(monkeypatch, no_dotenv):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.openai_api_key is None
    assert settings.gemini_api_key is None
    assert settings.llm_configured is False
    assert settings.llm_provider == "openai"  # unconfigured: offline default
    assert settings.providers == {
        "active": "openai",
        "openai": False,
        "gemini": False,
    }


def test_redaction_masks_secret_shapes():
    # Deliberately fake, human-readable placeholder — never a real credential.
    secret = "sk-test-PLACEHOLDER-not-a-real-key"

    masked = redact(f"failed auth with {secret}")
    assert secret not in masked
    assert "[REDACTED]" in masked

    assert secret not in redact(f"Authorization: Bearer {secret}")

    message = safe_error_message(RuntimeError(f"boom {secret}"))
    assert secret not in message
    assert message.startswith("RuntimeError: boom")

    assert "[REDACTED]" in safe_error_message(RuntimeError(secret))


def test_error_messages_are_truncated():
    long = "x" * 1000
    assert len(safe_error_message(ValueError(long), limit=120)) <= 120
