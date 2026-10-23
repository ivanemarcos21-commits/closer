"""Runtime settings, read from environment variables (or a local ``.env`` file)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Supports Groq structured outputs on the free plan. Llama 3.3 was removed from that plan.
DEFAULT_MODEL = "openai/gpt-oss-20b"

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


def _load_dotenv() -> None:
    """Load the project's ``.env``; real environment variables always win."""
    if not _ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.warning(".env was found but python-dotenv is not installed; run: pip install -r requirements.txt")
        return
    load_dotenv(_ENV_FILE, override=False)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = field(default=None, repr=False)  # never print the key
    model: str = DEFAULT_MODEL
    match_model: str | None = None  # model that judges eligibility; defaults to ``model``
    zero_retention: bool = False  # true only after Zero Data Retention is on in the Groq console
    operator_name: str | None = None  # shown on the terms page when this copy is published
    operator_email: str | None = None
    llm_timeout_s: float = 60.0
    max_trials: int = 20  # studies the AI reads per search
    batch_size: int = 5  # trials judged per Gemini request
    max_workers: int = 3  # concurrent Gemini requests
    story_max_chars: int = 4000
    searches_per_hour: int = 40  # app-wide cap, protects a shared key from abuse

    @property
    def has_ai(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def effective_match_model(self) -> str:
        return self.match_model or self.model

    @classmethod
    def load(
        cls, extra: Mapping[str, str | None] | None = None, *, use_dotenv: bool = True
    ) -> Settings:
        """Read settings from the environment; ``extra`` (e.g. Streamlit secrets) fills the gaps."""
        if use_dotenv:
            _load_dotenv()
        extra = extra or {}

        def get(name: str) -> str | None:
            value = str(os.environ.get(name) or extra.get(name) or "").strip()
            return value or None

        return cls(
            groq_api_key=get("GROQ_API_KEY"),
            model=get("GROQ_MODEL") or DEFAULT_MODEL,
            match_model=get("GROQ_MATCH_MODEL"),
            zero_retention=(get("GROQ_ZERO_RETENTION") or "").lower() in _TRUE_WORDS,
            operator_name=get("OPERATOR_NAME"),
            operator_email=get("OPERATOR_EMAIL"),
        )
