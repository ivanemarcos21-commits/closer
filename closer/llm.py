"""Groq access behind one small interface.

Everything else in the app talks to ``StructuredLLM.generate()``; only this file knows the
Groq chat-completions request. That keeps the rest of the code testable with a fake.

Privacy: prompts contain the patient's own words, so nothing in this module ever logs or
echoes prompt text. Error messages are built from status codes and our own wording only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .net import CERTIFICATE_HELP, is_certificate_error

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MODELS_DOC_URL = "https://console.groq.com/docs/models"
API_KEY_URL = "https://console.groq.com/keys"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_RETRY_DELAY_S = 30.0

# Failures that will repeat for every later request too: stop spending quota on them. (A
# "blocked" answer depends on that one request's text, so it is NOT in this set.)
FATAL_KINDS = frozenset({"auth", "model", "quota_daily", "certificate"})
_RETRYABLE_KINDS = frozenset({"rate_limit", "unavailable", "network", "bad_output"})
MAX_OUTPUT_TOKENS = 8192  # bounds the cost of one answer; a 5-trial verdict is ~1,500 tokens

_BUSY_FOR_PATIENTS = "The AI helper is busy right now. Please wait a minute and try again."
_PATIENT_MESSAGES = {
    "quota_daily": (
        "Closer has used up its free AI limit for today. Please try again tomorrow. "
        "You can also search ClinicalTrials.gov yourself."
    ),
    "rate_limit": _BUSY_FOR_PATIENTS,
    "unavailable": _BUSY_FOR_PATIENTS,
    "network": _BUSY_FOR_PATIENTS,
    "blocked": (
        "The AI helper couldn't read this description. Try describing the diagnosis in "
        "plainer words."
    ),
    "bad_output": "The AI helper gave an answer Closer couldn't use. Please try again.",
    "too_large": (
        "Some study descriptions were too long to check in one pass. Those studies are listed "
        "so you can read their criteria yourself."
    ),
}
_PATIENT_DEFAULT = (
    "Closer's AI helper isn't working right now. That isn't something you did. "
    "You can still search ClinicalTrials.gov yourself."
)


class LLMError(RuntimeError):
    """The AI step failed. ``kind`` says why.

    ``str(error)`` is written for the person running the app (it may say which environment
    variable to fix); ``user_message`` is written for a patient and never mentions setup.
    """

    def __init__(self, message: str, *, kind: str = "unknown", retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE_KINDS

    @property
    def user_message(self) -> str:
        return _PATIENT_MESSAGES.get(self.kind, _PATIENT_DEFAULT)


class StructuredLLM(Protocol):
    """Anything that can turn (instructions + prompt) into a validated Pydantic object."""

    def generate(
        self, *, system: str, prompt: str, schema: type[T], model: str | None = None
    ) -> T: ...


# --------------------------------------------------------------------------- schema
_DROPPED_KEYWORDS = frozenset({"title", "default", "examples", "$defs", "definitions"})


def to_gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``model`` in the plain, self-contained shape Gemini accepts best.

    Pydantic emits ``$ref``/``$defs``, ``title`` and ``default`` keywords. We inline the
    references, drop the keywords the API does not need, and mark every property required so
    the model always fills each field (Pydantic's defaults still apply when *we* build
    objects by hand).
    """
    schema = model.model_json_schema()
    defs = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    def resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in stack:
                raise ValueError(f"Recursive schema {name!r} is not supported")
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return {**resolve(defs[name], (*stack, name)), **resolve(siblings, stack)}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROPPED_KEYWORDS:
                continue
            if key == "properties":  # keys here are field names, not schema keywords
                out[key] = {name: resolve(sub, stack) for name, sub in value.items()}
            else:
                out[key] = resolve(value, stack)
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
        return out

    return resolve(schema, ())


# --------------------------------------------------------------------------- errors
def classify_error(exc: BaseException) -> LLMError:
    """Turn an SDK or network exception into an ``LLMError`` with advice a user can act on."""
    error = _classify(exc)
    error.__cause__ = exc
    return error


def _classify(exc: BaseException) -> LLMError:
    if is_certificate_error(exc):
        return LLMError(f"Groq: {CERTIFICATE_HELP}", kind="certificate")
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("Groq took too long to answer.", kind="network")
    if isinstance(exc, httpx.HTTPError):
        return LLMError("Could not reach Groq (connection problem).", kind="network")
    return LLMError(f"The AI step failed unexpectedly ({type(exc).__name__}).", kind="unknown")


# --------------------------------------------------------------------------- Groq client
class GroqLLM:
    """``StructuredLLM`` backed by Groq chat completions and a JSON schema."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        deadline_s: float = 120.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._max_attempts = max(1, max_attempts)
        self._deadline_s = deadline_s
        self._sleep = sleep
        self._clock = clock
        self._resume_at = 0.0
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout_s)

    def generate(
        self, *, system: str, prompt: str, schema: type[T], model: str | None = None
    ) -> T:
        body = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": to_gemini_schema(schema),
                },
            },
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        bad_outputs = 0
        started = self._clock()
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._wait_for_token_budget(started)
                response = self._http.post(
                    API_URL,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    raise _error_from_response(response)
                self._remember_token_budget(response)
                return _parse_response(response.json(), schema)
            except LLMError as err:
                error = err
            except Exception as exc:
                error = classify_error(exc)

            if error.kind == "bad_output":
                bad_outputs += 1
            if not error.retryable or attempt == self._max_attempts or bad_outputs > 1:
                raise error
            delay = error.retry_after if error.retry_after is not None else min(2.0**attempt, 20.0)
            if self._clock() - started + delay > self._deadline_s:
                raise error
            log.warning("Groq call failed (%s); retry %d in %.0fs", error.kind, attempt, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def _wait_for_token_budget(self, started: float) -> None:
        """The free plan allows about 8,000 tokens a minute. Wait out a known reset before asking again."""
        delay = self._resume_at - self._clock()
        if delay <= 0.5:
            return
        if self._clock() - started + delay > self._deadline_s:
            return
        log.warning("Waiting %.0fs for Groq's token limit to reset", delay)
        self._sleep(delay)

    def _remember_token_budget(self, response: httpx.Response) -> None:
        try:
            remaining = float(response.headers.get("x-ratelimit-remaining-tokens", ""))
        except ValueError:
            return
        if remaining >= 2500:
            return
        reset = _parse_reset_interval(response.headers.get("x-ratelimit-reset-tokens", ""))
        if reset is None:
            return
        self._resume_at = self._clock() + min(reset, _MAX_RETRY_DELAY_S)


def _error_from_response(response: httpx.Response) -> LLMError:
    """Map a Groq HTTP error to an ``LLMError``. The patient's words are never copied in."""
    message = ""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(error, str):
            message = error
    haystack = message.lower()
    code = response.status_code
    retry_after = _retry_after_header(response)

    if code == 429:
        if any(word in haystack for word in ("per day", "daily", "tokens per day", "tpd")):
            return LLMError(
                "The free daily limit for this Groq model has been reached. It resets later "
                f"today. You can also set GROQ_MODEL to another free model: {MODELS_DOC_URL}",
                kind="quota_daily",
            )
        return LLMError("Groq is receiving too many requests right now.", kind="rate_limit", retry_after=retry_after)
    if code in (401, 403) or "api key" in haystack:
        return LLMError(
            f"Groq rejected the API key. Check GROQ_API_KEY (free keys: {API_KEY_URL}).",
            kind="auth",
        )
    if code == 404 or ("model" in haystack and "not" in haystack):
        return LLMError(
            f"Groq does not know that model name. Set GROQ_MODEL to a current model: {MODELS_DOC_URL}",
            kind="model",
        )
    if code >= 500:
        return LLMError("Groq is temporarily overloaded.", kind="unavailable")
    if code == 413 or "request too large" in haystack:
        requested, limit = _token_figures(message)
        if requested is None or limit is None or requested > limit:
            return LLMError(
                "The study text sent to Groq was larger than its per-minute token limit, so it "
                "has to be checked in smaller pieces.",
                kind="too_large",
            )
        return LLMError(
            "Groq is receiving too many requests right now.",
            kind="rate_limit",
            retry_after=retry_after,
        )
    if code == 400:
        if "validate json" in haystack or "failed_generation" in haystack:
            return LLMError("Groq's answer did not match the expected format.", kind="bad_output")
        return LLMError(f"Groq rejected the request: {message[:200]}", kind="bad_request")
    return LLMError(f"Groq returned HTTP {code}.", kind="unknown")


def _token_figures(message: str) -> tuple[int | None, int | None]:
    """Pull ``Requested`` and ``Limit`` out of a Groq token-limit error, when it includes them."""
    requested = re.search(r"requested\s+(\d+)", message, flags=re.IGNORECASE)
    limit = re.search(r"limit\s+(\d+)", message, flags=re.IGNORECASE)
    return (
        int(requested.group(1)) if requested else None,
        int(limit.group(1)) if limit else None,
    )


def _parse_reset_interval(value: str) -> float | None:
    """Groq sends token-reset delays as ``6.547s`` or ``1ms``."""
    text = value.strip().lower()
    if not text:
        return None
    try:
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        if text.endswith("s"):
            return float(text[:-1])
        return float(text)
    except ValueError:
        return None


def _retry_after_header(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after", "")
    try:
        return min(max(float(raw), 0.0), _MAX_RETRY_DELAY_S)
    except ValueError:
        return None


def _parse_response(payload: Any, schema: type[T]) -> T:
    if not isinstance(payload, dict):
        raise LLMError("Groq returned an unexpected response.", kind="bad_output")
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        message = {}
    finish = str((choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else "") or "")
    text = message.get("content")
    if finish.lower() in {"content_filter", "safety"} or message.get("refusal"):
        raise LLMError(f"Groq declined to process this text ({finish or 'refusal'}).", kind="blocked")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("Groq returned an empty answer.", kind="bad_output")
    try:
        return schema.model_validate_json(_strip_json_fence(text))
    except ValidationError as exc:
        raise LLMError("Groq answered in an unexpected format.", kind="bad_output") from exc


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


def create_llm(settings: Settings) -> GroqLLM:
    if not settings.groq_api_key:
        raise LLMError(
            f"No Groq API key is configured. Create a free key at {API_KEY_URL} and set "
            "GROQ_API_KEY (see .env.example).",
            kind="auth",
        )
    return GroqLLM(settings.groq_api_key, settings.model, timeout_s=settings.llm_timeout_s)


class _Ping(BaseModel):
    ok: bool


def check_connection(llm: StructuredLLM, *, model: str | None = None) -> None:
    """Make one tiny structured call; raises ``LLMError`` with advice if anything is wrong."""
    answer = llm.generate(
        system="You are a connectivity check. Reply with JSON only.",
        prompt="Set ok to true.",
        schema=_Ping,
        model=model,
    )
    if not answer.ok:
        raise LLMError("Groq answered, but not as expected.", kind="bad_output")
