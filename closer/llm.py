"""Gemini access behind one small interface.

Everything else in the app talks to ``StructuredLLM.generate()``; only this file knows about
the google-genai SDK. That keeps the rest of the code testable with a fake, and moving to
another SDK surface (for example Google's newer Interactions API) a one-file change.

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

MODELS_DOC_URL = "https://ai.google.dev/gemini-api/docs/models"
API_KEY_URL = "https://aistudio.google.com/apikey"
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
        return LLMError(f"Gemini: {CERTIFICATE_HELP}", kind="certificate")
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("Gemini took too long to answer.", kind="network")
    if isinstance(exc, httpx.HTTPError):
        return LLMError("Could not reach Gemini (connection problem).", kind="network")

    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    status = str(getattr(exc, "status", "") or "")
    message = str(getattr(exc, "message", "") or "")
    details = getattr(exc, "details", None)
    haystack = f"{status} {message} {_dump(details)}".lower()

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        squashed = haystack.replace("_", "").replace(" ", "")
        if "perday" in squashed:
            return LLMError(
                "The free daily limit for this Gemini model has been reached. It resets at "
                "midnight Pacific time. You can also set GEMINI_MODEL to a model with a higher "
                "free limit.",
                kind="quota_daily",
            )
        return LLMError(
            "Gemini is receiving too many requests right now.",
            kind="rate_limit",
            retry_after=_retry_delay_seconds(details),
        )
    if code in (401, 403) or any(
        marker in haystack
        for marker in ("api_key_invalid", "api key not valid", "api key expired", "permission_denied")
    ):
        return LLMError(
            f"Gemini rejected the API key or the model is not available to your project. "
            f"Check GEMINI_API_KEY (free keys: {API_KEY_URL}).",
            kind="auth",
        )
    if code == 404 or ("not found" in haystack and "model" in haystack):
        return LLMError(
            f"Gemini does not know that model name. Set GEMINI_MODEL to a current model: "
            f"{MODELS_DOC_URL}",
            kind="model",
        )
    if isinstance(code, int) and code >= 500:
        return LLMError("Gemini is temporarily overloaded.", kind="unavailable")
    if code == 400:
        return LLMError(f"Gemini rejected the request: {message[:200]}", kind="bad_request")
    return LLMError(f"The AI step failed unexpectedly ({type(exc).__name__}).", kind="unknown")


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return ""


def _retry_delay_seconds(details: Any) -> float | None:
    """Find Gemini's ``retryDelay`` ("7s", "0.5s") anywhere inside an error payload."""
    pending = [details]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            value = item.get("retryDelay")
            if isinstance(value, str):
                match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", value)
                if match:
                    return min(float(match.group(1)), _MAX_RETRY_DELAY_S)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return None


# --------------------------------------------------------------------------- Gemini client
class GeminiLLM:
    """``StructuredLLM`` backed by the Gemini API (``generate_content`` + JSON schema)."""

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
        from google import genai  # imported here so the CLIs that skip AI start faster
        from google.genai import types

        self.model = model
        self._types = types
        self._max_attempts = max(1, max_attempts)
        self._deadline_s = deadline_s  # one call never keeps a patient waiting longer than this
        self._sleep = sleep
        self._clock = clock
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000), httpx_client=http_client),
        )

    def generate(
        self, *, system: str, prompt: str, schema: type[T], model: str | None = None
    ) -> T:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=to_gemini_schema(schema),
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        bad_outputs = 0
        started = self._clock()
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=model or self.model, contents=prompt, config=config
                )
                return _parse_response(response, schema)
            except LLMError as err:
                error = err
            except Exception as exc:  # SDK, HTTP and network failures all end up here
                error = classify_error(exc)

            if error.kind == "bad_output":
                bad_outputs += 1  # one re-ask is worth it; more rarely helps
            if not error.retryable or attempt == self._max_attempts or bad_outputs > 1:
                raise error
            delay = error.retry_after if error.retry_after is not None else min(2.0**attempt, 20.0)
            if self._clock() - started + delay > self._deadline_s:
                raise error  # waiting any longer would leave a patient staring at a spinner
            log.warning("Gemini call failed (%s); retry %d in %.0fs", error.kind, attempt, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises


def _parse_response(response: Any, schema: type[T]) -> T:
    try:
        text = response.text
    except Exception:  # the SDK can raise when there is no usable candidate
        text = None
    if not text or not text.strip():
        reason = _block_reason(response)
        if reason:
            raise LLMError(f"Gemini declined to process this text ({reason}).", kind="blocked")
        raise LLMError("Gemini returned an empty answer.", kind="bad_output")
    try:
        return schema.model_validate_json(_strip_json_fence(text))
    except ValidationError as exc:
        raise LLMError("Gemini answered in an unexpected format.", kind="bad_output") from exc


def _block_reason(response: Any) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    if reason:
        return str(getattr(reason, "name", reason))
    for candidate in getattr(response, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None)
        name = str(getattr(finish, "name", finish) or "")
        if name and name not in {"STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"}:
            return name
    return None


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


# --------------------------------------------------------------------------- helpers
def create_llm(settings: Settings) -> GeminiLLM:
    if not settings.gemini_api_key:
        raise LLMError(
            f"No Gemini API key is configured. Create a free key at {API_KEY_URL} and set "
            "GEMINI_API_KEY (see .env.example).",
            kind="auth",
        )
    return GeminiLLM(settings.gemini_api_key, settings.model, timeout_s=settings.llm_timeout_s)


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
        raise LLMError("Gemini answered, but not as expected.", kind="bad_output")
