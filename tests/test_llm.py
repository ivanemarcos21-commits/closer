"""Tests for closer.llm: schema conversion, the Gemini wire format, and error handling.

The SDK talks to an ``httpx.MockTransport``, so these tests run the real SDK code path and
check the exact request it would send, without a key or a network.
"""

from __future__ import annotations

import json
import ssl
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from closer.config import Settings
from closer.llm import (
    GroqLLM,
    LLMError,
    check_connection,
    classify_error,
    create_llm,
    to_gemini_schema,
)
from closer.models import BatchAssessment, PatientProfile


class Answer(BaseModel):
    ok: bool


def groq_reply(text: str, finish: str = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def error_reply(message: str) -> dict[str, Any]:
    return {"error": {"message": message}}


def make_groq(handler, **kwargs):
    sleeps: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    llm = GroqLLM("test-key", "test-model", http_client=http, sleep=sleeps.append, **kwargs)
    return llm, sleeps


def ask(llm: GroqLLM, **kwargs) -> Answer:
    return llm.generate(system="sys", prompt="hello", schema=Answer, **kwargs)


# --------------------------------------------------------------------------- schema


def _walk(node: Any):
    """Yield every schema node (dicts), but not the *names* inside a ``properties`` map."""
    if isinstance(node, dict):
        yield node
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    yield from _walk(sub)
            else:
                yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_schema_is_self_contained_and_free_of_unneeded_keywords():
    schema = to_gemini_schema(BatchAssessment)
    for node in _walk(schema):
        assert not {"$ref", "$defs", "title", "default"} & node.keys()
    item = schema["properties"]["assessments"]["items"]
    assert item["properties"]["verdict"]["enum"] == ["YES", "MAYBE", "NO"]


def test_every_property_is_required_so_the_model_fills_each_field():
    schema = to_gemini_schema(BatchAssessment)
    item = schema["properties"]["assessments"]["items"]
    assert item["required"] == list(item["properties"])
    assert schema["required"] == ["assessments"]


def test_profile_schema_keeps_nullability_and_array_bounds():
    props = to_gemini_schema(PatientProfile)["properties"]
    assert {"type": "null"} in props["age_years"]["anyOf"]
    assert {"type": "integer"} in props["age_years"]["anyOf"]
    assert props["searches"]["maxItems"] == 3 and props["searches"]["minItems"] == 1
    assert "description" in props["primary_condition"]  # descriptions steer the model


# --------------------------------------------------------------------------- wire format


def test_request_matches_the_groq_chat_completions_format():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=groq_reply('{"ok": true}'))

    llm, _ = make_groq(handler)
    assert ask(llm) == Answer(ok=True)

    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    schema = body["response_format"]["json_schema"]
    assert schema["name"] == "Answer"
    assert schema["schema"]["properties"]["ok"] == {"type": "boolean"}
    assert body["max_tokens"] == 2048
    assert body["reasoning_effort"] == "low"


def test_a_different_model_can_be_requested_per_call():
    seen: list[httpx.Request] = []
    llm, _ = make_groq(lambda r: (seen.append(r), httpx.Response(200, json=groq_reply('{"ok": true}')))[1])
    ask(llm, model="bigger-model")
    assert json.loads(seen[0].content)["model"] == "bigger-model"


def test_json_wrapped_in_a_code_fence_is_still_parsed():
    llm, _ = make_groq(lambda r: httpx.Response(200, json=groq_reply('```json\n{"ok": true}\n```')))
    assert ask(llm).ok is True


# --------------------------------------------------------------------------- failures


def test_daily_quota_fails_at_once_without_retrying():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json=error_reply("Rate limit reached for tokens per day"))

    llm, sleeps = make_groq(handler)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "quota_daily" and "daily limit" in str(caught.value)
    assert len(calls) == 1 and sleeps == []


def test_per_minute_limit_waits_for_the_delay_google_asks_for():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429, json=error_reply("Rate limit reached"), headers={"retry-after": "7"}
            )
        return httpx.Response(200, json=groq_reply('{"ok": true}'))

    llm, sleeps = make_groq(handler)
    assert ask(llm).ok
    assert sleeps == [7.0]


def test_a_low_token_budget_pauses_before_the_next_call():
    clock = {"now": 0.0}

    def handler(request):
        return httpx.Response(
            200,
            json=groq_reply('{"ok": true}'),
            headers={"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "5s"},
        )

    llm, sleeps = make_groq(handler, clock=lambda: clock["now"])
    assert ask(llm).ok and sleeps == []
    assert ask(llm).ok and sleeps == [5.0]


def test_a_request_bigger_than_the_token_limit_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            413,
            json=error_reply(
                "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 14951"
            ),
        )

    llm, sleeps = make_groq(handler)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "too_large"
    assert "smaller pieces" in str(caught.value)
    assert len(calls) == 1 and sleeps == []


def test_a_json_validation_failure_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                400, json=error_reply("Failed to validate JSON. See 'failed_generation' for more details.")
            )
        return httpx.Response(200, json=groq_reply('{"ok": true}'))

    llm, sleeps = make_groq(handler)
    assert ask(llm).ok
    assert len(calls) == 2 and len(sleeps) == 1


def test_overloaded_service_backs_off_and_then_gives_up():
    llm, sleeps = make_groq(
        lambda r: httpx.Response(503, json=error_reply("overloaded")),
        max_attempts=3,
    )
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "unavailable"
    assert sleeps == [2.0, 4.0]


def test_a_rejected_key_is_reported_with_how_to_fix_it():
    llm, sleeps = make_groq(
        lambda r: httpx.Response(401, json=error_reply("Invalid API Key"))
    )
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "auth" and "GROQ_API_KEY" in str(caught.value)
    assert sleeps == []


def test_an_unknown_model_name_points_to_the_model_list():
    llm, _ = make_groq(
        lambda r: httpx.Response(404, json=error_reply("The model `x` does not exist"))
    )
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "model" and "GROQ_MODEL" in str(caught.value)


def test_a_blocked_prompt_is_reported_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=groq_reply("", finish="content_filter"))

    llm, _ = make_groq(handler)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "blocked" and "content_filter" in str(caught.value)
    assert len(calls) == 1


def test_a_candidate_stopped_for_safety_is_treated_as_blocked():
    reply = groq_reply("", finish="content_filter")
    llm, _ = make_groq(lambda r: httpx.Response(200, json=reply))
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "blocked"


def test_malformed_json_is_asked_for_once_more_and_then_reported():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=groq_reply("this is not json"))

    llm, sleeps = make_groq(handler)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "bad_output"
    assert len(calls) == 2 and len(sleeps) == 1


def test_malformed_json_that_recovers_on_the_second_try_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        text = "oops" if len(calls) == 1 else '{"ok": true}'
        return httpx.Response(200, json=groq_reply(text))

    llm, _ = make_groq(handler)
    assert ask(llm).ok and len(calls) == 2


def test_schema_violations_count_as_bad_output():
    llm, _ = make_groq(lambda r: httpx.Response(200, json=groq_reply('{"ok": "not a bool"}')))
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "bad_output"


def test_network_timeouts_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=groq_reply('{"ok": true}'))

    llm, sleeps = make_groq(handler)
    assert ask(llm).ok and len(sleeps) == 1


def test_retries_stop_once_waiting_would_blow_the_total_deadline():
    clock = {"now": 0.0}
    calls = []

    def handler(request):
        calls.append(1)
        clock["now"] += 100.0  # every attempt "takes" 100 seconds before timing out
        raise httpx.ReadTimeout("slow", request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    llm = GroqLLM(
        "k", "m", http_client=http, sleep=lambda s: None, clock=lambda: clock["now"], deadline_s=120.0
    )
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "network"
    assert len(calls) == 2  # not the 4 attempts x 60 s a patient would otherwise sit through


def test_certificate_problems_fail_fast():
    calls = []

    def handler(request):
        calls.append(1)
        error = httpx.ConnectError("certificate verify failed", request=request)
        error.__cause__ = ssl.SSLCertVerificationError("self-signed certificate")
        raise error

    llm, sleeps = make_groq(handler)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.kind == "certificate" and "security certificate" in str(caught.value)
    assert len(calls) == 1 and sleeps == []


def test_error_messages_never_contain_the_prompt():
    secret = "SECRET-PATIENT-TEXT"
    llm, _ = make_groq(
        lambda r: httpx.Response(400, json=error_reply("bad schema"))
    )
    with pytest.raises(LLMError) as caught:
        llm.generate(system="s", prompt=secret, schema=Answer)
    assert secret not in str(caught.value)


def test_classify_error_handles_unexpected_exceptions():
    error = classify_error(ValueError("boom"))
    assert error.kind == "unknown" and "ValueError" in str(error)
    assert not error.retryable


def test_patients_never_see_setup_instructions():
    # str(error) is for the person running the app; user_message is for the patient.
    operator = LLMError("Check GROQ_API_KEY (free keys: https://console.groq.com/keys).", kind="auth")
    assert "GROQ_API_KEY" in str(operator) and "GROQ_API_KEY" not in operator.user_message
    for kind in ("auth", "model", "certificate", "bad_request", "unknown"):
        message = LLMError("x", kind=kind).user_message
        assert "isn't working right now" in message and "ClinicalTrials.gov" in message
    assert "tomorrow" in LLMError("x", kind="quota_daily").user_message
    assert "busy" in LLMError("x", kind="rate_limit").user_message
    assert "too long" in LLMError("x", kind="too_large").user_message
    assert "plainer words" in LLMError("x", kind="blocked").user_message


def test_a_blocked_answer_is_not_fatal_for_the_other_requests():
    from closer.llm import FATAL_KINDS

    assert "blocked" not in FATAL_KINDS  # it depends on that one request's text
    assert {"auth", "model", "quota_daily", "certificate"} <= FATAL_KINDS


def test_extra_list_items_do_not_invalidate_an_answer():
    # The schema tells the model "at most 3", but our own validation stays lenient.
    from closer.models import TrialAssessment

    item = TrialAssessment(
        nct_id="NCT00000001", plain_summary="s", verdict="MAYBE", reasons_for=["a", "b", "c", "d", "e"]
    )
    assert len(item.reasons_for) == 5
    schema = to_gemini_schema(TrialAssessment)["properties"]
    assert schema["reasons_for"]["maxItems"] == 3  # ...yet the model is still told the bound
    assert list(schema)[-1] == "verdict"  # reasoning is written before the verdict


# --------------------------------------------------------------------------- helpers


def test_create_llm_without_a_key_explains_how_to_get_one():
    with pytest.raises(LLMError, match="GROQ_API_KEY") as caught:
        create_llm(Settings(groq_api_key=None))
    assert caught.value.kind == "auth"


def test_create_llm_returns_a_client_when_a_key_is_present():
    assert create_llm(Settings(groq_api_key="k", model="m")).model == "m"


def test_check_connection_accepts_a_good_answer_and_rejects_a_bad_one(make_llm):
    check_connection(make_llm(lambda **_: Answer(ok=True)))
    with pytest.raises(LLMError):
        check_connection(make_llm(lambda **_: Answer(ok=False)))
