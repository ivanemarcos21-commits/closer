"""Shared fixtures and builders for the test-suite."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from closer.clinicaltrials import parse_study
from closer.models import BatchAssessment, Site, Trial, TrialAssessment

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM:
    """A scripted ``StructuredLLM``: records every call and answers through ``handler``.

    ``handler(schema=..., prompt=..., system=..., model=...)`` returns a model instance, or an
    exception to raise. Calls may arrive from several threads, so keep handlers stateless.
    """

    def __init__(self, handler: Callable[..., BaseModel | Exception]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, system: str, prompt: str, schema: type[BaseModel], model: str | None = None):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "model": model})
        result = self.handler(schema=schema, prompt=prompt, system=system, model=model)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture()
def make_llm() -> type[FakeLLM]:
    return FakeLLM


@pytest.fixture()
def verdict_answer() -> Callable[..., BatchAssessment]:
    """Build a BatchAssessment with one assessment per ``<trial id=...>`` found in ``prompt``."""

    def _answer(
        prompt: str, verdicts: dict[str, str] | None = None, default: str = "MAYBE", **fields: Any
    ) -> BatchAssessment:
        verdicts = verdicts or {}
        ids = re.findall(r'<trial id="(NCT\d{8})">', prompt)
        return BatchAssessment(
            assessments=[
                TrialAssessment(
                    nct_id=nct,
                    verdict=verdicts.get(nct, default),
                    plain_summary=f"Plain summary of {nct}.",
                    **fields,
                )
                for nct in ids
            ]
        )

    return _answer


@pytest.fixture(scope="session")
def ctgov_raw() -> dict[str, Any]:
    """A real (trimmed, anonymised) ClinicalTrials.gov v2 response: two recruiting NSCLC studies."""
    return json.loads((FIXTURES / "ctgov_nsclc.json").read_text(encoding="utf-8"))


@pytest.fixture()
def fixture_trials(ctgov_raw: dict[str, Any]) -> list[Trial]:
    return [t for t in (parse_study(s) for s in ctgov_raw["studies"]) if t]


@pytest.fixture()
def make_site() -> Callable[..., Site]:
    def _make(
        city: str | None = "Boston",
        state: str | None = "Massachusetts",
        country: str | None = "United States",
        status: str | None = "RECRUITING",
        facility: str | None = "Test Hospital",
        zip: str | None = None,
        contacts: list | None = None,
    ) -> Site:
        return Site(
            facility=facility,
            city=city,
            state=state,
            country=country,
            status=status,
            zip=zip,
            contacts=contacts or [],
        )

    return _make


@pytest.fixture()
def make_trial() -> Callable[..., Trial]:
    def _make(nct_id: str = "NCT00000001", **overrides: Any) -> Trial:
        data: dict[str, Any] = {
            "title": "A study of a new treatment",
            "status": "RECRUITING",
            "eligibility_text": "Inclusion Criteria:\n* Adults\n\nExclusion Criteria:\n* Pregnancy",
        }
        data.update(overrides)
        return Trial(nct_id=nct_id, **data)

    return _make
