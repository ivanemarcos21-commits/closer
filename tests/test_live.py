"""Opt-in smoke tests against the REAL services.

Skipped by default. After you add your key (see .env.example), run:

    RUN_LIVE=1 python -m pytest -m live -s

They use the live ClinicalTrials.gov API and, for the Gemini tests, a little of your free quota
(about 3 requests in total).
"""

from __future__ import annotations

import os

import pytest

from closer.clinicaltrials import ClinicalTrialsClient
from closer.config import Settings
from closer.llm import check_connection, create_llm
from closer.pipeline import find_trials

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to run live tests"),
]


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.load()


@pytest.fixture()
def needs_key(settings: Settings) -> None:
    if not settings.has_ai:
        pytest.skip("no GROQ_API_KEY configured")


def test_registry_returns_recruiting_studies_with_criteria_and_contacts():
    with ClinicalTrialsClient() as registry:
        page = registry.search(condition="type 1 diabetes", page_size=5)
    assert page.total_count and page.trials
    assert {t.status for t in page.trials} == {"RECRUITING"}
    assert all(t.url for t in page.trials)
    assert any(t.eligibility_text for t in page.trials)
    assert any(t.central_contacts or any(s.contacts for s in t.sites) for t in page.trials)


def test_the_configured_gemini_models_answer_a_structured_request(settings, needs_key):
    llm = create_llm(settings)
    for model in dict.fromkeys([settings.model, settings.effective_match_model]):
        check_connection(llm, model=model)


def test_the_whole_pipeline_works_end_to_end(settings, needs_key):
    story = "My 58 year old mother has glioblastoma that came back after surgery, radiation and temozolomide."
    with ClinicalTrialsClient() as registry:
        report = find_trials(story, llm=create_llm(settings), registry=registry, settings=settings, max_trials=5)
    assert "glioblastoma" in report.profile.primary_condition.lower()
    assert report.profile.age_years == 58 and report.profile.sex == "FEMALE"
    assert report.candidates_checked > 0 and len(report.matches) == report.candidates_checked
    assert not [w for w in report.warnings if "interrupted" in w], report.warnings
