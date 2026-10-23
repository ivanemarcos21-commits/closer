"""Tests for app.py and closer.ui, using Streamlit's headless AppTest runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from closer.clinicaltrials import ClinicalTrialsError, ClinicalTrialsRejectedError
from closer.llm import LLMError
from closer.models import Contact, MatchReport, PatientProfile, TrialMatch
from closer.profile import NoConditionError

APP_FILE = Path(__file__).parent.parent / "app.py"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep the developer's real .env and Streamlit secrets out of the tests."""
    monkeypatch.setattr("closer.config._load_dotenv", lambda: None)
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_MATCH_MODEL", "GROQ_ZERO_RETENTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # so no stray .streamlit/secrets.toml is found
    # The app caches its services per process (including the shared search budget); start clean.
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture()
def with_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def new_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=20)


def find_button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def submit(at: AppTest, story: str = "My uncle has lung cancer.") -> AppTest:
    at.checkbox(key="consent").set_value(True)
    at.text_area(key="story").set_value(story)
    find_button(at, "Find trials").click()
    return at.run()


def page_text(at: AppTest) -> str:
    return " ".join(el.value for el in at.markdown)


def report_for(make_trial, make_site, **match_fields) -> MatchReport:
    trial = make_trial(
        "NCT01234567",
        title="A study of a new lung cancer drug",
        sponsor="Example Pharma",
        sites=[make_site(city="Boston", facility="General Hospital")],
    )
    fields = {"verdict": "YES", "plain_summary": "Tests a new drug.", "reasons_for": ["Stage III"]}
    fields.update(match_fields)
    match = TrialMatch(trial=trial, sites=trial.sites, open_site_count=1, nearby_site_count=1, **fields)
    return MatchReport(
        profile=PatientProfile(summary="A 62-year-old man.", primary_condition="Lung cancer", age_years=62),
        matches=[match],
        total_available=50,
        candidates_checked=1,
    )


# --------------------------------------------------------------------------- the page itself


def test_a_repeated_registry_query_is_remembered_without_pickling():
    from app import CachedRegistry
    from closer.clinicaltrials import SearchPage

    calls = []

    class FakeClient:
        def search(self, **kwargs):
            calls.append(kwargs)
            return SearchPage(trials=[], total_count=3)

    registry = CachedRegistry()
    registry._client = FakeClient()
    assert registry.search(condition="Lung cancer").total_count == 3
    assert registry.search(condition="Lung cancer").total_count == 3
    assert len(calls) == 1


def test_an_ai_error_is_explained_even_if_its_class_was_reloaded():
    class LLMError(Exception):
        def __init__(self) -> None:
            super().__init__("operator detail the visitor must not see")

        @property
        def user_message(self) -> str:
            return "Closer has used up its free AI limit for today. Please try again tomorrow."

    from app import explain

    message = explain(LLMError())
    assert "tomorrow" in message
    assert "operator detail" not in message
    assert "unexpected" not in message


def test_a_validation_error_asks_the_visitor_to_try_again():
    from pydantic import ValidationError

    from app import explain
    from closer.models import PatientProfile

    with pytest.raises(ValidationError) as caught:
        PatientProfile.model_validate({"age_years": "about sixty"})
    message = explain(caught.value)
    assert "couldn't use" in message
    assert "about sixty" not in message


def test_page_renders_with_safety_and_privacy_copy(with_key):
    at = new_app().run()
    assert not at.exception
    assert at.title[0].value == "Find clinical trials that fit your story"
    text = page_text(at)
    assert "not medical advice" in text and "emergency" in text
    assert "names, birth dates, addresses or ID numbers" in text
    assert "up to 30 days" in text  # honest when Zero Data Retention is off
    assert "### Privacy" in text and "### Medical notice" in text
    assert not find_button(at, "Find trials").disabled


def test_the_privacy_notice_changes_if_the_key_is_billed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_ZERO_RETENTION", "true")
    at = new_app().run()
    text = page_text(at)
    assert "does not keep the messages" in text
    assert "Unless Zero Data Retention is turned on" not in text


def test_the_form_asks_about_the_patient_not_the_user(with_key):
    at = new_app().run()
    assert at.number_input(key="age").label == "The patient's age (optional)"
    assert at.selectbox(key="sex").label == "The patient's sex (optional)"
    captions = " ".join(el.value for el in at.caption)
    assert "Enter the patient's details, not yours" in captions and "age in months" in captions


def test_without_an_api_key_searching_is_switched_off_and_the_owner_is_told_why():
    at = new_app().run()
    assert not at.exception
    assert find_button(at, "Find trials").disabled
    assert any("isn't connected" in el.value for el in at.error)


def test_an_example_fills_in_the_story(with_key):
    at = new_app().run()
    at.pills(key="example").set_value("Adult with type 1 diabetes").run()
    assert "type 1 diabetes" in at.text_area(key="story").value
    assert at.pills(key="example").value is None  # ready to be picked again


def test_a_search_without_the_agreement_is_refused(with_key, monkeypatch):
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: pytest.fail("must not search"))
    at = new_app().run()
    at.text_area(key="story").set_value("My uncle has lung cancer.")
    find_button(at, "Find trials").click()
    at = at.run()
    assert any("18 or older" in el.value for el in at.error)
    assert at.session_state.get("report") is None


def test_an_empty_story_asks_for_one_without_searching(with_key, monkeypatch):
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: pytest.fail("must not search"))
    at = submit(new_app().run(), story="   ")
    assert any("describe the situation" in el.value for el in at.error)


# --------------------------------------------------------------------------- results


def test_results_are_shown_with_contacts_and_a_safe_link(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site)
    report.matches[0].trial.central_contacts = [Contact(name="Study Desk", phone="617-555-0100", email="desk@example.org")]
    seen = {}

    def fake_find(story, **kwargs):
        seen.update(kwargs, story=story)
        return report

    monkeypatch.setattr("closer.pipeline.find_trials", fake_find)
    at = new_app().run()
    at.number_input(key="age").set_value(62)
    at.selectbox(key="sex").select("Male")
    at.text_input(key="location").set_value("Boston, MA")
    at = submit(at, "My uncle has lung cancer.")

    assert not at.exception and at.session_state["report"] is report
    assert (seen["age_years"], seen["sex"], seen["location"]) == (62, "MALE", "Boston, MA")
    text = page_text(at)
    assert "Worth asking about (1)" in text
    assert "[A study of a new lung cancer drug](https://clinicaltrials.gov/study/NCT01234567)" in text
    assert "[617-555-0100](tel:6175550100)" in text and "(mailto:desk@example.org)" in text
    assert "General Hospital" in text and "What we understood" in " ".join(el.value for el in at.subheader)
    assert "Before you call" in text and "placebo" in text
    assert "**AI summary:** Tests a new drug." in text


def test_the_counts_say_how_many_studies_the_ai_did_not_read(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site)
    report.total_available = 1299
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    text = page_text(submit(new_app().run()))
    assert "read the 1 most relevant of 1,299 recruiting studies" in text
    assert "The other 1,298 were not read" in text
    assert "https://clinicaltrials.gov/search?cond=Lung+cancer&aggFilters=status:rec" in text


def test_the_green_check_is_not_used_for_a_study_the_ai_merely_could_not_rule_out(with_key, monkeypatch, make_trial, make_site):
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report_for(make_trial, make_site))
    text = page_text(submit(new_app().run()))
    assert ":blue-badge[:material/forum: Worth asking about]" in text
    assert "green-badge" not in text and "Likely match" not in text


def test_a_place_with_no_open_site_is_named_and_other_places_are_still_shown(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site)
    report.profile.location = "Portland, Oregon"
    report.location_relaxed = True
    report.matches[0].nearby_site_count = 0
    report.matches[0].sites = [make_site(city="Seattle", state="Washington")]
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    at = submit(new_app().run())
    info = " ".join(el.value for el in at.info)
    text = page_text(at)
    assert "No recruiting study we found has an open site in Portland, Oregon" in info
    assert "other places" in info
    assert "A study of a new lung cancer drug" in text
    captions = " ".join(el.value for el in at.caption)
    assert "No open site in Portland, Oregon" in captions
    assert "another place" in captions


def test_observational_studies_say_so_and_unhelpful_phase_labels_are_hidden(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site)
    report.matches[0].trial.study_type = "OBSERVATIONAL"
    report.matches[0].trial.phases = ["NA"]
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    text = page_text(submit(new_app().run()))
    assert "Observational: no treatment is given" in text and "Not applicable" not in text


def test_dates_are_readable_and_status_freshness_is_shown(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site)
    trial = report.matches[0].trial
    trial.last_updated, trial.status_verified = "2025-03-12", "2026-02"
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    at = submit(new_app().run())
    captions = " ".join(el.value for el in at.caption)
    assert "Record updated Mar 12, 2025" in captions and "Status confirmed Feb 2026" in captions
    assert "Call to confirm it is still enrolling" in captions
    assert "2025-03-12" not in captions


def test_hostile_registry_and_ai_text_is_rendered_inert(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site, plain_summary="![x](https://evil.test/p.png) costs $5 and $9 www.evil.test")
    report.matches[0].trial.title = "Study [click](http://evil.test) **now**"
    report.matches[0].trial.sponsor = "Visit www.evil.test or mail me@evil.test"
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    at = submit(new_app().run())
    text = page_text(at) + " " + " ".join(el.value for el in at.caption)
    assert "![x](" not in text and "[click](http" not in text and "**now**" not in text
    assert r"\!\[x\]\(`https://evil.test/p.png`\)" in text and r"\$5" in text
    assert "`www.evil.test`" in text and "`me@evil.test`" in text  # shown as code, never clickable


def test_possible_and_unlikely_studies_are_sectioned(with_key, monkeypatch, make_trial, make_site):
    report = report_for(make_trial, make_site, verdict="MAYBE", reasons_for=[], to_confirm=["ECOG status"])
    unlikely = report_for(make_trial, make_site, verdict="NO", reasons_for=[], reasons_against=["Needs EGFR mutation"])
    report.matches.append(unlikely.matches[0].model_copy(update={"trial": make_trial("NCT07654321", title="Other study")}))
    report.more_excluded = 4
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report)
    at = submit(new_app().run())
    text = page_text(at)
    assert "Possible matches (1)" in text and "Confirm with the study team" in text
    assert any("Probably not a fit (1)" in ex.label for ex in at.expander)
    assert "The AI's reason: Needs EGFR mutation" in text
    assert "4 more studies were skipped" in " ".join(el.value for el in at.caption)


def test_start_over_forgets_the_whole_person(with_key, monkeypatch, make_trial, make_site):
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: report_for(make_trial, make_site))
    at = new_app().run()
    at.number_input(key="age").set_value(62)
    at.selectbox(key="sex").select("Male")
    at.text_input(key="location").set_value("Boston, MA")
    at = submit(at)
    assert at.session_state["report"] is not None
    find_button(at, "Start over").click()
    at.run()
    assert not at.exception
    assert "A study of a new lung cancer drug" not in page_text(at)
    assert at.text_area(key="story").value == ""
    assert at.number_input(key="age").value is None  # the next person must not inherit "62, male"
    assert at.selectbox(key="sex").value == "Not specified"
    assert at.text_input(key="location").value == ""
    assert at.checkbox(key="consent").value is False
    assert not [b for b in at.button if b.label == "Start over"]


# --------------------------------------------------------------------------- failures and limits


@pytest.mark.parametrize(
    ("raised", "expected", "forbidden"),
    [
        (LLMError("Check GROQ_API_KEY (free keys: https://console.groq.com/keys).", kind="auth"), "isn't working right now", "GROQ_API_KEY"),
        (LLMError("Set GROQ_MODEL to a model with a higher limit", kind="quota_daily"), "try again tomorrow", "GROQ_MODEL"),
        (LLMError("Groq is receiving too many requests", kind="rate_limit"), "busy right now", "Groq is receiving"),
        (NoConditionError("We could not find a medical condition in that text."), "emergency number", "GROQ"),
        (ClinicalTrialsRejectedError("HTTP 400: Error parsing query SECRET-PATIENT-TEXT"), "plainer words", "SECRET-PATIENT-TEXT"),
        (ClinicalTrialsError("not responding (HTTP 503)"), "couldn't reach ClinicalTrials.gov", "503"),
        (RuntimeError("SECRET-PATIENT-TEXT"), "Something unexpected went wrong", "SECRET-PATIENT-TEXT"),
    ],
)
def test_failures_become_friendly_patient_messages_without_leaking_details(
    with_key, monkeypatch, raised, expected, forbidden
):
    def boom(*a, **k):
        raise raised

    monkeypatch.setattr("closer.pipeline.find_trials", boom)
    at = submit(new_app().run())
    assert not at.exception
    shown = " ".join(el.value for el in at.error)
    assert expected in shown and forbidden not in shown
    assert at.session_state["report"] is None


def test_a_refused_second_click_keeps_the_results_on_screen(with_key, monkeypatch, make_trial, make_site):
    calls = []

    def fake_find(*a, **k):
        calls.append(1)
        return report_for(make_trial, make_site)

    monkeypatch.setattr("closer.pipeline.find_trials", fake_find)
    at = submit(new_app().run())
    at = submit(at)  # immediately again, within the cooldown
    assert len(calls) == 1
    assert any("Please wait" in el.value for el in at.error)
    assert "Worth asking about (1)" in page_text(at)  # the patient's results did not vanish
    assert at.session_state["report"] is not None


def test_personal_details_get_a_nudge_before_anything_is_sent(with_key, monkeypatch, make_trial, make_site):
    calls = []

    def fake_find(*a, **k):
        calls.append(1)
        return report_for(make_trial, make_site)

    monkeypatch.setattr("closer.pipeline.find_trials", fake_find)
    story = "My uncle has lung cancer. Call his daughter on (617) 555-0100."
    at = submit(new_app().run(), story)
    assert calls == []  # nothing has been sent to the AI yet
    assert any("a phone number" in el.value and "remove personal details" in el.value for el in at.error)

    at = submit(at, story)  # pressing it again, unchanged, means "search anyway"
    assert len(calls) == 1 and at.session_state["report"] is not None


def test_editing_the_text_asks_again_if_details_remain(with_key, monkeypatch, make_trial, make_site):
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: pytest.fail("must not search"))
    at = submit(new_app().run(), "Lung cancer. Call 617-555-0100.")
    at = submit(at, "Lung cancer. Call 617-555-0199.")  # changed, but still has a number
    assert any("a phone number" in el.value for el in at.error)


def test_the_app_wide_budget_protects_the_shared_quota(with_key, monkeypatch):
    monkeypatch.setattr("closer.net.SearchBudget.try_acquire", lambda self: False)
    monkeypatch.setattr("closer.pipeline.find_trials", lambda *a, **k: pytest.fail("must not search"))
    at = submit(new_app().run())
    assert any("very busy" in el.value for el in at.error)
    assert "report" not in at.session_state or at.session_state["report"] is None
