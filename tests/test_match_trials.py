"""Tests for match_trials.py: the Phase 2 command-line tool, with Gemini and the registry faked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import match_trials
from closer.clinicaltrials import ClinicalTrialsClient, ClinicalTrialsError, SearchPage
from closer.llm import LLMError
from closer.models import PatientProfile, SearchPlan

PROFILE = PatientProfile(
    summary="62-year-old man with stage III NSCLC.",
    primary_condition="Non-small cell lung cancer",
    stage_or_severity="Stage III",
    age_years=62,
    sex="MALE",
    searches=[SearchPlan(condition="Non-small cell lung cancer")],
    missing_info=["biomarker results"],
)


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr("closer.config._load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MATCH_MODEL", raising=False)


@pytest.fixture()
def fake_services(monkeypatch, make_llm, verdict_answer, fixture_trials):
    """Install a scripted Gemini and a registry that returns the real fixture trials."""
    llm = make_llm(
        lambda schema, prompt, **_: PROFILE
        if schema is PatientProfile
        else verdict_answer(
            prompt, {fixture_trials[0].nct_id: "YES"}, reasons_for=["Stage III NSCLC"], to_confirm=["ECOG status"]
        )
    )
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)
    monkeypatch.setattr(
        ClinicalTrialsClient, "search", lambda self, **kw: SearchPage(fixture_trials, 114)
    )
    return llm


def test_a_story_produces_a_grouped_readable_report(fake_services, capsys):
    code = match_trials.main(["--story", "My uncle has stage 3 lung cancer.", "--location", "Boston, MA"])
    out = capsys.readouterr().out
    assert code == 0
    assert "WHAT WE UNDERSTOOD" in out and "Main condition: Non-small cell lung cancer" in out
    assert "Would help to know: biomarker results" in out
    assert "=== WORTH ASKING ABOUT: 1 ===" in out and "=== POSSIBLE MATCH (needs confirmation): 1 ===" in out
    assert "NCT06855771" in out and "Check with the study team: ECOG status" in out
    assert "Site: Dana-Farber Cancer Institute" in out
    assert "https://clinicaltrials.gov/study/NCT06855771" in out
    assert "not medical advice" in out


def test_progress_goes_to_stderr_so_output_can_be_piped(fake_services, capsys):
    match_trials.main(["--story", "lung cancer"])
    captured = capsys.readouterr()
    assert "Reading the story..." in captured.err and "Reading the story..." not in captured.out


def test_age_and_sex_flags_override_the_story(fake_services):
    match_trials.main(["--story", "lung cancer", "--age", "70", "--sex", "female"])
    judging_prompt = fake_services.calls[-1]["prompt"]
    assert "Age (years): 70" in judging_prompt and "Sex: FEMALE" in judging_prompt


def test_the_story_can_come_from_a_file(fake_services, tmp_path, capsys):
    story = tmp_path / "story.txt"
    story.write_text("He has lung cancer.\n", encoding="utf-8")
    assert match_trials.main(["--story-file", str(story)]) == 0
    assert "He has lung cancer." in fake_services.calls[0]["prompt"]


def test_a_pasted_block_of_criteria_gets_a_single_verdict(monkeypatch, make_llm, verdict_answer, tmp_path, capsys):
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("Inclusion: stage III NSCLC.\nExclusion: prior immunotherapy.", encoding="utf-8")
    llm = make_llm(
        lambda schema, prompt, **_: PROFILE
        if schema is PatientProfile
        else verdict_answer(prompt, {"NCT00000000": "YES"}, reasons_for=["Stage III NSCLC"])
    )
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)
    code = match_trials.main(["--story", "62 year old man, stage 3 lung cancer", "--criteria-file", str(criteria)])
    out = capsys.readouterr().out
    assert code == 0 and "VERDICT: YES  (WORTH ASKING ABOUT)" in out and "Why it may fit: Stage III NSCLC" in out
    assert "Exclusion: prior immunotherapy." in llm.calls[-1]["prompt"]


def test_a_criteria_check_the_ai_could_not_do_is_an_error_not_an_answer(monkeypatch, make_llm, tmp_path, capsys):
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("Inclusion: stage III NSCLC.", encoding="utf-8")
    llm = make_llm(
        lambda schema, **_: PROFILE if schema is PatientProfile else LLMError("quota", kind="quota_daily")
    )
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)
    code = match_trials.main(["--story", "lung cancer", "--criteria-file", str(criteria)])
    captured = capsys.readouterr()
    assert code == 1 and "VERDICT" not in captured.out
    assert "free AI limit" in captured.err


def test_check_tests_both_configured_models(monkeypatch, make_llm, capsys):
    monkeypatch.setenv("GEMINI_MODEL", "small-model")
    monkeypatch.setenv("GEMINI_MATCH_MODEL", "big-model")
    llm = make_llm(lambda **_: SimpleNamespace(ok=True))
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)
    assert match_trials.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "model 'small-model'" in out and "model 'big-model'" in out
    assert [c["model"] for c in llm.calls] == ["small-model", "big-model"]


def test_check_explains_a_rejected_key(monkeypatch, make_llm, capsys):
    llm = make_llm(lambda **_: LLMError("Gemini rejected the API key.", kind="auth"))
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)
    assert match_trials.main(["--check"]) == 1
    assert "Gemini rejected the API key." in capsys.readouterr().err


def test_no_key_gives_setup_instructions(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert match_trials.main(["--check"]) == 1
    assert "aistudio.google.com/apikey" in capsys.readouterr().err


def test_asking_for_nothing_is_a_usage_error(capsys):
    assert match_trials.main([]) == 2
    assert "--story" in capsys.readouterr().err


def test_registry_outages_exit_cleanly(monkeypatch, make_llm, capsys):
    llm = make_llm(lambda **_: PROFILE)
    monkeypatch.setattr(match_trials, "create_llm", lambda settings: llm)

    def down(self, **kwargs):
        raise ClinicalTrialsError("ClinicalTrials.gov is not responding (HTTP 503).")

    monkeypatch.setattr(ClinicalTrialsClient, "search", down)
    assert match_trials.main(["--story", "lung cancer"]) == 1
    assert "not responding" in capsys.readouterr().err
