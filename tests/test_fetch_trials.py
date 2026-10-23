"""Tests for fetch_trials.py: the Phase 1 command-line tool."""

from __future__ import annotations

import json

import pytest

import fetch_trials
from closer.clinicaltrials import ClinicalTrialsClient, ClinicalTrialsError, SearchPage


@pytest.fixture()
def registry_returns(monkeypatch, fixture_trials):
    """Make ClinicalTrialsClient.search return the real fixture, recording how it was called."""
    calls: list[dict] = []

    def fake_search(self, **kwargs):
        calls.append(kwargs)
        return SearchPage(fixture_trials, 114)

    monkeypatch.setattr(ClinicalTrialsClient, "search", fake_search)
    return calls


def test_prints_a_readable_summary_of_each_study(registry_returns, capsys):
    code = fetch_trials.main(["non-small cell lung cancer", "--location", "Boston, MA", "--limit", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "114 matching study(ies)" in out and "showing 2" in out
    assert "NCT06855771" in out and "[Recruiting]" in out and "Phase 2" in out
    assert "Dana-Farber Cancer Institute - Boston, Massachusetts, United States" in out
    assert "matching 'Boston, MA'" in out
    assert "https://clinicaltrials.gov/study/NCT06855771" in out
    assert registry_returns[0]["condition"] == "non-small cell lung cancer"
    assert registry_returns[0]["location"] == "Boston, MA"
    assert registry_returns[0]["page_size"] == 2


def test_opening_soon_widens_the_status_filter(registry_returns):
    fetch_trials.main(["asthma", "--opening-soon"])
    assert registry_returns[0]["statuses"] == ("RECRUITING", "NOT_YET_RECRUITING")


def test_json_output_is_valid_and_complete(registry_returns, capsys):
    assert fetch_trials.main(["asthma", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [t["nct_id"] for t in data] == ["NCT06855771", "NCT07063745"]
    assert "eligibility_text" in data[0] and data[0]["sites"]


def test_registry_problems_exit_cleanly_with_a_message(monkeypatch, capsys):
    def boom(self, **kwargs):
        raise ClinicalTrialsError("ClinicalTrials.gov is not responding (HTTP 503).")

    monkeypatch.setattr(ClinicalTrialsClient, "search", boom)
    assert fetch_trials.main(["asthma"]) == 1
    captured = capsys.readouterr()
    assert "not responding" in captured.err and captured.out == ""
