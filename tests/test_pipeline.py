"""Tests for closer.pipeline: candidate gathering, ranking and the end-to-end flow."""

from __future__ import annotations

import pytest

from closer.clinicaltrials import ClinicalTrialsError, ClinicalTrialsRejectedError, SearchPage
from closer.config import Settings
from closer.llm import LLMError
from closer.models import PatientProfile, SearchPlan, TrialMatch
from closer.pipeline import MAX_EXCLUDED_SHOWN, find_trials, gather_candidates, rank_matches
from closer.profile import NoConditionError

SPECIFIC = SearchPlan(condition="Lung cancer", extra_terms="stage III")
BROAD = SearchPlan(condition="Lung cancer")


def nct(i: int) -> str:
    return f"NCT{i:08d}"


class FakeRegistry:
    """Stands in for ClinicalTrialsClient: ``handler(**search_kwargs)`` -> SearchPage or exception."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        result = self.handler(**kwargs)
        if isinstance(result, Exception):
            raise result
        return result


def lung_profile(plans, location=None, **fields) -> PatientProfile:
    return PatientProfile(primary_condition="Lung cancer", location=location, searches=plans, **fields)


# --------------------------------------------------------------------------- candidates


class TestGatherCandidates:
    def test_specific_and_broad_results_are_interleaved_without_duplicates(self, make_trial):
        a, b, c, d = (make_trial(nct(i)) for i in (1, 2, 3, 4))

        def handler(terms=None, **_):
            return SearchPage([a, b], 5) if terms else SearchPage([b, c, d], 99)

        got = gather_candidates(
            FakeRegistry(handler), lung_profile([SPECIFIC, BROAD]), statuses=("RECRUITING",), limit=10
        )
        assert [t.nct_id for t in got.trials] == [nct(1), nct(2), nct(3), nct(4)]
        assert got.total_available == 99  # the broad search's total, not the specific one's
        assert got.location_relaxed is False

    def test_the_limit_is_respected(self, make_trial):
        trials = [make_trial(nct(i)) for i in range(1, 9)]
        got = gather_candidates(
            FakeRegistry(lambda **_: SearchPage(trials, 8)), lung_profile([BROAD]), statuses=("RECRUITING",), limit=3
        )
        assert len(got.trials) == 3

    def test_a_search_plan_the_registry_refuses_is_skipped(self, make_trial):
        def handler(terms=None, **_):
            return ClinicalTrialsRejectedError("bad syntax") if terms else SearchPage([make_trial(nct(1))], 1)

        got = gather_candidates(
            FakeRegistry(handler), lung_profile([SPECIFIC, BROAD]), statuses=("RECRUITING",), limit=5
        )
        assert [t.nct_id for t in got.trials] == [nct(1)]

    def test_the_broad_search_being_refused_is_an_error(self):
        registry = FakeRegistry(lambda **_: ClinicalTrialsRejectedError("bad"))
        with pytest.raises(ClinicalTrialsRejectedError):
            gather_candidates(registry, lung_profile([BROAD]), statuses=("RECRUITING",), limit=5)

    def test_an_outage_is_not_skipped_or_retried_per_plan(self):
        registry = FakeRegistry(lambda **_: ClinicalTrialsError("not responding"))
        with pytest.raises(ClinicalTrialsError, match="not responding"):
            gather_candidates(registry, lung_profile([SPECIFIC, BROAD]), statuses=("RECRUITING",), limit=5)
        assert len(registry.calls) == 1  # fails on the first plan instead of waiting out all three

    def test_when_everything_is_empty_a_free_text_search_is_the_last_resort(self, make_trial):
        def handler(condition=None, terms=None, **_):
            if condition is None and terms == "Lung cancer":
                return SearchPage([make_trial(nct(7))], 1)
            return SearchPage([], 0)

        got = gather_candidates(FakeRegistry(handler), lung_profile([BROAD]), statuses=("RECRUITING",), limit=5)
        assert [t.nct_id for t in got.trials] == [nct(7)]

    def test_statuses_are_passed_to_every_search(self, make_trial):
        registry = FakeRegistry(lambda **_: SearchPage([make_trial(nct(1))], 1))
        gather_candidates(
            registry, lung_profile([SPECIFIC, BROAD]), statuses=("RECRUITING", "NOT_YET_RECRUITING"), limit=4
        )
        assert all(c["statuses"] == ("RECRUITING", "NOT_YET_RECRUITING") and c["page_size"] == 4 for c in registry.calls)


class TestPlaceRanksButNeverFilters:
    """A typed or story-read place used to be a hard registry filter, which silently hid studies."""

    def by_location(self, make_trial):
        a, b, c = (make_trial(nct(i)) for i in (1, 2, 3))

        def handler(location=None, **_):
            return SearchPage([b], 10) if location else SearchPage([a, b, c], 1299)

        return handler

    def test_studies_with_a_site_at_the_place_come_first_and_nothing_is_lost(self, make_trial):
        got = gather_candidates(
            FakeRegistry(self.by_location(make_trial)),
            lung_profile([BROAD], location="Boston, MA"),
            statuses=("RECRUITING",),
            limit=10,
        )
        assert [t.nct_id for t in got.trials] == [nct(2), nct(1), nct(3)]
        assert got.location_relaxed is False
        assert got.total_available == 1299  # the count for the condition anywhere, not the filtered count

    def test_a_study_whose_site_at_the_place_has_closed_does_not_outrank_one_with_an_open_site(
        self, make_trial, make_site
    ):
        closed_only = make_trial(nct(1), sites=[make_site(city="Boston", status="COMPLETED")])
        open_there = make_trial(nct(2), sites=[make_site(city="Boston", status="RECRUITING")])
        elsewhere = make_trial(nct(3), sites=[make_site(city="Austin", state="Texas")])

        def handler(location=None, **_):
            # The registry matches closed sites too, so it lists the closed-only study first.
            return SearchPage([closed_only, open_there], 2) if location else SearchPage([elsewhere], 3)

        got = gather_candidates(
            FakeRegistry(handler),
            lung_profile([BROAD], location="Boston, MA"),
            statuses=("RECRUITING",),
            limit=10,
        )
        assert [t.nct_id for t in got.trials] == [nct(2), nct(1), nct(3)]

    def test_the_search_runs_everywhere_and_again_at_the_place(self, make_trial):
        registry = FakeRegistry(self.by_location(make_trial))
        gather_candidates(
            registry, lung_profile([BROAD], location="Boston, MA"), statuses=("RECRUITING",), limit=10
        )
        assert [c["location"] for c in registry.calls] == [None, "Boston, MA"]

    def test_the_place_is_cleaned_up_before_it_reaches_the_registry(self, make_trial):
        registry = FakeRegistry(self.by_location(make_trial))
        gather_candidates(
            registry, lung_profile([BROAD], location="Portland, OR, USA"), statuses=("RECRUITING",), limit=10
        )
        assert registry.calls[1]["location"] == "Portland, Oregon, United States"

    def test_a_place_with_no_open_study_is_reported_not_hidden(self, make_trial):
        def handler(location=None, **_):
            return SearchPage([], 0) if location else SearchPage([make_trial(nct(1))], 40)

        got = gather_candidates(
            FakeRegistry(handler), lung_profile([BROAD], location="Smalltown"), statuses=("RECRUITING",), limit=5
        )
        assert got.location_relaxed and [t.nct_id for t in got.trials] == [nct(1)]

    def test_a_closed_site_at_the_place_still_counts_as_no_open_study(self, make_trial, make_site):
        closed = make_trial(nct(1), sites=[make_site(city="Portland", state="Oregon", status="COMPLETED")])
        elsewhere = make_trial(nct(2), sites=[make_site(city="Seattle", state="Washington")])

        def handler(location=None, **_):
            return SearchPage([closed], 1) if location else SearchPage([elsewhere], 8)

        got = gather_candidates(
            FakeRegistry(handler),
            lung_profile([BROAD], location="Portland, Oregon"),
            statuses=("RECRUITING",),
            limit=5,
        )
        assert got.location_relaxed
        assert [t.nct_id for t in got.trials] == [nct(1), nct(2)]

    @pytest.mark.parametrize(
        "failure", [ClinicalTrialsRejectedError("bad place"), ClinicalTrialsError("not responding")]
    )
    def test_a_place_search_that_fails_never_fails_the_whole_search(self, make_trial, failure):
        def handler(location=None, **_):
            return failure if location else SearchPage([make_trial(nct(1))], 40)

        got = gather_candidates(
            FakeRegistry(handler), lung_profile([BROAD], location="???"), statuses=("RECRUITING",), limit=5
        )
        assert [t.nct_id for t in got.trials] == [nct(1)]
        assert got.location_relaxed is False  # we do not know the place has no studies; we only failed to ask


# --------------------------------------------------------------------------- ranking


def test_ranking_puts_yes_first_and_near_studies_ahead_within_a_verdict(make_trial):
    def m(i: int, verdict: str, nearby: int = 0) -> TrialMatch:
        return TrialMatch(trial=make_trial(nct(i)), verdict=verdict, nearby_site_count=nearby)

    matches = [m(1, "NO"), m(2, "MAYBE", 0), m(3, "YES", 0), m(4, "MAYBE", 2), m(5, "YES", 1)]
    ranked = rank_matches(matches, has_location=True)
    assert [x.trial.nct_id for x in ranked] == [nct(5), nct(3), nct(4), nct(2), nct(1)]


def test_without_a_location_the_registrys_own_order_is_kept_within_a_verdict(make_trial):
    matches = [TrialMatch(trial=make_trial(nct(i)), verdict="MAYBE") for i in (3, 1, 2)]
    assert [x.trial.nct_id for x in rank_matches(matches, has_location=False)] == [nct(3), nct(1), nct(2)]


# --------------------------------------------------------------------------- end to end


@pytest.fixture()
def settings() -> Settings:
    return Settings(groq_api_key="test", model="profile-model", match_model="match-model", max_workers=2)


def scripted_llm(make_llm, verdict_answer, profile, verdicts=None):
    def handler(schema, prompt, **_):
        if schema is PatientProfile:
            return profile
        return verdict_answer(prompt, verdicts or {}, default="MAYBE")

    return make_llm(handler)


def test_full_run_with_real_trial_data(settings, fixture_trials, make_llm, verdict_answer):
    profile = PatientProfile(
        summary="62-year-old man, stage III NSCLC",
        primary_condition="Non-small cell lung cancer",
        age_years=30,  # the story says 30 but the user typed 62 below: the typed value must win
        searches=[SearchPlan(condition="Non-small cell lung cancer")],
    )
    registry = FakeRegistry(lambda **_: SearchPage(fixture_trials, 114))
    progress: list[str] = []

    def with_reasons(schema, prompt, **_):
        if schema is PatientProfile:
            return profile
        return verdict_answer(prompt, {fixture_trials[0].nct_id: "YES"}, reasons_for=["Stage III NSCLC"])

    llm = make_llm(with_reasons)
    report = find_trials(
        "My uncle has stage 3 lung cancer.",
        llm=llm,
        registry=registry,
        settings=settings,
        age_years=62,
        sex="MALE",
        location="Boston, MA",
        on_progress=progress.append,
    )

    assert report.profile.age_years == 62 and report.profile.sex == "MALE"
    assert report.total_available == 114 and report.candidates_checked == 2 and report.ai_checked == 2
    assert [m.verdict for m in report.matches] == ["YES", "MAYBE"]
    assert report.matches[0].trial.nct_id == fixture_trials[0].nct_id
    assert report.matches[0].nearby_site_count == 2  # the two Boston sites in the real record
    # The typed age differs from the age the story gave, and the user is told which one was used.
    assert any("You entered age 62, but your description said 30" in w for w in report.warnings)
    # one call to read the story, one batched call to judge both trials, each with its own model
    assert [c["model"] for c in llm.calls] == ["profile-model", "match-model"]
    assert [c["location"] for c in registry.calls] == [None, "Boston, MA"]
    assert progress[0] == "Reading the story..." and any("Searching ClinicalTrials.gov" in p for p in progress)


def test_opening_soon_and_max_trials_reach_the_registry(settings, fixture_trials, make_llm, verdict_answer):
    registry = FakeRegistry(lambda **_: SearchPage(fixture_trials[:1], 1))
    find_trials(
        "story",
        llm=scripted_llm(make_llm, verdict_answer, lung_profile([BROAD])),
        registry=registry,
        settings=settings,
        include_opening_soon=True,
        max_trials=7,
    )
    assert registry.calls[0]["statuses"] == ("RECRUITING", "NOT_YET_RECRUITING")
    assert registry.calls[0]["page_size"] == 21  # fetch 3x what the AI will read


def test_studies_excluded_by_age_do_not_use_up_the_ais_reading_slots(settings, make_trial, make_llm, verdict_answer):
    adults_only = [make_trial(nct(i), min_age="18 Years", min_age_years=18) for i in range(1, 7)]
    open_to_teens = [make_trial(nct(i)) for i in range(7, 10)]
    registry = FakeRegistry(lambda **_: SearchPage(adults_only + open_to_teens, 500))
    teen = lung_profile([BROAD], age_years=14)
    llm = scripted_llm(make_llm, verdict_answer, teen)

    report = find_trials("story", llm=llm, registry=registry, settings=settings, max_trials=3)

    judging_prompt = "\n".join(call["prompt"] for call in llm.calls if "trial id=" in call["prompt"])
    assert {nct(i) for i in range(7, 10)} == set(
        n for n in (nct(i) for i in range(1, 10)) if n in judging_prompt
    )  # all three eligible studies were read, none of the six excluded ones
    assert report.ai_checked == 3
    assert sum(1 for m in report.matches if m.source == "rules") == 6  # but they are still listed
    assert report.more_excluded == 0


def test_a_flood_of_excluded_studies_is_counted_not_all_listed(settings, make_trial, make_llm, verdict_answer):
    excluded = [make_trial(nct(i), min_age="18 Years", min_age_years=18) for i in range(1, 1 + MAX_EXCLUDED_SHOWN + 5)]
    registry = FakeRegistry(lambda **_: SearchPage(excluded + [make_trial(nct(99))], 700))
    llm = scripted_llm(make_llm, verdict_answer, lung_profile([BROAD], age_years=14))

    report = find_trials("story", llm=llm, registry=registry, settings=settings, max_trials=20)

    assert sum(1 for m in report.matches if m.source == "rules") == MAX_EXCLUDED_SHOWN
    assert report.more_excluded == 5


def test_typed_sex_that_contradicts_the_story_is_called_out(settings, fixture_trials, make_llm, verdict_answer):
    profile = lung_profile([BROAD], sex="MALE")
    report = find_trials(
        "my father...",
        llm=scripted_llm(make_llm, verdict_answer, profile),
        registry=FakeRegistry(lambda **_: SearchPage(fixture_trials[:1], 1)),
        settings=settings,
        sex="FEMALE",
    )
    assert any("You chose female, but your description suggested male" in w for w in report.warnings)


def test_no_candidates_means_an_empty_report_and_no_judging_calls(settings, make_llm, verdict_answer):
    llm = scripted_llm(make_llm, verdict_answer, lung_profile([BROAD]))
    report = find_trials(
        "story", llm=llm, registry=FakeRegistry(lambda **_: SearchPage([], 0)), settings=settings
    )
    assert report.matches == [] and report.candidates_checked == 0
    assert len(llm.calls) == 1  # only the profile call


def test_the_story_is_cut_to_the_configured_length(fixture_trials, make_llm, verdict_answer):
    short = Settings(groq_api_key="k", story_max_chars=50)
    llm = scripted_llm(make_llm, verdict_answer, lung_profile([BROAD]))
    find_trials(
        "a" * 500,
        llm=llm,
        registry=FakeRegistry(lambda **_: SearchPage(fixture_trials[:1], 1)),
        settings=short,
    )
    assert llm.calls[0]["prompt"] == "<patient_story>\n" + "a" * 50 + "\n</patient_story>"


def test_a_judging_failure_degrades_gracefully(settings, fixture_trials, make_llm):
    def handler(schema, **_):
        return lung_profile([BROAD]) if schema is PatientProfile else LLMError("busy", kind="rate_limit")

    report = find_trials(
        "story",
        llm=make_llm(handler),
        registry=FakeRegistry(lambda **_: SearchPage(fixture_trials, 2)),
        settings=settings,
    )
    assert len(report.matches) == 2 and {m.source for m in report.matches} == {"unchecked"}
    assert report.warnings and report.ai_checked == 0


def test_failing_to_read_the_story_is_an_error_the_caller_can_show(settings, make_llm):
    llm = make_llm(lambda **_: LLMError("bad key", kind="auth"))
    with pytest.raises(LLMError, match="bad key"):
        find_trials("story", llm=llm, registry=FakeRegistry(lambda **_: SearchPage([], 0)), settings=settings)


def test_a_story_without_a_condition_never_touches_the_registry(settings, make_llm):
    registry = FakeRegistry(lambda **_: SearchPage([], 0))
    llm = make_llm(lambda **_: PatientProfile(primary_condition="", searches=[]))
    with pytest.raises(NoConditionError):
        find_trials("hello", llm=llm, registry=registry, settings=settings)
    assert registry.calls == []
