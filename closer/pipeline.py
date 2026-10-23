"""The whole search in one function: story -> profile -> candidate trials -> verdicts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .clinicaltrials import (
    MAX_PAGE_SIZE,
    OPEN_STATUSES,
    OPENING_SOON_STATUSES,
    ClinicalTrialsError,
    ClinicalTrialsRejectedError,
    SearchPage,
    registry_location,
    screen_demographics,
    select_sites,
)
from .config import Settings
from .llm import StructuredLLM
from .matcher import assess_trials
from .models import MatchReport, PatientProfile, SearchPlan, Trial, TrialMatch
from .profile import apply_overrides, extract_profile

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]
_VERDICT_ORDER = {"YES": 0, "MAYBE": 1, "NO": 2}

FETCH_FACTOR = 3  # fetch this many times what the AI will read, so rule exclusions are free
MAX_EXCLUDED_SHOWN = 15  # age/sex-excluded studies listed in the report (the rest are counted)


class Registry(Protocol):
    """What the pipeline needs from the trial registry (``ClinicalTrialsClient`` fits, and so
    does the app's cached wrapper around it)."""

    def search(
        self,
        *,
        condition: str | None = None,
        terms: str | None = None,
        location: str | None = None,
        statuses: Sequence[str] = ...,
        page_size: int = ...,
    ) -> SearchPage: ...


@dataclass(frozen=True)
class Candidates:
    trials: list[Trial]  # studies with an open site at the place first, then the rest
    total_available: int | None  # recruiting studies for the condition, anywhere
    location_relaxed: bool  # a place was given but no study had an open site there


def _run_plans(
    registry: Registry,
    plans: Sequence[SearchPlan],
    location: str | None,
    statuses: Sequence[str],
    limit: int,
) -> tuple[list[list[Trial]], int | None]:
    """Run each plan; the last (broad) plan supplies the total and must succeed."""
    lists: list[list[Trial]] = []
    total: int | None = None
    for index, plan in enumerate(plans):
        is_broad = index == len(plans) - 1
        try:
            page = registry.search(
                condition=plan.condition,
                terms=plan.extra_terms,
                location=location,
                statuses=statuses,
                page_size=limit,
            )
        except ClinicalTrialsRejectedError:
            if is_broad:
                raise
            # The model wrote search syntax the registry refuses; the broad plan still covers it.
            log.warning("Skipping a specific search plan that the registry rejected")
            lists.append([])
            continue
        lists.append(page.trials)
        if is_broad:
            total = page.total_count
    return lists, total


def _interleave(lists: Sequence[Sequence[Trial]], limit: int) -> list[Trial]:
    """Round-robin merge, so specific plans lead without crowding out the broad one."""
    merged: dict[str, Trial] = {}
    for rank in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if rank < len(items):
                merged.setdefault(items[rank].nct_id, items[rank])
                if len(merged) >= limit:
                    return list(merged.values())
    return list(merged.values())


def gather_candidates(
    registry: Registry,
    profile: PatientProfile,
    *,
    statuses: Sequence[str],
    limit: int,
) -> Candidates:
    """Find candidate studies; ``limit`` is how many to return (callers ask for extra).

    The place only RANKS studies: the search is run everywhere, and again restricted to the
    place, and studies with an open site there come first. A typed (or story-read) place can
    therefore never hide a study, and a place the registry cannot parse costs nothing.
    """
    everywhere, total = _run_plans(registry, profile.searches, None, statuses, limit)

    place = registry_location(profile.location)
    local: list[list[Trial]] = []
    place_known = False
    if place:
        try:
            local, _ = _run_plans(registry, profile.searches, place, statuses, limit)
            place_known = True
        except ClinicalTrialsError:  # an odd place or a hiccup: fall back to the unranked list
            log.warning("The place-restricted search failed; showing studies from everywhere")
            local = []

    ordered: dict[str, Trial] = {}
    for trial in [*_interleave(local, limit), *_interleave(everywhere, limit)]:
        ordered.setdefault(trial.nct_id, trial)

    if not ordered:  # last resort: plain-text search instead of the condition index
        page = registry.search(terms=profile.primary_condition, statuses=statuses, page_size=limit)
        for trial in page.trials:
            ordered.setdefault(trial.nct_id, trial)
        total = page.total_count if total is None else total

    candidates = list(ordered.values())
    if place:
        # The registry also matches studies whose site at the place has since closed. Lead with
        # studies that really have an OPEN site there (a stable sort keeps relevance order), so
        # they get the AI's limited reading slots first.
        candidates.sort(key=lambda t: 0 if select_sites(t, profile.location).nearby_count else 1)

    shown = candidates[:limit]
    # An empty location search means nothing was listed there. A search that only comes back
    # with closed sites, or sites in some other place, means the same thing. Studies with no
    # site list at all are left alone: we cannot tell where they run.
    relaxed = False
    if place_known:
        local_trials = [trial for group in local for trial in group]
        saw_sites = any(trial.sites for trial in local_trials)
        open_there = any(select_sites(trial, profile.location).nearby_count for trial in shown)
        relaxed = (not local_trials) or (saw_sites and not open_there)

    return Candidates(trials=shown, total_available=total, location_relaxed=relaxed)


def rank_matches(matches: Sequence[TrialMatch], *, has_location: bool) -> list[TrialMatch]:
    """YES first, then MAYBE, then NO; within a verdict, studies with an open site nearby lead."""

    def sort_key(item: tuple[int, TrialMatch]) -> tuple[int, int, int]:
        index, match = item
        far_away = has_location and match.nearby_site_count == 0
        return (_VERDICT_ORDER[match.verdict], 1 if far_away else 0, index)

    return [match for _, match in sorted(enumerate(matches), key=sort_key)]


def _disagreements(
    story_age: int | None,
    story_sex: str,
    typed_age: int | None,
    typed_sex: str | None,
) -> list[str]:
    """Tell the user when what they typed overrode what the AI read in their story."""
    notes = []
    if typed_age is not None and story_age is not None and typed_age != story_age:
        notes.append(
            f"You entered age {typed_age}, but your description said {story_age}. "
            f"Closer used {typed_age}."
        )
    if typed_sex in ("FEMALE", "MALE") and story_sex in ("FEMALE", "MALE") and typed_sex != story_sex:
        notes.append(
            f"You chose {typed_sex.lower()}, but your description suggested {story_sex.lower()}. "
            f"Closer used {typed_sex.lower()}."
        )
    return notes


def find_trials(
    story: str,
    *,
    llm: StructuredLLM,
    registry: Registry,
    settings: Settings,
    age_years: int | None = None,
    sex: str | None = None,
    location: str | None = None,
    include_opening_soon: bool = False,
    max_trials: int | None = None,
    on_progress: ProgressFn | None = None,
) -> MatchReport:
    """Run the full pipeline.

    Raises ``LLMError`` if the story cannot be read, ``NoConditionError`` if it names no
    condition, and ``ClinicalTrialsError`` if the registry is unreachable. A failure while
    *judging* trials is not fatal: those trials are returned unjudged, with a warning.
    """
    progress = on_progress or (lambda _message: None)
    limit = max(1, max_trials or settings.max_trials)
    statuses = OPENING_SOON_STATUSES if include_opening_soon else OPEN_STATUSES
    story = story.strip()[: settings.story_max_chars]

    progress("Reading the story...")
    from_story = extract_profile(llm, story, model=settings.model)
    profile = apply_overrides(from_story, age_years=age_years, sex=sex, location=location)
    warnings = _disagreements(from_story.age_years, from_story.sex, age_years, sex)

    progress(f"Searching ClinicalTrials.gov for {profile.primary_condition}...")
    candidates = gather_candidates(
        registry, profile, statuses=statuses, limit=min(limit * FETCH_FACTOR, MAX_PAGE_SIZE)
    )

    # Age and sex limits are free to check, so only studies that pass use up one of the
    # `limit` places the AI reads; the excluded ones are listed (up to a cap) without AI cost.
    allowed: list[Trial] = []
    excluded: list[Trial] = []
    for trial in candidates.trials:
        bucket = excluded if screen_demographics(trial, profile.age_years, profile.sex) else allowed
        bucket.append(trial)
    to_assess = [*allowed[:limit], *excluded[:MAX_EXCLUDED_SHOWN]]
    more_excluded = max(0, len(excluded) - MAX_EXCLUDED_SHOWN)

    if to_assess:
        progress(f"Checking eligibility for {len(allowed[:limit])} studies...")
        outcome = assess_trials(
            llm,
            profile,
            story,
            to_assess,
            location=profile.location,
            batch_size=settings.batch_size,
            max_workers=settings.max_workers,
            model=settings.effective_match_model,
            on_progress=progress,
        )
        matches = rank_matches(outcome.matches, has_location=bool(profile.location))
        warnings = [*warnings, *outcome.warnings]
    else:
        matches = []

    return MatchReport(
        profile=profile,
        matches=matches,
        total_available=candidates.total_available,
        candidates_checked=len(candidates.trials),
        more_excluded=more_excluded,
        location_relaxed=candidates.location_relaxed,
        warnings=warnings,
    )
