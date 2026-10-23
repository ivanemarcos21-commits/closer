#!/usr/bin/env python3
"""Phase 2: match a patient's story against live trials with Gemini.

Setup: put your free key in a .env file (see .env.example), then try:

    python match_trials.py --check
    python match_trials.py --story "My uncle is 62 and has stage 3 lung cancer. The first \
chemotherapy did not work." --location "Boston, MA"
    python match_trials.py --story-file story.txt --max-trials 10

The simplest test of the AI step: judge ONE pasted block of eligibility criteria.

    python match_trials.py --story "62 year old man, stage 3 lung cancer" --criteria-file criteria.txt
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence

from closer.clinicaltrials import ClinicalTrialsClient, ClinicalTrialsError
from closer.config import Settings
from closer.display import contact_summary, format_status, site_label
from closer.llm import LLMError, check_connection, create_llm
from closer.matcher import assess_trials
from closer.models import MatchReport, PatientProfile, Trial, TrialMatch
from closer.pipeline import find_trials
from closer.profile import NoConditionError, apply_overrides, extract_profile

_LABELS = {
    "YES": "WORTH ASKING ABOUT",
    "MAYBE": "POSSIBLE MATCH (needs confirmation)",
    "NO": "PROBABLY NOT A FIT",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match a patient story to recruiting clinical trials.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--story", help="The patient's situation, in plain English.")
    source.add_argument("--story-file", help="Read the story from a text file ('-' for stdin).")
    parser.add_argument("--age", type=int, help="Patient age in years (overrides the story).")
    parser.add_argument("--sex", choices=["female", "male"], help="Patient sex (overrides the story).")
    parser.add_argument("--location", help='City, state or country, e.g. "Boston, MA".')
    parser.add_argument("--max-trials", type=int, help="How many studies to check (default 20).")
    parser.add_argument("--opening-soon", action="store_true", help="Include studies not yet recruiting.")
    parser.add_argument(
        "--criteria-file", help="Judge this one block of eligibility criteria instead of searching."
    )
    parser.add_argument("--check", action="store_true", help="Test the Gemini key and model, then exit.")
    return parser


def read_story(args: argparse.Namespace) -> str | None:
    if args.story:
        return args.story
    if args.story_file:
        if args.story_file == "-":
            return sys.stdin.read()
        with open(args.story_file, encoding="utf-8") as handle:
            return handle.read()
    return None


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def print_profile(profile: PatientProfile) -> None:
    print("WHAT WE UNDERSTOOD")
    print(textwrap.indent(profile.describe(), "  "))
    if profile.missing_info:
        print("  Would help to know: " + "; ".join(profile.missing_info))
    print()


def print_match(match: TrialMatch, *, brief: bool = False) -> None:
    trial = match.trial
    print(f"  {trial.nct_id}  [{format_status(trial.status)}]  {trial.title}")
    if brief:
        for reason in match.reasons_against:
            print(wrap(f"- {reason}", "      "))
        return
    if match.plain_summary:
        print(wrap(match.plain_summary, "      "))
    for label, items in (
        ("Why it may fit", match.reasons_for),
        ("Check with the study team", match.to_confirm),
    ):
        for item in items:
            print(wrap(f"{label}: {item}", "      "))
    for site in match.sites[:2]:
        print(f"      Site: {site_label(site)}")
    for contact in trial.central_contacts[:1]:
        print(f"      Contact: {contact_summary(contact)}")
    print(f"      {trial.url}\n")


def print_report(report: MatchReport) -> None:
    print_profile(report.profile)
    total = f" (of {report.total_available} recruiting)" if report.total_available else ""
    print(f"Checked {report.candidates_checked} studies{total}.")
    if report.location_relaxed:
        print(f"No recruiting study had an open site in '{report.profile.location}'; searched everywhere.")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    print()
    for verdict in ("YES", "MAYBE", "NO"):
        group = report.by_verdict(verdict)  # type: ignore[arg-type]
        if not group:
            continue
        print(f"=== {_LABELS[verdict]}: {len(group)} ===")
        for match in group:
            print_match(match, brief=verdict == "NO")
        print()
    print("This is information, not medical advice. Only the study team can confirm eligibility.")


def run_check(settings: Settings) -> int:
    llm = create_llm(settings)
    for model in dict.fromkeys([settings.model, settings.effective_match_model]):
        check_connection(llm, model=model)
        print(f"OK: model '{model}' answered a structured request.")
    return 0


def run_criteria_check(settings: Settings, args: argparse.Namespace, story: str) -> int:
    with open(args.criteria_file, encoding="utf-8") as handle:
        criteria = handle.read()
    llm = create_llm(settings)
    profile = apply_overrides(
        extract_profile(llm, story, model=settings.model),
        age_years=args.age,
        sex=args.sex.upper() if args.sex else None,
        location=args.location,
    )
    trial = Trial(nct_id="NCT00000000", title="Pasted eligibility criteria", eligibility_text=criteria)
    outcome = assess_trials(llm, profile, story, [trial], model=settings.effective_match_model)
    print_profile(profile)
    match = outcome.matches[0]
    if match.source == "unchecked":  # the AI could not judge it; do not pass that off as an answer
        for warning in outcome.warnings:
            print(f"Error: {warning}", file=sys.stderr)
        return 1
    print(f"VERDICT: {match.verdict}  ({_LABELS[match.verdict]})")
    for label, items in (
        ("Why it may fit", match.reasons_for),
        ("Why not", match.reasons_against),
        ("To confirm", match.to_confirm),
    ):
        for item in items:
            print(wrap(f"{label}: {item}", "  "))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    try:
        if args.check:
            return run_check(settings)
        story = read_story(args)
        if not story or not story.strip():
            print("Error: give --story, --story-file or --check.", file=sys.stderr)
            return 2
        if args.criteria_file:
            return run_criteria_check(settings, args, story)
        with ClinicalTrialsClient() as registry:
            report = find_trials(
                story,
                llm=create_llm(settings),
                registry=registry,
                settings=settings,
                age_years=args.age,
                sex=args.sex.upper() if args.sex else None,
                location=args.location,
                include_opening_soon=args.opening_soon,
                max_trials=args.max_trials,
                on_progress=lambda message: print(message, file=sys.stderr),
            )
    except (LLMError, ClinicalTrialsError, NoConditionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
