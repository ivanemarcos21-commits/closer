#!/usr/bin/env python3
"""Phase 1: pull live, recruiting studies from ClinicalTrials.gov and print them.

No AI and no API key needed - this just proves the data pipeline works.

Examples:
    python fetch_trials.py "non-small cell lung cancer"
    python fetch_trials.py "type 1 diabetes" --location "Houston, TX" --limit 5
    python fetch_trials.py glioblastoma --terms "recurrent" --opening-soon
    python fetch_trials.py "breast cancer" --json > trials.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from closer.clinicaltrials import (
    OPEN_STATUSES,
    OPENING_SOON_STATUSES,
    ClinicalTrialsClient,
    ClinicalTrialsError,
    select_sites,
)
from closer.display import (
    contact_summary,
    format_age_range,
    format_phase,
    format_sex,
    format_status,
    site_label,
)
from closer.models import Trial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch live, recruiting clinical studies from ClinicalTrials.gov."
    )
    parser.add_argument("condition", help='Condition to search for, e.g. "non-small cell lung cancer".')
    parser.add_argument(
        "--terms", help='Extra words that must also match, e.g. "(stage III OR locally advanced)".'
    )
    parser.add_argument("--location", help='City, state, ZIP or country, e.g. "Boston, MA".')
    parser.add_argument(
        "--limit", type=int, default=10, help="How many studies to fetch (1-100, default 10)."
    )
    parser.add_argument(
        "--opening-soon",
        action="store_true",
        help="Also include studies that have not started recruiting yet.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the parsed studies as JSON instead of text."
    )
    return parser


def print_trial(index: int, trial: Trial, location: str | None) -> None:
    phase = format_phase(trial.phases)
    print(f"{index:>2}. {trial.nct_id}  [{format_status(trial.status)}]" + (f"  {phase}" if phase else ""))
    print(f"    {trial.title}")
    facts = [f"Ages: {format_age_range(trial)}", format_sex(trial.sex)]
    if trial.sponsor:
        facts.append(f"Sponsor: {trial.sponsor}")
    print("    " + "  |  ".join(facts))
    if trial.conditions:
        print("    Conditions: " + "; ".join(trial.conditions[:4]))

    picked = select_sites(trial, location, limit=2)
    summary = f"{picked.open_count} open site(s)"
    if location:
        summary += f", {picked.nearby_count} matching '{location}'"
    print(f"    Sites: {summary}")
    for site in picked.sites:
        print(f"      - {site_label(site)}")
    for contact in trial.central_contacts[:1]:
        print(f"    Contact: {contact_summary(contact)}")
    print(f"    {trial.url}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    statuses = OPENING_SOON_STATUSES if args.opening_soon else OPEN_STATUSES
    try:
        with ClinicalTrialsClient() as client:
            page = client.search(
                condition=args.condition,
                terms=args.terms,
                location=args.location,
                statuses=statuses,
                page_size=args.limit,
            )
    except ClinicalTrialsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([t.model_dump(mode="json") for t in page.trials], indent=2))
        return 0

    total = page.total_count if page.total_count is not None else len(page.trials)
    print(f"{total} matching study(ies) on ClinicalTrials.gov; showing {len(page.trials)}.\n")
    for index, trial in enumerate(page.trials, start=1):
        print_trial(index, trial, args.location)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
