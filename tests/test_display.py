"""Tests for closer.display: the plain-text labels shown to people."""

from __future__ import annotations

import pytest

from closer.display import (
    contact_summary,
    format_age_range,
    format_date,
    format_phase,
    format_role,
    format_sex,
    format_status,
    site_label,
)
from closer.models import Contact, Site


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("RECRUITING", "Recruiting"),
        ("NOT_YET_RECRUITING", "Not yet recruiting"),
        ("ACTIVE_NOT_RECRUITING", "Active, not recruiting"),
        ("SOMETHING_NEW", "Something new"),  # unknown statuses still read well
        (None, "Status unknown"),
    ],
)
def test_format_status(raw, label):
    assert format_status(raw) == label


def test_format_phase_joins_combined_phases():
    assert format_phase(["PHASE2", "PHASE3"]) == "Phase 2 / Phase 3"
    assert format_phase(["EARLY_PHASE1"]) == "Early phase 1"
    assert format_phase([]) == ""


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [
        ("18 Years", "65 Years", "18 years to 65 years"),
        ("18 Years", None, "18 years and older"),
        (None, "17 Years", "up to 17 years"),
        ("6 Months", None, "6 months and older"),
        (None, None, "all ages"),
    ],
)
def test_format_age_range(make_trial, low, high, label):
    assert format_age_range(make_trial(min_age=low, max_age=high)) == label


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("2025-03-12", "Mar 12, 2025"),
        ("2026-02", "Feb 2026"),
        ("2026-10-01", "Oct 1, 2026"),
        ("2026-13", "2026-13"),  # not a real month: shown as given rather than guessed at
        ("sometime", "sometime"),
        ("", ""),
        (None, ""),
    ],
)
def test_format_date(raw, label):
    assert format_date(raw) == label


def test_format_sex():
    assert format_sex("FEMALE") == "Women only"
    assert format_sex("MALE") == "Men only"
    assert format_sex("ALL") == "All sexes"
    assert format_sex(None) == "All sexes"


def test_default_contact_role_is_hidden_but_others_are_shown():
    assert format_role("CONTACT") is None
    assert format_role(None) is None
    assert format_role("PRINCIPAL_INVESTIGATOR") == "Principal investigator"
    assert format_role("SOMETHING_ELSE") == "Something else"


def test_site_label_degrades_gracefully():
    full = Site(facility="Mercy Hospital", city="Austin", state="Texas", country="United States")
    assert site_label(full) == "Mercy Hospital - Austin, Texas, United States"
    assert site_label(Site(city="Austin", country="United States")) == "Austin, United States"
    assert site_label(Site(facility="Mercy Hospital")) == "Mercy Hospital"
    assert site_label(Site()) == "Location not listed"


def test_contact_summary():
    contact = Contact(name="Jane Doe", role="STUDY_CHAIR", phone="555-0100", email="jane@example.org")
    assert contact_summary(contact) == "Jane Doe (Study chair) - 555-0100 - jane@example.org"
    assert contact_summary(Contact(phone="555-0100")) == "Study contact - 555-0100"
