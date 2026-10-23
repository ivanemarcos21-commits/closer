"""Plain-text formatting shared by the command-line tools and the web app.

Nothing here imports Streamlit, so the CLIs stay light and these helpers are easy to test.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Contact, Site, Trial

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_STATUS_LABELS = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "COMPLETED": "Completed",
    "SUSPENDED": "Suspended",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "UNKNOWN": "Status unknown",
}

_PHASE_LABELS = {
    "EARLY_PHASE1": "Early phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not applicable",
}

_ROLE_LABELS = {
    "PRINCIPAL_INVESTIGATOR": "Principal investigator",
    "SUB_INVESTIGATOR": "Sub-investigator",
    "STUDY_CHAIR": "Study chair",
    "STUDY_DIRECTOR": "Study director",
}


def format_status(status: str | None) -> str:
    key = (status or "UNKNOWN").upper()
    return _STATUS_LABELS.get(key, key.replace("_", " ").capitalize())


def format_phase(phases: Sequence[str]) -> str:
    labels = [_PHASE_LABELS.get(p.upper(), p.replace("_", " ").capitalize()) for p in phases]
    return " / ".join(labels)


def format_age_range(trial: Trial) -> str:
    low = trial.min_age.lower() if trial.min_age else None
    high = trial.max_age.lower() if trial.max_age else None
    if low and high:
        return f"{low} to {high}"
    if low:
        return f"{low} and older"
    if high:
        return f"up to {high}"
    return "all ages"


def format_date(value: str | None) -> str:
    """"2025-03-12" -> "Mar 12, 2025"; "2026-02" -> "Feb 2026"; anything else is returned as is."""
    if not value:
        return ""
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", value.strip())
    if not match or not 1 <= int(match.group(2)) <= 12:
        return value
    year, month, day = match.groups()
    name = _MONTHS[int(month) - 1]
    return f"{name} {int(day)}, {year}" if day else f"{name} {year}"


def format_sex(sex: str | None) -> str:
    return {"FEMALE": "Women only", "MALE": "Men only"}.get((sex or "ALL").upper(), "All sexes")


def format_role(role: str | None) -> str | None:
    """Readable contact role; the default role ("CONTACT") says nothing, so it is hidden."""
    if not role or role.upper() == "CONTACT":
        return None
    key = role.upper()
    return _ROLE_LABELS.get(key, key.replace("_", " ").capitalize())


def site_place(site: Site) -> str:
    return ", ".join(part for part in (site.city, site.state, site.country) if part)


def site_label(site: Site) -> str:
    place = site_place(site)
    if site.facility and place:
        return f"{site.facility} - {place}"
    return site.facility or place or "Location not listed"


def contact_summary(contact: Contact) -> str:
    """One line such as 'Jane Doe (Study chair) - 555-0100 - jane@example.org'."""
    name = contact.name or "Study contact"
    role = format_role(contact.role)
    head = f"{name} ({role})" if role else name
    return " - ".join(part for part in (head, contact.phone, contact.email) if part)
