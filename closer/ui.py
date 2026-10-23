"""Streamlit rendering for the results page.

Every string that comes from the registry or the model passes through ``escape_md`` (or
``escape_link_text`` inside a link) before it reaches Markdown, and links are built only from
validated pieces (see ``closer.text``), so a hostile trial record or a manipulated model reply
cannot inject links, images or math.
"""

from __future__ import annotations

from urllib.parse import quote_plus

import streamlit as st

from .display import (
    format_age_range,
    format_date,
    format_phase,
    format_role,
    format_sex,
    format_status,
    site_place,
)
from .legal import BEFORE_YOU_CALL
from .models import Contact, MatchReport, PatientProfile, Site, TrialMatch
from .text import (
    collapse_whitespace,
    escape_link_text,
    escape_md,
    mailto_url,
    shorten,
    tel_url,
)

# Invented composite cases, only here to show how the tool reads a story.
EXAMPLES: dict[str, str] = {
    "Lung cancer, first chemo failed": (
        "My uncle is 62 and was diagnosed with stage 3 non-small cell lung cancer last year. "
        "His first round of chemotherapy didn't work and the latest scan shows the tumor grew. "
        "He is still active and walks every day."
    ),
    "Teen with type 1 diabetes": (
        "My 14-year-old daughter was diagnosed with type 1 diabetes four months ago. She uses an "
        "insulin pump and we would like to find studies that try to protect the insulin her body "
        "still makes."
    ),
    "Glioblastoma came back": (
        "My mother is 58. She had surgery, radiation and temozolomide for glioblastoma 14 months "
        "ago, and her latest MRI shows the tumor has come back. She is in good spirits and can "
        "take care of herself."
    ),
}

# Neutral wording and colours: a green check would read as "approved".
_VERDICT_STYLE = {
    "YES": {"label": "Worth asking about", "icon": ":material/forum:", "color": "blue"},
    "MAYBE": {"label": "Possible match", "icon": ":material/help:", "color": "orange"},
    "NO": {"label": "Probably not a fit", "icon": ":material/block:", "color": "gray"},
}


# --------------------------------------------------------------------------- markdown builders
def bullets(items: list[str]) -> str:
    return "\n".join(f"- {escape_md(item)}" for item in items)


def contact_markdown(contact: Contact) -> str:
    """'Name (Role) - 555-0100 - name@site.org', with call/e-mail links when they are valid."""
    name = escape_md(contact.name) if contact.name else "Contact"
    role = format_role(contact.role)
    parts = [f"{name} ({escape_md(role)})" if role else name]
    if contact.phone:
        link = tel_url(contact.phone)
        text = escape_link_text(contact.phone)
        parts.append(f"[{text}]({link})" if link else escape_md(contact.phone))
    if contact.email:
        link = mailto_url(contact.email)
        text = escape_link_text(contact.email)
        parts.append(f"[{text}]({link})" if link else escape_md(contact.email))
    return " · ".join(parts)


def site_markdown(site: Site) -> str:
    place = escape_md(site_place(site))
    head = f"**{escape_md(site.facility)}**" if site.facility else ""
    line = " · ".join(part for part in (head, place) if part) or "Location not listed"
    lines = [f":material/location_on: {line}"]
    lines += [f"Contact: {contact_markdown(c)}" for c in site.contacts[:2]]
    return "  \n".join(lines)  # two spaces + newline = a line break inside one paragraph


def _badges(match: TrialMatch) -> str:
    style = _VERDICT_STYLE[match.verdict]
    trial = match.trial
    parts = [f":{style['color']}-badge[{style['icon']} {style['label']}]"]
    status_color = "gray" if trial.status == "RECRUITING" else "orange"
    parts.append(f":{status_color}-badge[{escape_md(format_status(trial.status))}]")
    if trial.study_type == "OBSERVATIONAL":
        parts.append(":violet-badge[:material/visibility: Observational: no treatment is given]")
    phase = "" if set(trial.phases) <= {"NA"} else format_phase(trial.phases)
    if phase:
        parts.append(f":gray-badge[{escape_md(phase)}]")
    if match.source == "unchecked":
        parts.append(":orange-badge[:material/warning: Not checked by the AI]")
    return " ".join(parts)


def all_studies_url(condition: str) -> str:
    """The registry's own search page for recruiting studies of a condition."""
    return f"https://clinicaltrials.gov/search?cond={quote_plus(condition)}&aggFilters=status:rec"


# --------------------------------------------------------------------------- pieces of the page
def render_profile(profile: PatientProfile) -> None:
    """Show what the AI understood, so a misread story is caught before anyone trusts results."""
    facts = [
        ("Condition", profile.primary_condition),
        ("Stage or severity", profile.stage_or_severity),
        ("Age", str(profile.age_years) if profile.age_years is not None else ""),
        ("Sex", profile.sex.capitalize() if profile.sex != "UNKNOWN" else ""),
        ("Tried so far", "; ".join(profile.prior_treatments)),
        ("Biomarkers", "; ".join(profile.biomarkers)),
        ("Other conditions", "; ".join(profile.other_conditions)),
        ("Looking near", profile.location or ""),
    ]
    with st.container(border=True):
        st.subheader("What we understood", anchor=False)
        if profile.summary:
            st.markdown(escape_md(profile.summary))
        for label, value in facts:
            if value:
                st.markdown(f"**{label}**  \n{escape_md(value)}")
        if profile.missing_info:
            st.markdown(
                "Adding these would improve the matches: "
                + escape_md("; ".join(profile.missing_info))
                + "."
            )
        st.caption("Something look wrong? Edit your description above and search again.")


def _render_where(match: TrialMatch, location: str | None) -> None:
    trial = match.trial
    if match.sites:
        st.markdown("**Sites listed as open**")
        if location and match.nearby_site_count == 0:
            st.caption(
                f"No open site in {escape_md(location)}. This study is in another place:"
            )
        for site in match.sites:
            st.markdown(site_markdown(site))
        extra = match.open_site_count - len(match.sites)
        if extra > 0:
            # No leading "+": Markdown would turn it into a bullet.
            st.caption(f"Plus {extra} more open site{'s' if extra != 1 else ''} on ClinicalTrials.gov")
    else:
        st.caption(
            "This listing doesn't name an open site. Open the ClinicalTrials.gov page for "
            "contact details."
        )
    for contact in trial.central_contacts[:2]:
        st.markdown(f":material/support_agent: **Study contact:** {contact_markdown(contact)}")


def render_trial_card(match: TrialMatch, location: str | None) -> None:
    trial = match.trial
    with st.container(border=True):
        st.markdown(_badges(match))
        title = escape_link_text(shorten(trial.title, 200))  # the full title is one tap away
        st.markdown(f"##### [{title}]({trial.url})" if trial.url else f"##### {title}")

        meta = [trial.nct_id]
        if trial.sponsor:
            meta.append(trial.sponsor)
        meta += [f"Ages: {format_age_range(trial)}", format_sex(trial.sex)]
        st.caption(escape_md(" · ".join(meta)))
        freshness = []
        if trial.last_updated:
            freshness.append(f"Record updated {format_date(trial.last_updated)}")
        if trial.status_verified:
            freshness.append(f"Status confirmed {format_date(trial.status_verified)}")
        if freshness:
            st.caption(escape_md(" · ".join(freshness)) + " · Call to confirm it is still enrolling")

        if match.plain_summary:
            st.markdown(f"**AI summary:** {escape_md(match.plain_summary)}")
        elif trial.summary:
            st.markdown(
                "**From the registry:** " + escape_md(shorten(collapse_whitespace(trial.summary), 400))
            )

        blocks = [
            ("Why it may fit (the AI's reading)", match.reasons_for),
            ("Confirm with the study team", match.to_confirm),
        ]
        blocks = [(heading, items) for heading, items in blocks if items]
        for heading, items in blocks:
            st.markdown(f"**{heading}**\n\n{bullets(items)}")

        st.markdown("**Who to call**")
        _render_where(match, location)

        with st.expander("Read the full eligibility criteria"):
            if trial.eligibility_text:
                st.caption("Exactly as published on ClinicalTrials.gov")
                st.code(trial.eligibility_text, language=None, wrap_lines=True, height=320)
            else:
                st.caption("No eligibility criteria were published for this study.")
        if trial.url:
            st.link_button("View on ClinicalTrials.gov", trial.url, icon=":material/open_in_new:")


def _render_unlikely(matches: list[TrialMatch], more_excluded: int) -> None:
    with st.expander(f"Probably not a fit ({len(matches)})"):
        st.markdown(
            "These looked like a poor fit for what we understood (see above). The AI can be wrong, "
            "or a detail in your description may have been misread. If you're unsure, ask your "
            "doctor about any of them."
        )
        for match in matches:
            trial = match.trial
            title = escape_link_text(shorten(trial.title, 200))
            st.markdown(f"**[{title}]({trial.url})**" if trial.url else f"**{title}**")
            if match.reasons_against:
                prefix = "Study rule: " if match.source == "rules" else "The AI's reason: "
                st.markdown("\n".join(f"- {prefix}{escape_md(r)}" for r in match.reasons_against))
        if more_excluded:
            st.caption(
                f"{more_excluded} more studies were skipped because their age or sex limits "
                "don't fit. They aren't listed."
            )


def _render_counts(report: MatchReport) -> None:
    condition = report.profile.primary_condition
    ai, total = report.ai_checked, report.total_available
    if ai == 0:
        st.markdown(
            f"Found {report.candidates_checked} recruiting studies for {escape_md(condition)}, "
            "but the AI could not read them (see the note above)."
        )
    elif total and total > ai:
        st.markdown(
            f"Closer's AI read the {ai} most relevant of {total:,} recruiting studies for "
            f"{escape_md(condition)}. The other {total - ai:,} were not read. "
            f"[See them all on ClinicalTrials.gov]({all_studies_url(condition)})"
        )
    else:
        st.markdown(
            f"Closer's AI read {ai} recruiting {'study' if ai == 1 else 'studies'} for "
            f"{escape_md(condition)}."
        )
    st.markdown("The AI compared each study's published rules with what you wrote. It can be wrong in either direction.")


def render_report(report: MatchReport) -> None:
    render_profile(report.profile)
    location = report.profile.location

    for warning in report.warnings:
        st.warning(escape_md(warning), icon=":material/warning:")
    if report.location_relaxed and location:
        st.info(
            f"No recruiting study we found has an open site in {escape_md(location)}. "
            "The studies below are in other places, in case traveling is possible. "
            "Call and ask whether they accept patients from elsewhere. A site can be missing "
            "from the registry, so this is not proof that no study exists there.",
            icon=":material/travel_explore:",
        )

    worth, possible, unlikely = (report.by_verdict(v) for v in ("YES", "MAYBE", "NO"))
    if not report.matches:
        st.info(
            "We couldn't find recruiting studies for that description. That doesn't mean none "
            "exist. Try the diagnosis in plainer words, clear the place box, or tick **Include "
            "studies that open soon** under More options. You can also ask your doctor to search "
            "ClinicalTrials.gov with you.",
            icon=":material/search_off:",
        )
        return

    st.subheader("Studies that may fit", anchor=False)
    _render_counts(report)
    if worth or possible:
        with st.container(border=True):
            st.markdown(BEFORE_YOU_CALL)

    if worth:
        st.markdown(f"#### Worth asking about ({len(worth)})")
        st.markdown("Nothing in your description ruled these out. Only the study team can say if you qualify.")
        for match in worth:
            render_trial_card(match, location)
    if possible:
        st.markdown(f"#### Possible matches ({len(possible)})")
        st.markdown(
            "We can't tell yet, usually because something wasn't in your description. Each card "
            "lists what to ask the study team."
        )
        for match in possible:
            render_trial_card(match, location)
    if not worth and not possible:
        n = report.ai_checked
        st.info(
            f"None of the {n} studies the AI read looked like a fit. That doesn't mean no study "
            f"could help: Closer reads only a few of the studies listed. Add more detail, clear "
            "the place box, or ask your doctor to search ClinicalTrials.gov with you.",
            icon=":material/tune:",
        )
    if unlikely:
        _render_unlikely(unlikely, report.more_excluded)
