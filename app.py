"""Closer - find clinical trials that fit a patient's story.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (main file: app.py; add GROQ_API_KEY under Secrets).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import streamlit as st

from closer.chrome import install_style, render_footer, render_masthead
from closer.clinicaltrials import (
    OPEN_STATUSES,
    ClinicalTrialsClient,
    ClinicalTrialsError,
    ClinicalTrialsRejectedError,
    SearchPage,
)
from closer.config import Settings
from closer.llm import LLMError, create_llm
from closer.net import SearchBudget
from closer.pipeline import find_trials
from closer.legal import AGREEMENT_REQUIRED, CONSENT_LABEL
from closer.profile import NoConditionError
from closer.text import escape_md, find_personal_details
from closer.ui import EXAMPLES, render_report

log = logging.getLogger("closer.app")
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs full request URLs at INFO

COOLDOWN_SECONDS = 15  # per visitor: stops accidental double searches
_SEX_CHOICES = {"Not specified": None, "Female": "FEMALE", "Male": "MALE"}
_SECRET_NAMES = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_MATCH_MODEL",
    "GROQ_ZERO_RETENTION",
    "OPERATOR_NAME",
    "OPERATOR_EMAIL",
)


# --------------------------------------------------------------------------- services
def load_settings() -> Settings:
    """Environment / .env first, then Streamlit secrets (how Community Cloud passes the key)."""
    try:
        secrets = {name: st.secrets.get(name) for name in _SECRET_NAMES}
    except Exception:  # no secrets file is the normal case on a laptop
        secrets = {}
    return Settings.load(secrets)


@st.cache_resource(show_spinner=False)
def get_registry() -> "CachedRegistry":
    """One shared registry cache per server. Results stay in memory, so nothing is pickled."""
    return CachedRegistry()


@st.cache_resource(show_spinner=False)
def get_llm(settings: Settings):
    return create_llm(settings)


@st.cache_resource(show_spinner=False)
def get_budget(per_hour: int) -> SearchBudget:
    """One budget per server process, shared by every visitor."""
    return SearchBudget(per_hour)


class CachedRegistry:
    """Same ``search`` interface as ``ClinicalTrialsClient``, remembering identical recent queries.

    The cache key is the search terms only (a diagnosis and a place), never the patient's story.
    Results stay on this object. ``st.cache_data`` would pickle them, and that pickle fails for
    these records.
    """

    _TTL_SECONDS = 900
    _MAX_ENTRIES = 256

    def __init__(self) -> None:
        self._client = ClinicalTrialsClient()
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], tuple[float, SearchPage]] = {}

    def search(
        self,
        *,
        condition: str | None = None,
        terms: str | None = None,
        location: str | None = None,
        statuses: Any = OPEN_STATUSES,
        page_size: int = 25,
    ) -> SearchPage:
        key = (condition, terms, location, tuple(statuses), int(page_size))
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and now - hit[0] < self._TTL_SECONDS:
                return hit[1]
        page = self._client.search(
            condition=condition, terms=terms, location=location, statuses=statuses, page_size=page_size
        )
        with self._lock:
            self._cache[key] = (time.monotonic(), page)
            while len(self._cache) > self._MAX_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
        return page


# --------------------------------------------------------------------------- page
@dataclass(frozen=True)
class SearchInputs:
    story: str
    age: int | None
    sex: str | None
    location: str | None
    include_soon: bool
    max_trials: int


def render_header(settings: Settings) -> None:
    render_masthead()
    st.title("Find clinical trials that fit your story", anchor=False)
    st.markdown(
        "Describe the illness in everyday words. Closer looks up recruiting studies on "
        "[ClinicalTrials.gov](https://clinicaltrials.gov) and shows who to call."
    )


def render_setup_help() -> None:
    st.error(
        "The AI helper isn't connected yet, so searching is switched off.",
        icon=":material/key:",
    )
    with st.expander("For whoever runs this app"):
        st.markdown(
            "1. Create a free key at [Groq console](https://console.groq.com/keys).\n"
            "2. **On your computer:** copy `.env.example` to `.env` and set `GROQ_API_KEY`.\n"
            "3. **On Streamlit Community Cloud:** open the app's **Settings → Secrets** and add "
            "`GROQ_API_KEY = \"your-key\"`.\n"
            "4. In Groq, open **Settings → Data Controls** and turn on **Zero Data Retention**.\n"
            "5. Reload this page."
        )


def _apply_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["story"] = EXAMPLES[choice]
        st.session_state["example"] = None  # let the same example be picked again later


def render_form(settings: Settings) -> tuple[bool, SearchInputs]:
    st.pills("Try an example", list(EXAMPLES), key="example", on_change=_apply_example)
    return _search_form(settings)


def _search_form(settings: Settings) -> tuple[bool, SearchInputs]:
    with st.form("search", border=False):
        story = st.text_area(
            "What is going on?",
            key="story",
            height=220,
            max_chars=settings.story_max_chars,
            placeholder=(
                "Example: My uncle is 62 and has stage 3 lung cancer. His first round of "
                "chemotherapy didn't work."
            ),
        )
        st.caption(
            "Include the diagnosis, stage, treatments tried so far, and how the person is doing. "
            "For a baby or toddler, give the age in months. Leave out names and contact details."
        )
        st.markdown("##### About the patient")
        age_col, sex_col = st.columns(2, gap="medium")
        age = age_col.number_input(
            "The patient's age (optional)",
            min_value=0,
            max_value=120,
            value=None,
            step=1,
            placeholder="e.g. 62",
            key="age",
        )
        sex = sex_col.selectbox("The patient's sex (optional)", list(_SEX_CHOICES), key="sex")
        location = st.text_input(
            "Where could the patient go? (optional)",
            placeholder="City, state or country",
            max_chars=100,
            key="location",
        )
        st.caption(
            "Enter the patient's details, not yours. Age and sex are only used to skip studies "
            "with age or sex limits (studies usually mean sex at birth). Studies with an open "
            "site in the city, state, or country you give are shown first. If none are open "
            "there, studies in other places are still shown."
        )
        with st.expander("More options"):
            include_soon = st.checkbox("Include studies that open soon", value=False)
            max_trials = st.select_slider(
                "How many studies should the AI read?",
                options=[10, 20, 30],
                value=10,
                help="10 is the fastest. 20 and 30 read more studies and take longer on the free AI plan.",
            )
        st.checkbox(CONSENT_LABEL, key="consent")
        submitted = st.form_submit_button(
            "Find trials",
            type="primary",
            icon=":material/search:",
            disabled=not settings.has_ai,
        )
    return submitted, SearchInputs(
        story=story or "",
        age=int(age) if age is not None else None,
        sex=_SEX_CHOICES[sex],
        location=(location or "").strip() or None,
        include_soon=include_soon,
        max_trials=int(max_trials),
    )


def explain(exc: BaseException) -> str:
    """A message a worried visitor can act on. Never includes their own text or operator advice."""
    if isinstance(exc, NoConditionError):
        return f"{exc} If you need urgent help, call your local emergency number."
    if isinstance(exc, LLMError) or type(exc).__name__ == "LLMError":
        # Streamlit can reload the app without reloading this class, so isinstance may miss it.
        message = getattr(exc, "user_message", None)
        if isinstance(message, str) and message.strip():
            return message
    if isinstance(exc, ClinicalTrialsRejectedError):
        return (
            "We couldn't search ClinicalTrials.gov with that wording. Try describing the "
            "diagnosis in plainer words."
        )
    if isinstance(exc, ClinicalTrialsError):
        return (
            "We couldn't reach ClinicalTrials.gov just now. Please try again in a moment, or "
            "search it yourself at clinicaltrials.gov."
        )
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, MemoryError)):
        return "The search was interrupted before it finished. Please wait a minute and try again."
    if type(exc).__name__ == "ValidationError":
        return "The AI helper gave an answer Closer couldn't use. Please try again."
    return f"Something unexpected went wrong ({type(exc).__name__}). Please try again in a moment."


def _refuse(message: str) -> None:
    """Show a message without touching any results already on the page."""
    st.session_state["error"] = message


def _show_progress(status: Any, message: str) -> None:
    """Update the status line. A display glitch must not throw away a search that is still running."""
    try:
        status.write(escape_md(message))
    except Exception as exc:
        log.warning("Progress update failed: %s", type(exc).__name__)


def handle_submit(settings: Settings, inputs: SearchInputs) -> None:
    story = inputs.story.strip()
    if not story:
        return _refuse("Please describe the situation first. Even a sentence or two helps.")
    if not st.session_state.get("consent"):
        return _refuse(AGREEMENT_REQUIRED)
    wait = COOLDOWN_SECONDS - (time.monotonic() - st.session_state.get("last_run", -1e9))
    if wait > 0:
        return _refuse(f"Please wait {int(wait) + 1} more seconds before searching again.")

    found = find_personal_details(story)
    if found and st.session_state.get("pii_ok") != story:
        st.session_state["pii_ok"] = story  # an unchanged second press means "search anyway"
        return _refuse(
            f"Your text may include {' and '.join(found)}. Please remove personal details, then "
            "press Find trials again. To search anyway, just press it once more."
        )
    if not get_budget(settings.searches_per_hour).try_acquire():
        return _refuse("Closer is very busy right now. Please try again in a few minutes.")

    st.session_state["report"] = None
    st.session_state["error"] = None
    with st.status(
        "Looking for trials. This can take a minute or two. Please keep this page open.",
        expanded=True,
    ) as status:
        try:
            report = find_trials(
                inputs.story,
                llm=get_llm(settings),
                registry=get_registry(),
                settings=settings,
                age_years=inputs.age,
                sex=inputs.sex,
                location=inputs.location,
                include_opening_soon=inputs.include_soon,
                max_trials=inputs.max_trials,
                on_progress=lambda message: _show_progress(status, message),
            )
        except Exception as exc:  # shown to the visitor as a plain message; logged by kind only
            log.warning("Search failed: %s (%s)", type(exc).__name__, getattr(exc, "kind", "-"))
            status.update(label="We couldn't finish that search", state="error", expanded=False)
            st.session_state["error"] = explain(exc)
        else:
            st.session_state["report"] = report
            status.update(label="Done", state="complete", expanded=False)
        # Not in a `finally`: if this run is interrupted (a double click re-runs the page), the
        # new run must be allowed to start straight away.
        st.session_state["last_run"] = time.monotonic()


def _start_over() -> None:
    """Forget the person completely, including the age, sex and place typed for them."""
    st.session_state["story"] = ""
    st.session_state["age"] = None
    st.session_state["sex"] = "Not specified"
    st.session_state["location"] = ""
    st.session_state["consent"] = False
    for key in ("report", "error", "pii_ok"):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(
        page_title="Closer - find clinical trials that fit",
        page_icon=":material/biotech:",
        layout="centered",
    )
    install_style()
    settings = load_settings()
    render_header(settings)
    if not settings.has_ai:
        render_setup_help()

    submitted, inputs = render_form(settings)
    if submitted:
        handle_submit(settings, inputs)

    if error := st.session_state.get("error"):
        st.error(escape_md(error), icon=":material/error:")
    if report := st.session_state.get("report"):
        render_report(report)
        st.button("Start over", on_click=_start_over, icon=":material/restart_alt:")
    render_footer(settings)


if __name__ == "__main__":
    main()
