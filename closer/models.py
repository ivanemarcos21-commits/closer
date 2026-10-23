"""Data models shared by the whole pipeline.

Two groups live here:

* Registry models (``Trial``, ``Site``, ``Contact``): what we read from ClinicalTrials.gov.
* AI-facing models (``PatientProfile``, ``TrialAssessment`` ...): the JSON contracts we ask
  Gemini to fill in. Their ``Field(description=...)`` texts are part of the prompt, so edit
  them with the same care as any other prompt. Array size limits are given to the model as
  schema hints only (``maxItems``): our own validation stays lenient and trims afterwards, so
  one extra bullet never invalidates a whole batch of answers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .text import clinicaltrials_url

Verdict = Literal["YES", "MAYBE", "NO"]
Sex = Literal["FEMALE", "MALE", "UNKNOWN"]


# --------------------------------------------------------------------------- registry data
class Contact(BaseModel):
    name: str | None = None
    role: str | None = None
    phone: str | None = None
    email: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.phone or self.email)


class Site(BaseModel):
    """One place a study runs. ``status`` is the *site's* status, not the study's."""

    facility: str | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    country: str | None = None
    status: str | None = None
    contacts: list[Contact] = Field(default_factory=list)


class Trial(BaseModel):
    nct_id: str
    title: str
    official_title: str | None = None
    status: str = "UNKNOWN"
    status_verified: str | None = None  # when the sponsor last confirmed the status, "2026-02"
    summary: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    study_type: str | None = None  # INTERVENTIONAL | OBSERVATIONAL | EXPANDED_ACCESS
    sponsor: str | None = None
    last_updated: str | None = None
    eligibility_text: str = ""
    sex: str = "ALL"  # ALL | FEMALE | MALE
    gender_based: bool | None = None  # eligibility depends on gender identity, not only sex
    gender_description: str | None = None
    min_age: str | None = None  # as published, e.g. "18 Years"
    max_age: str | None = None
    min_age_years: float | None = None
    max_age_years: float | None = None
    healthy_volunteers: bool | None = None
    central_contacts: list[Contact] = Field(default_factory=list)
    sites: list[Site] = Field(default_factory=list)

    @property
    def url(self) -> str | None:
        return clinicaltrials_url(self.nct_id)


# --------------------------------------------------------------------------- AI-facing data
class SearchPlan(BaseModel):
    """One ClinicalTrials.gov query. Several plans run from most to least specific."""

    condition: str = Field(
        description=(
            "Disease or condition name only, as ClinicalTrials.gov would list it, "
            "e.g. 'Non-small cell lung cancer'."
        )
    )
    extra_terms: str | None = Field(
        default=None,
        description=(
            "Optional boolean search expression (AND / OR / parentheses) over stage, biomarkers "
            "or treatment setting, e.g. '(stage III OR locally advanced) AND chemotherapy'. "
            "null for a broad search. Never include age, sex or place names."
        ),
    )


class PatientProfile(BaseModel):
    """What the patient's story says, in clinical terms. Unknown stays unknown."""

    summary: str = Field(
        default="",
        description="One or two neutral sentences summarising the patient's situation in clinical terms.",
    )
    primary_condition: str = Field(
        default="",
        description="The main disease, named as ClinicalTrials.gov would, e.g. 'Non-small cell lung cancer'.",
    )
    stage_or_severity: str | None = Field(
        default=None, description="Stage, grade or severity if stated (e.g. 'Stage III'), else null."
    )
    biomarkers: list[str] = Field(
        default_factory=list, description="Genetic or lab markers explicitly mentioned, else []."
    )
    prior_treatments: list[str] = Field(
        default_factory=list,
        description=(
            "Treatments already tried, in medical terms, including the outcome if stated "
            "(e.g. 'first-line platinum chemotherapy - disease progressed'). [] if none mentioned."
        ),
    )
    other_conditions: list[str] = Field(
        default_factory=list, description="Other health conditions or medicines mentioned, else []."
    )
    performance_status: str | None = Field(
        default=None,
        description="ECOG/Karnofsky or a plain description of daily functioning if stated, else null.",
    )
    age_years: int | None = Field(
        default=None,
        description=(
            "The patient's exact CURRENT age in whole years, only if the text states it "
            "(not an age at diagnosis, not a guess from 'in his 60s'). null otherwise, and null "
            "for a child under 2 (put that age in the summary instead)."
        ),
    )
    sex: Sex = Field(
        default="UNKNOWN",
        description=(
            "FEMALE or MALE only from words about the patient (mother, husband, daughter, he, "
            "she). Never from a first name. Otherwise UNKNOWN."
        ),
    )
    location: str | None = Field(
        default=None,
        description="City, state or country where the patient lives or can travel, else null.",
    )
    searches: list[SearchPlan] = Field(
        default_factory=list,
        json_schema_extra={"minItems": 1, "maxItems": 3},
        description=(
            "1 to 3 ClinicalTrials.gov search plans, most specific first. The LAST plan must be "
            "the broad one: just the condition, with extra_terms null."
        ),
    )
    missing_info: list[str] = Field(
        default_factory=list,
        json_schema_extra={"maxItems": 4},
        description=(
            "Up to 4 facts that are NOT in the story but would most improve trial matching "
            "(e.g. 'cancer stage', 'biomarker test results'). [] if nothing important is missing."
        ),
    )

    def describe(self) -> str:
        """Human-readable block for prompts; absent facts are stated as 'not stated'."""

        def line(label: str, value: object) -> str:
            if isinstance(value, list):
                value = "; ".join(value)
            missing = value is None or value == ""  # note: an age of 0 is a real value
            return f"{label}: {'not stated' if missing else value}"

        return "\n".join(
            [
                line("Summary", self.summary),
                line("Main condition", self.primary_condition),
                line("Stage / severity", self.stage_or_severity),
                line("Biomarkers", self.biomarkers),
                line("Prior treatments", self.prior_treatments),
                line("Other conditions", self.other_conditions),
                line("Performance status", self.performance_status),
                line("Age (years)", self.age_years),
                line("Sex", None if self.sex == "UNKNOWN" else self.sex),
                line("Location", self.location),
            ]
        )


class TrialAssessment(BaseModel):
    """The model's verdict for one trial.

    Field order matters: the model writes its answer top to bottom, so the reasoning comes
    first and the verdict last, which keeps the two consistent.
    """

    nct_id: str = Field(description="The trial's NCT number, copied exactly from its id attribute.")
    plain_summary: str = Field(
        description=(
            "One or two sentences at about an 8th-grade reading level saying what this study "
            "tests and who it is for. Explain any medical term you use."
        )
    )
    reasons_for: list[str] = Field(
        default_factory=list,
        json_schema_extra={"maxItems": 3},
        description="Up to 3 short phrases: criteria the patient appears to meet.",
    )
    reasons_against: list[str] = Field(
        default_factory=list,
        json_schema_extra={"maxItems": 3},
        description=(
            "Up to 3 short phrases: criteria the patient's stated facts clearly conflict with, "
            "each quoting the criterion. Required when verdict is NO."
        ),
    )
    to_confirm: list[str] = Field(
        default_factory=list,
        json_schema_extra={"maxItems": 3},
        description="Up to 3 specific facts the study team must verify (labs, biomarkers, prior therapy details...).",
    )
    verdict: Verdict = Field(
        description=(
            "Decide last. YES = the patient appears to meet the key criteria and nothing clearly "
            "excludes them. NO = a criterion clearly rules the patient out. MAYBE = important "
            "criteria cannot be checked from the information given."
        )
    )


class BatchAssessment(BaseModel):
    assessments: list[TrialAssessment] = Field(
        description="Exactly one assessment per trial in the request, in the same order."
    )


# --------------------------------------------------------------------------- results
class TrialMatch(BaseModel):
    trial: Trial
    verdict: Verdict
    # "ai" = Gemini judged it; "rules" = excluded by an age/sex rule (no AI call);
    # "unchecked" = the AI step failed, so the patient must read the criteria themselves.
    source: Literal["ai", "rules", "unchecked"] = "ai"
    plain_summary: str | None = None
    reasons_for: list[str] = Field(default_factory=list)
    reasons_against: list[str] = Field(default_factory=list)
    to_confirm: list[str] = Field(default_factory=list)
    sites: list[Site] = Field(default_factory=list)  # sites worth showing, best match first
    open_site_count: int = 0
    nearby_site_count: int = 0  # open sites that match the whole requested place


class MatchReport(BaseModel):
    profile: PatientProfile
    matches: list[TrialMatch] = Field(default_factory=list)
    total_available: int | None = None  # recruiting studies for the condition, anywhere
    candidates_checked: int = 0  # studies fetched and screened (by rules or by the AI)
    more_excluded: int = 0  # age/sex-excluded studies not listed, to keep the page short
    location_relaxed: bool = False  # a place was given but no study had an open site there
    warnings: list[str] = Field(default_factory=list)

    @property
    def ai_checked(self) -> int:
        """How many studies the AI actually read."""
        return sum(1 for m in self.matches if m.source == "ai")

    def by_verdict(self, verdict: Verdict) -> list[TrialMatch]:
        return [m for m in self.matches if m.verdict == verdict]
