"""ClinicalTrials.gov v2 API client and record parser.

The registry is free, needs no API key, and lists studies from all over the world
(docs: https://clinicaltrials.gov/data-api/api). We ask only for the fields the pipeline
uses, which keeps responses small and fast.

Besides talking to the API this module holds the deterministic, no-AI logic that works on
registry data: parsing records, turning a place name into something the registry accepts,
ranking sites against a place, and the age/sex rules.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .models import Contact, Site, Trial
from .net import CERTIFICATE_HELP, is_certificate_error
from .text import NCT_ID_RE

log = logging.getLogger(__name__)

API_BASE = "https://clinicaltrials.gov/api/v2"
USER_AGENT = "closer-trial-finder/0.1"

OPEN_STATUSES: tuple[str, ...] = ("RECRUITING",)
OPENING_SOON_STATUSES: tuple[str, ...] = ("RECRUITING", "NOT_YET_RECRUITING")

# "Piece" names understood by the API's `fields` parameter.
DEFAULT_FIELDS: tuple[str, ...] = (
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "StatusVerifiedDate",
    "BriefSummary",
    "Condition",
    "InterventionName",
    "Phase",
    "StudyType",
    "LeadSponsorName",
    "LastUpdatePostDate",
    "EligibilityCriteria",
    "HealthyVolunteers",
    "Sex",
    "GenderBased",
    "GenderDescription",
    "MinimumAge",
    "MaximumAge",
    "CentralContactName",
    "CentralContactRole",
    "CentralContactPhone",
    "CentralContactPhoneExt",
    "CentralContactEMail",
    "LocationFacility",
    "LocationStatus",
    "LocationCity",
    "LocationState",
    "LocationZip",
    "LocationCountry",
    "LocationContactName",
    "LocationContactRole",
    "LocationContactPhone",
    "LocationContactPhoneExt",
    "LocationContactEMail",
)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_S = 15.0
MAX_PAGE_SIZE = 100


class ClinicalTrialsError(RuntimeError):
    """ClinicalTrials.gov could not be reached, or it rejected the request."""


class ClinicalTrialsRejectedError(ClinicalTrialsError):
    """The registry understood the request and refused it (HTTP 4xx), e.g. a bad query.

    Unlike an outage this will fail the same way every time, so callers may skip that
    query and carry on with others.
    """


@dataclass(frozen=True)
class SearchPage:
    trials: list[Trial]
    total_count: int | None  # how many studies match in total (we only fetch one page)


# --------------------------------------------------------------------------- HTTP client
class ClinicalTrialsClient:
    """Thin, retrying wrapper around ``GET /api/v2/studies``. Safe to share between threads."""

    def __init__(
        self,
        *,
        base_url: str = API_BASE,
        timeout: float = 20.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(0, max_retries)
        self._backoff = backoff
        self._sleep = sleep
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=False,  # the API never redirects; never follow one to another host
        )

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> ClinicalTrialsClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def search(
        self,
        *,
        condition: str | None = None,
        terms: str | None = None,
        location: str | None = None,
        statuses: Sequence[str] = OPEN_STATUSES,
        page_size: int = 25,
        fields: Sequence[str] = DEFAULT_FIELDS,
    ) -> SearchPage:
        """Find studies. ``condition`` and ``terms`` are AND-ed; both accept AND/OR/NOT.

        ``location`` is free text (city, state, ZIP or country); it is cleaned first, see
        ``registry_location``. Results come back in the registry's own relevance order.
        """
        condition = (condition or "").strip()
        terms = (terms or "").strip()
        if not condition and not terms:
            raise ValueError("search() needs a condition or some search terms")

        params: dict[str, Any] = {
            "filter.overallStatus": "|".join(statuses),
            "pageSize": max(1, min(int(page_size), MAX_PAGE_SIZE)),
            "countTotal": "true",
            "fields": "|".join(fields),
        }
        if condition:
            params["query.cond"] = condition
        if terms:
            params["query.term"] = terms
        place = registry_location(location)
        if place:
            params["query.locn"] = place

        data = self._get("/studies", params)
        trials: list[Trial] = []
        for raw in data.get("studies") or []:
            try:
                trial = parse_study(raw)
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                # One oddly shaped record must not sink the whole search.
                log.warning("Skipping a malformed registry record (%s)", type(exc).__name__)
                continue
            if trial:
                trials.append(trial)
        total = data.get("totalCount")
        return SearchPage(trials=trials, total_count=total if isinstance(total, int) else None)

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        problem = "no response"
        for attempt in range(self._max_retries + 1):
            wait: float | None = None
            try:
                response = self._http.get(url, params=params)
            except httpx.HTTPError as exc:  # timeouts, DNS and connection failures
                if is_certificate_error(exc):  # retrying cannot fix this
                    raise ClinicalTrialsError(f"ClinicalTrials.gov: {CERTIFICATE_HELP}") from exc
                problem = type(exc).__name__
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ClinicalTrialsError(
                            "ClinicalTrials.gov sent a response we could not read."
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ClinicalTrialsError("ClinicalTrials.gov sent an unexpected response.")
                    return payload
                if response.status_code not in _RETRY_STATUSES:
                    detail = response.text.strip()[:200]
                    raise ClinicalTrialsRejectedError(
                        f"ClinicalTrials.gov rejected the request (HTTP {response.status_code}): {detail}"
                    )
                problem = f"HTTP {response.status_code}"
                wait = _retry_after_seconds(response)
            if attempt < self._max_retries:
                delay = wait if wait is not None else self._backoff * (2**attempt)
                log.info("ClinicalTrials.gov: %s; retrying in %.1fs", problem, delay)
                self._sleep(delay)
        raise ClinicalTrialsError(
            f"ClinicalTrials.gov is not responding ({problem}). Please try again in a moment."
        )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After", "")
    try:
        return min(max(float(raw), 0.0), _MAX_RETRY_AFTER_S)
    except ValueError:
        return None


# --------------------------------------------------------------------------- parsing
def _text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _text_list(values: Any) -> list[str]:
    """Clean, de-duplicated (order-preserving) list of non-empty strings."""
    if not isinstance(values, list):
        return []
    seen: dict[str, None] = {}
    for value in values:
        cleaned = _text(value)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


_AGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$", re.IGNORECASE
)
_YEARS_PER_UNIT = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 7 / 365.25,
    "day": 1 / 365.25,
    "hour": 1 / 8766,
    "minute": 1 / 525960,
}


def parse_age_years(value: str | None) -> float | None:
    """Convert the registry's age text ("18 Years", "6 Months") to years; None for "N/A"."""
    match = _AGE_RE.match(value or "")
    if not match:
        return None
    return float(match.group(1)) * _YEARS_PER_UNIT[match.group(2).lower()]


def _parse_contact(raw: Any) -> Contact:
    raw = raw if isinstance(raw, Mapping) else {}
    phone = _text(raw.get("phone"))
    extension = _text(raw.get("phoneExt"))
    if phone and extension:
        phone = f"{phone} ext. {extension}"
    return Contact(
        name=_text(raw.get("name")),
        role=_text(raw.get("role")),
        phone=phone,
        email=_text(raw.get("email")),
    )


def _parse_contacts(raw_list: Any) -> list[Contact]:
    contacts = [_parse_contact(raw) for raw in raw_list or [] if isinstance(raw, Mapping)]
    return [c for c in contacts if not c.is_empty]


def _parse_site(raw: Mapping[str, Any]) -> Site:
    return Site(
        facility=_text(raw.get("facility")),
        city=_text(raw.get("city")),
        state=_text(raw.get("state")),
        zip=_text(raw.get("zip")),
        country=_text(raw.get("country")),
        status=_text(raw.get("status")),
        contacts=_parse_contacts(raw.get("contacts")),
    )


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes"}
    return None


def parse_study(raw: Mapping[str, Any]) -> Trial | None:
    """Turn one raw API record into a ``Trial``; ``None`` if it has no valid NCT id.

    Every module is optional in the API (older studies miss many), so nothing here may
    assume a field exists.
    """

    def module(name: str) -> Mapping[str, Any]:
        value = (raw.get("protocolSection") or {}).get(name)
        return value if isinstance(value, Mapping) else {}

    ident = module("identificationModule")
    nct_id = (_text(ident.get("nctId")) or "").upper()
    if not NCT_ID_RE.match(nct_id):
        return None

    eligibility = module("eligibilityModule")
    places = module("contactsLocationsModule")
    status_module = module("statusModule")
    interventions = module("armsInterventionsModule").get("interventions")
    sponsor = module("sponsorCollaboratorsModule").get("leadSponsor")
    min_age = _text(eligibility.get("minimumAge"))
    max_age = _text(eligibility.get("maximumAge"))

    return Trial(
        nct_id=nct_id,
        title=_text(ident.get("briefTitle")) or _text(ident.get("officialTitle")) or nct_id,
        official_title=_text(ident.get("officialTitle")),
        status=(_text(status_module.get("overallStatus")) or "UNKNOWN").upper(),
        status_verified=_text(status_module.get("statusVerifiedDate")),
        summary=_text(module("descriptionModule").get("briefSummary")),
        conditions=_text_list(module("conditionsModule").get("conditions")),
        interventions=_text_list(
            [i.get("name") for i in interventions or [] if isinstance(i, Mapping)]
        ),
        phases=_text_list(module("designModule").get("phases")),
        study_type=_text(module("designModule").get("studyType")),
        sponsor=_text(sponsor.get("name")) if isinstance(sponsor, Mapping) else None,
        last_updated=_text((status_module.get("lastUpdatePostDateStruct") or {}).get("date")),
        eligibility_text=_text(eligibility.get("eligibilityCriteria")) or "",
        sex=(_text(eligibility.get("sex")) or "ALL").upper(),
        gender_based=_parse_bool(eligibility.get("genderBased")),
        gender_description=_text(eligibility.get("genderDescription")),
        min_age=min_age,
        max_age=max_age,
        min_age_years=parse_age_years(min_age),
        max_age_years=parse_age_years(max_age),
        healthy_volunteers=_parse_bool(eligibility.get("healthyVolunteers")),
        central_contacts=_parse_contacts(places.get("centralContacts")),
        sites=[_parse_site(s) for s in places.get("locations") or [] if isinstance(s, Mapping)],
    )


# --------------------------------------------------------------------------- places
US_STATES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}  # fmt: skip
_STATE_NAMES = {name.lower() for name in US_STATES.values()}
COUNTRY_ALIASES = {
    "US": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "UAE": "United Arab Emirates",
}
_PLACE_ALIASES = {
    "nyc": "new york",
    "new york city": "new york",
    "sf": "san francisco",
    "philly": "philadelphia",
}
_QUERY_SYNTAX_RE = re.compile(r'[()\[\]{}"\\]')
_BOOLEAN_WORDS = frozenset({"AND", "OR", "NOT"})


def _split_place(location: str) -> list[str]:
    """Break "Boston, MA" / "Boston MA" / "Houston Texas" into its parts, in the user's words."""
    parts = [p.strip() for p in re.split(r"[,;]", location) if p.strip()]
    if len(parts) == 1:  # no comma: peel a trailing state off "Boston MA" / "Houston Texas"
        words = parts[0].split()
        if len(words) >= 2 and parts[0].lower() not in _STATE_NAMES:
            last = words[-1].rstrip(".")
            last_two = " ".join(words[-2:])
            if len(last) == 2 and last.upper() in US_STATES:
                parts = [" ".join(words[:-1]), last]
            elif len(words) >= 3 and last_two.lower() in _STATE_NAMES:
                parts = [" ".join(words[:-2]), last_two]
            elif last.lower() in _STATE_NAMES:
                parts = [" ".join(words[:-1]), last]
    return parts


def registry_location(location: str | None) -> str | None:
    """Make free text safe for the registry's boolean ``query.locn`` field.

    In that language "OR" (Oregon!) is an operator and brackets or quotes are syntax, so
    "Portland, OR" is rejected outright. And it matches country names literally: "USA" finds
    a handful of studies where "United States" finds hundreds.
    """
    if not location:
        return None
    parts: list[str] = []
    for part in _split_place(_QUERY_SYNTAX_RE.sub(" ", location)):
        if part.upper().replace(".", "") == "OR":  # a bare "OR" is the state of Oregon
            parts.append(US_STATES["OR"])
            continue
        # AND / OR / NOT inside a part are operators to the registry: treat them as separators.
        chunk: list[str] = []
        for word in part.split():
            if word.upper() in _BOOLEAN_WORDS:
                if chunk:
                    parts.append(" ".join(chunk))
                    chunk = []
            else:
                chunk.append(word)
        if chunk:
            text = " ".join(chunk)
            parts.append(COUNTRY_ALIASES.get(text.upper().replace(".", ""), text))
    return ", ".join(parts) or None


def location_tokens(location: str) -> list[str]:
    """Lower-case, de-duplicated match tokens for a place; "MA" becomes "massachusetts"."""
    tokens: list[str] = []
    for part in _split_place(location):
        key = part.upper().replace(".", "")
        expanded = (
            US_STATES.get(key) or COUNTRY_ALIASES.get(key) or _PLACE_ALIASES.get(part.lower()) or part
        )
        token = expanded.lower()
        if token not in tokens:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class SiteScore:
    matched: int  # how many of the place's parts this site satisfies
    points: int  # tie-breaker: a city or ZIP hit counts for more than a state or country


def site_score(site: Site, tokens: Sequence[str]) -> SiteScore:
    city = (site.city or "").lower()
    state = (site.state or "").lower()
    country = (site.country or "").lower()
    postcode = (site.zip or "").lower().replace(" ", "")
    facility = (site.facility or "").lower()
    matched = points = 0
    for token in tokens:
        compact = token.replace(" ", "")
        if token == city or (compact[:1].isdigit() and postcode.startswith(compact)):
            gained = 3
        elif token in (state, country):
            gained = 2
        elif len(token) >= 4 and token in facility:
            gained = 1
        else:
            continue
        matched += 1
        points += gained
    return SiteScore(matched, points)


def site_is_open(site: Site, trial_status: str) -> bool:
    """Is this site (not just the study as a whole) taking patients?

    The registry keeps closed sites on recruiting studies, so the site's own status matters.
    Many older records give no site status; those count as open.
    """
    status = (site.status or "").upper()
    if not status or status == "RECRUITING":
        return True
    return status == "NOT_YET_RECRUITING" and trial_status.upper() == "NOT_YET_RECRUITING"


@dataclass(frozen=True)
class SiteSelection:
    sites: list[Site]  # sites worth showing, best match first
    open_count: int  # all sites currently taking patients
    nearby_count: int  # open sites that match the WHOLE requested place (0 if none requested)


def select_sites(trial: Trial, location: str | None = None, limit: int = 4) -> SiteSelection:
    """Pick the open sites to show: those matching ``location`` first, then those with a contact.

    "Nearby" means every part of the place matched ("Boston, MA" needs Boston *and*
    Massachusetts), so Cambridge, MA or Boston in another country do not count.
    """
    open_sites = [s for s in trial.sites if site_is_open(s, trial.status)]
    tokens = location_tokens(location) if location and location.strip() else []

    def contact_first(index: int, site: Site) -> tuple[int, int]:
        return (0 if site.contacts else 1, index)

    if tokens:
        scored = [(site_score(s, tokens), i, s) for i, s in enumerate(open_sites)]
        hits = sorted(
            (item for item in scored if item[0].matched > 0),
            key=lambda item: (-item[0].matched, -item[0].points, *contact_first(item[1], item[2])),
        )
        if hits:
            nearby = sum(1 for score, _, _ in hits if score.matched == len(tokens))
            return SiteSelection([s for _, _, s in hits[:limit]], len(open_sites), nearby)

    ranked = sorted(enumerate(open_sites), key=lambda pair: contact_first(*pair))
    return SiteSelection([s for _, s in ranked[:limit]], len(open_sites), 0)


# --------------------------------------------------------------------------- hard rules
_MIN_AGE_FOR_RULES = 2.0  # infants are reported in months; leave those to the AI


def screen_demographics(trial: Trial, age_years: float | None, sex: str | None) -> list[str]:
    """Reasons the patient is ruled out by the study's published age or sex limits.

    These are black-and-white registry fields, so no AI call is needed. An empty list means
    "not excluded here", not "eligible". Ages are whole years, so the lower limit only excludes
    when even the end of that year falls short (a 2-year-old may be 2 years 11 months old).
    """
    reasons: list[str] = []
    if age_years is not None and age_years >= _MIN_AGE_FOR_RULES:
        if trial.min_age_years is not None and age_years + 1 <= trial.min_age_years:
            reasons.append(
                f"Requires age {(trial.min_age or '').lower()} or older; the patient is {age_years:g}."
            )
        if trial.max_age_years is not None and age_years > trial.max_age_years:
            reasons.append(
                f"Requires age {(trial.max_age or '').lower()} or younger; the patient is {age_years:g}."
            )
    wanted = (trial.sex or "ALL").upper()
    # A study that sets eligibility by gender identity is not decided by a sex field alone.
    if not trial.gender_based and sex in ("FEMALE", "MALE") and wanted in ("FEMALE", "MALE") and wanted != sex:
        reasons.append(f"Enrolls {wanted.lower()} participants only.")
    return reasons
