"""Tests for closer.clinicaltrials: parsing, the HTTP client, place matching and age/sex rules."""

from __future__ import annotations

import ssl

import httpx
import pytest

from closer.clinicaltrials import (
    ClinicalTrialsClient,
    ClinicalTrialsError,
    ClinicalTrialsRejectedError,
    location_tokens,
    parse_age_years,
    parse_study,
    registry_location,
    screen_demographics,
    select_sites,
    site_is_open,
)
from closer.models import Contact

NCT_FIRST = "NCT06855771"

# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18 Years", 18.0),
        ("6 Months", 0.5),
        ("1 Year", 1.0),
        ("52 Weeks", pytest.approx(52 * 7 / 365.25)),
        ("N/A", None),
        ("", None),
        (None, None),
        ("eighteen", None),
    ],
)
def test_parse_age_years(text, expected):
    assert parse_age_years(text) == expected


def test_parse_real_record(fixture_trials):
    trial = fixture_trials[0]
    assert trial.nct_id == "NCT06855771"
    assert trial.status == "RECRUITING"
    assert trial.phases == ["PHASE2"]
    assert trial.sponsor == "Bristol-Myers Squibb"
    assert trial.conditions == ["Carcinoma, Non-Small-Cell Lung"]
    assert trial.min_age == "18 Years" and trial.min_age_years == 18
    assert trial.max_age is None and trial.max_age_years is None  # no upper limit published
    assert trial.sex == "ALL"
    assert trial.healthy_volunteers is False
    assert "Inclusion Criteria" in trial.eligibility_text
    assert trial.url == "https://clinicaltrials.gov/study/NCT06855771"
    assert trial.last_updated
    assert trial.study_type == "INTERVENTIONAL"


def test_gender_based_eligibility_and_status_confirmation_are_read():
    trial = parse_study(
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678"},
                "statusModule": {"overallStatus": "RECRUITING", "statusVerifiedDate": "2026-02"},
                "eligibilityModule": {
                    "sex": "FEMALE",
                    "genderBased": True,
                    "genderDescription": "Anyone who identifies as a woman",
                },
            }
        }
    )
    assert trial is not None
    assert trial.gender_based is True and trial.gender_description == "Anyone who identifies as a woman"
    assert trial.status_verified == "2026-02"


def test_central_contact_includes_extension(fixture_trials):
    contact = fixture_trials[0].central_contacts[0]
    assert contact.phone == "555-0100 ext. 7"
    assert contact.email and contact.email.endswith("@example.org")


def test_site_status_and_contacts_are_kept(fixture_trials):
    sites = fixture_trials[0].sites
    assert [s.status for s in sites].count("COMPLETED") == 2
    assert any(s.status is None for s in sites)  # older records omit the status
    boston = next(s for s in sites if s.facility == "Dana-Farber Cancer Institute")
    assert boston.city == "Boston" and boston.state == "Massachusetts"
    assert boston.contacts and "ext. 42" in (boston.contacts[0].phone or "")


def test_minimal_record_does_not_crash():
    trial = parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}})
    assert trial is not None
    assert trial.title == "NCT12345678"  # falls back to the id when there is no title
    assert trial.status == "UNKNOWN" and trial.sex == "ALL"
    assert trial.sites == [] and trial.central_contacts == [] and trial.eligibility_text == ""


@pytest.mark.parametrize("raw", [{}, {"protocolSection": {}}, {"protocolSection": {"identificationModule": {"nctId": "BAD"}}}])
def test_records_without_a_valid_nct_id_are_dropped(raw):
    assert parse_study(raw) is None


def test_contacts_with_nothing_in_them_are_dropped():
    raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT12345678"},
            "contactsLocationsModule": {"centralContacts": [{"role": "CONTACT"}, {"name": "A. Person"}]},
        }
    }
    trial = parse_study(raw)
    assert trial is not None
    assert [c.name for c in trial.central_contacts] == ["A. Person"]


# --------------------------------------------------------------------------- HTTP client


def make_client(handler, **kwargs):
    sleeps: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ClinicalTrialsClient(http_client=http, sleep=sleeps.append, **kwargs), sleeps


def test_search_sends_the_documented_parameters(ctgov_raw):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ctgov_raw)

    client, _ = make_client(handler)
    page = client.search(
        condition="lung cancer",
        terms="(stage III OR locally advanced)",
        location="Boston, MA",
        statuses=("RECRUITING", "NOT_YET_RECRUITING"),
        page_size=500,
    )
    params = seen[0].url.params
    assert seen[0].url.path == "/api/v2/studies"
    assert params["query.cond"] == "lung cancer"
    assert params["query.term"] == "(stage III OR locally advanced)"
    assert params["query.locn"] == "Boston, MA"
    assert params["filter.overallStatus"] == "RECRUITING|NOT_YET_RECRUITING"
    assert params["pageSize"] == "100"  # clamped
    assert params["countTotal"] == "true"
    assert "NCTId" in params["fields"] and "EligibilityCriteria" in params["fields"]
    assert [t.nct_id for t in page.trials] == ["NCT06855771", "NCT07063745"]
    assert page.total_count == 114


def test_the_place_is_made_safe_for_the_registrys_query_language(ctgov_raw):
    seen: list[httpx.Request] = []
    client, _ = make_client(lambda r: (seen.append(r), httpx.Response(200, json=ctgov_raw))[1])
    client.search(condition="asthma", location="Portland, OR")
    client.search(condition="asthma", location="Boston, MA, USA")
    assert seen[0].url.params["query.locn"] == "Portland, Oregon"
    assert seen[1].url.params["query.locn"] == "Boston, MA, United States"


def test_one_malformed_record_does_not_sink_the_search(ctgov_raw):
    broken = {"protocolSection": {"identificationModule": {"nctId": "NCT99999999"}, "contactsLocationsModule": 5}}
    payload = {**ctgov_raw, "studies": [broken, ctgov_raw["studies"][0], "not even a dict"]}
    client, _ = make_client(lambda r: httpx.Response(200, json=payload))
    page = client.search(condition="x")
    assert NCT_FIRST in [t.nct_id for t in page.trials]


def test_redirects_are_never_followed():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    client = ClinicalTrialsClient(http_client=http, sleep=lambda s: None)
    with pytest.raises(ClinicalTrialsRejectedError, match="HTTP 302"):
        client.search(condition="x")
    assert len(calls) == 1


def test_a_default_client_does_not_follow_redirects():
    with ClinicalTrialsClient() as client:
        assert client._http.follow_redirects is False


def test_search_omits_empty_optional_parameters(ctgov_raw):
    seen: list[httpx.Request] = []
    client, _ = make_client(lambda r: (seen.append(r), httpx.Response(200, json=ctgov_raw))[1])
    client.search(condition="asthma", terms="  ", location="")
    params = seen[0].url.params
    assert "query.term" not in params and "query.locn" not in params


def test_search_needs_something_to_search_for():
    client, _ = make_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.search(condition=" ", terms=None)


def test_transient_errors_are_retried_with_backoff(ctgov_raw):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=ctgov_raw)

    client, sleeps = make_client(handler, backoff=1.0)
    assert client.search(condition="x").trials
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_retry_after_header_is_respected(ctgov_raw):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=ctgov_raw)

    client, sleeps = make_client(handler)
    client.search(condition="x")
    assert sleeps == [3.0]


def test_gives_up_with_a_friendly_error():
    client, sleeps = make_client(lambda r: httpx.Response(502), max_retries=2)
    with pytest.raises(ClinicalTrialsError, match=r"not responding.*HTTP 502"):
        client.search(condition="x")
    assert len(sleeps) == 2


def test_client_errors_are_not_retried_and_carry_the_servers_explanation():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="Invalid value in parameter `overallStatus`: `BOGUS`")

    client, sleeps = make_client(handler)
    with pytest.raises(ClinicalTrialsError, match="BOGUS"):
        client.search(condition="x")
    assert len(calls) == 1 and sleeps == []


def test_connection_failures_are_wrapped():
    def handler(request):
        raise httpx.ConnectTimeout("slow", request=request)

    client, _ = make_client(handler, max_retries=1)
    with pytest.raises(ClinicalTrialsError, match="ConnectTimeout"):
        client.search(condition="x")


def test_certificate_problems_fail_fast_without_retrying():
    calls = []

    def handler(request):
        calls.append(1)
        error = httpx.ConnectError("certificate verify failed", request=request)
        error.__cause__ = ssl.SSLCertVerificationError("self-signed certificate")
        raise error

    client, sleeps = make_client(handler)
    with pytest.raises(ClinicalTrialsError, match="security certificate"):
        client.search(condition="x")
    assert len(calls) == 1 and sleeps == []


def test_unreadable_response_is_reported():
    client, _ = make_client(lambda r: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ClinicalTrialsError, match="could not read"):
        client.search(condition="x")


# --------------------------------------------------------------------------- places and sites


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        ("Boston, MA", ["boston", "massachusetts"]),
        ("Boston MA", ["boston", "massachusetts"]),
        ("Boston, Massachusetts", ["boston", "massachusetts"]),
        ("Boston, MA, USA", ["boston", "massachusetts", "united states"]),
        ("Houston Texas", ["houston", "texas"]),
        ("New York", ["new york"]),
        ("New York, NY", ["new york"]),  # city and state share a name: one token, not two
        ("NYC", ["new york"]),
        ("Albany New York", ["albany", "new york"]),
        ("West Virginia", ["west virginia"]),
        ("Washington DC", ["washington", "district of columbia"]),
        ("USA", ["united states"]),
        ("United States", ["united states"]),
        ("Madrid, Spain", ["madrid", "spain"]),
        ("02115", ["02115"]),
        ("Kansas City", ["kansas city"]),
    ],
)
def test_location_tokens(place, expected):
    assert location_tokens(place) == expected


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        ("Portland, OR", "Portland, Oregon"),  # OR is the registry's boolean operator
        ("Portland OR", "Portland, Oregon"),
        ("Eugene, or", "Eugene, Oregon"),
        ("Boston, MA", "Boston, MA"),  # the registry understands state abbreviations itself
        ("Boston, MA, USA", "Boston, MA, United States"),  # but matches country names literally
        ("London, UK", "London, United Kingdom"),
        ("USA", "United States"),
        ("Houston, TX (north)", "Houston, TX north"),  # brackets are query syntax
        ('Say "hello"', "Say hello"),
        ("Paris AND Lyon", "Paris, Lyon"),  # operators inside a place become separators
        ("Boston OR Cambridge", "Boston, Cambridge"),
        ("NOT", None),
        ("  ", None),
        (None, None),
    ],
)
def test_registry_location_is_safe_for_the_boolean_query_language(place, expected):
    assert registry_location(place) == expected


def test_site_is_open_uses_the_sites_own_status(make_site):
    assert site_is_open(make_site(status="RECRUITING"), "RECRUITING")
    assert site_is_open(make_site(status=None), "RECRUITING")  # unknown counts as open
    assert not site_is_open(make_site(status="COMPLETED"), "RECRUITING")
    assert not site_is_open(make_site(status="WITHDRAWN"), "RECRUITING")
    assert not site_is_open(make_site(status="NOT_YET_RECRUITING"), "RECRUITING")
    assert site_is_open(make_site(status="NOT_YET_RECRUITING"), "NOT_YET_RECRUITING")


def test_select_sites_puts_the_requested_place_first(fixture_trials):
    chosen = select_sites(fixture_trials[0], "Boston, MA", limit=4)
    assert chosen.open_count == 5  # 7 sites minus the 2 completed ones
    assert chosen.nearby_count == 2
    assert all(s.city == "Boston" for s in chosen.sites[:2])


def test_select_sites_never_shows_closed_sites(fixture_trials):
    chosen = select_sites(fixture_trials[0], "Anchorage, Alaska", limit=10)
    assert all(s.status != "COMPLETED" for s in chosen.sites)
    assert chosen.nearby_count == 0  # the Anchorage site is closed, so nothing nearby


def test_nearby_means_the_whole_place_matched(make_trial, make_site):
    trial = make_trial(
        sites=[
            make_site(city="Cambridge", state="Massachusetts", facility="Cambridge site"),
            make_site(city="Boston", state="Lincolnshire", country="United Kingdom", facility="UK site"),
            make_site(city="Boston", state="Massachusetts", facility="Boston site"),
        ]
    )
    chosen = select_sites(trial, "Boston, MA")
    assert chosen.nearby_count == 1  # only the real Boston, Massachusetts
    assert chosen.sites[0].facility == "Boston site"
    assert {s.facility for s in chosen.sites} == {"Boston site", "Cambridge site", "UK site"}  # still listed


def test_washington_dc_does_not_count_seattle_as_nearby(make_trial, make_site):
    trial = make_trial(
        sites=[
            make_site(city="Seattle", state="Washington", facility="Seattle site"),
            make_site(city="Washington", state="District of Columbia", facility="DC site"),
        ]
    )
    chosen = select_sites(trial, "Washington, DC")
    assert chosen.nearby_count == 1 and chosen.sites[0].facility == "DC site"


def test_metro_nicknames_find_the_city(make_trial, make_site):
    trial = make_trial(sites=[make_site(city="New York", state="New York"), make_site(city="Austin", state="Texas")])
    assert select_sites(trial, "NYC").nearby_count == 1


def test_a_country_or_state_alone_counts_every_site_in_it(make_trial, make_site):
    trial = make_trial(
        sites=[make_site(city="Austin", state="Texas"), make_site(city="Dallas", state="Texas"), make_site(city="Reno", state="Nevada")]
    )
    assert select_sites(trial, "Texas").nearby_count == 2


def test_select_sites_falls_back_to_other_open_sites(fixture_trials):
    chosen = select_sites(fixture_trials[1], "Tampa, Florida", limit=2)
    assert chosen.nearby_count == 0  # Tampa is not yet open for this study
    assert len(chosen.sites) == 2 and chosen.open_count == 3


def test_select_sites_without_a_place_prefers_sites_with_a_contact(make_trial, make_site):
    quiet = make_site(city="Austin", facility="No contact")
    reachable = make_site(
        city="Dallas", facility="Has contact", contacts=[Contact(name="A", phone="555-0100")]
    )
    chosen = select_sites(make_trial(sites=[quiet, reachable]), None, limit=1)
    assert chosen.sites[0].facility == "Has contact"
    assert chosen.nearby_count == 0


def test_zip_codes_match_by_prefix(make_trial, make_site):
    trial = make_trial(
        sites=[
            make_site(city="Boston", zip="02215-1234"),
            make_site(city="Austin", state="Texas", zip="78701"),
        ]
    )
    chosen = select_sites(trial, "02215")
    assert chosen.nearby_count == 1 and chosen.sites[0].city == "Boston"


# --------------------------------------------------------------------------- age / sex rules


def test_age_below_minimum_is_excluded(make_trial):
    trial = make_trial(min_age="18 Years", min_age_years=18)
    reasons = screen_demographics(trial, 16, "UNKNOWN")
    assert len(reasons) == 1 and "18 years" in reasons[0] and "16" in reasons[0]
    assert screen_demographics(trial, 17, None)  # even at 17 years 11 months they are under 18
    assert screen_demographics(trial, 18, None) == []


def test_age_above_maximum_is_excluded_but_the_boundary_is_allowed(make_trial):
    trial = make_trial(max_age="65 Years", max_age_years=65)
    assert screen_demographics(trial, 66, None)
    assert screen_demographics(trial, 65, None) == []


def test_a_minimum_given_in_months_does_not_exclude_a_child_who_may_be_old_enough(make_trial):
    # Minimum 30 months (2.5 years). A "2 year old" may be 2 years 11 months old.
    trial = make_trial(min_age="30 Months", min_age_years=2.5)
    assert screen_demographics(trial, 2, None) == []
    # A maximum of 42 months (3.5 years): a "3 year old" is still within it.
    trial = make_trial(max_age="42 Months", max_age_years=3.5)
    assert screen_demographics(trial, 3, None) == []


def test_gender_based_studies_are_not_excluded_by_the_sex_field_alone(make_trial):
    trial = make_trial(sex="FEMALE", gender_based=True, gender_description="Anyone who identifies as a woman")
    assert screen_demographics(trial, None, "MALE") == []
    assert screen_demographics(make_trial(sex="FEMALE", gender_based=False), None, "MALE")


def test_sex_mismatch_is_excluded_and_all_is_open(make_trial):
    assert screen_demographics(make_trial(sex="FEMALE"), None, "MALE")
    assert screen_demographics(make_trial(sex="ALL"), None, "MALE") == []
    assert screen_demographics(make_trial(sex="FEMALE"), None, "UNKNOWN") == []  # unknown is never a reason


def test_unknown_age_never_excludes(make_trial):
    trial = make_trial(min_age="18 Years", min_age_years=18, max_age="65 Years", max_age_years=65)
    assert screen_demographics(trial, None, None) == []


def test_infant_ages_are_left_to_the_ai(make_trial):
    # "1" may really be 14 months; whole-year ages below 2 are too coarse for a hard rule.
    trial = make_trial(min_age="12 Months", min_age_years=1.0, max_age="18 Months", max_age_years=1.5)
    assert screen_demographics(trial, 1, None) == []
