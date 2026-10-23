"""Plain-language notices shown on the site.

These are explanations of how Closer works, written for a patient or a family member.
They are not a lawyer's opinion, and they do not make Closer a medical provider, a
HIPAA-covered service, or an FDA-cleared device. The person who publishes a public
copy still needs billing turned on for the AI key and a review for their own location.
"""

from __future__ import annotations

from .text import escape_link_text, escape_md, mailto_url

MEDICAL_BANNER = (
    "**Information only, not medical advice.** Closer can't say whether you qualify and doesn't "
    "recommend any study. Please talk with your doctor. In an emergency, call your local "
    "emergency number."
)

_PRIVACY_LEAD = "**Don't type names, birth dates, addresses or ID numbers.** "
_PRIVACY_TAIL = " Closer doesn't save it. The diagnosis and place also go to ClinicalTrials.gov."
_PRIVACY_LOGS = (
    "Your text goes to Groq, the company that runs the AI. Groq says it does not use that text to "
    "train its models. Unless Zero Data Retention is turned on, Groq may keep the text for up to "
    "30 days to fix outages or check for misuse."
)
_PRIVACY_ZDR = (
    "Your text goes to Groq, the company that runs the AI. Groq says it does not use that text to "
    "train its models, and Zero Data Retention is turned on, so Groq does not keep the messages."
)

CONSENT_LABEL = "I am 18 or older, and I agree to the notices at the bottom of this page."

BEFORE_YOU_CALL = (
    "**Before you call:** joining a study is your choice, and you can leave at any time. Studies "
    "can have risks and side effects, and some give a standard treatment or a placebo instead of "
    "the new one. Closer doesn't recommend or endorse any study. Call to confirm a study is still "
    "enrolling, and to ask about costs and travel."
)

AGREEMENT_REQUIRED = (
    "Please check the box confirming you are 18 or older and that you understand where your "
    "description is sent. Then press Find trials again."
)


def privacy_summary(zero_retention: bool) -> str:
    """The short privacy text. It must match whether Zero Data Retention is actually on."""
    return _PRIVACY_LEAD + (_PRIVACY_ZDR if zero_retention else _PRIVACY_LOGS) + _PRIVACY_TAIL


def medical_document() -> str:
    return """\
##### What Closer is

Closer is an information sheet. It reads a description you type, searches the public
[ClinicalTrials.gov](https://clinicaltrials.gov) registry for studies that are recruiting, and
sorts them. It then shows phone numbers and email addresses copied from that registry.

##### What Closer is not

- It is not medical advice, a diagnosis, or a treatment plan.
- It does not decide whether you qualify for a study. Only the study team can do that, after they
  review records Closer never sees.
- It does not recommend or endorse any study, drug, hospital, or sponsor.
- It does not create a doctor-patient relationship with you.
- It is not cleared or approved by the U.S. Food and Drug Administration, and it is not a medical device.

##### Emergencies

If this is an emergency, call your local emergency number. Do not wait for a search result.

##### About the studies themselves

Joining a study is voluntary. You can say no, and you can leave. Studies can have risks and side
effects. Some give a standard treatment or a placebo instead of the new one. There may be costs
and travel. Closer does not know your insurance, your other conditions, or your test results.

Talk with your own doctor before you contact a study and before you join one.

##### How wrong a result can be

The AI can misread your description or a study's rules, in either direction. A study marked
"Worth asking about" has not been ruled out by the text you typed. It is not an offer, and it is
not a statement that you are eligible. Recruiting status and contact details come from the
registry and can be out of date. Call to confirm.

Closer reads only a limited number of studies (20 by default). It does not list every study, and
it does not cover expanded-access (compassionate use) programs or studies that are not on
ClinicalTrials.gov.
"""


def privacy_document(zero_retention: bool) -> str:
    groq = (
        "Zero Data Retention is turned on for this copy. Groq says it does not keep the messages "
        "and does not use them to train its models."
        if zero_retention
        else
        "This copy has not turned on Zero Data Retention. Groq says it does not use the text to "
        "train its models, but it may keep the text for up to 30 days to fix outages or check for "
        "misuse. Do not put names, birth dates, addresses, phone numbers, email addresses, "
        "medical-record numbers, or ID numbers in the box."
    )
    return f"""\
##### Who may use this page

This page is for adults 18 or older. An adult may use it to look for a study for a child or for
another person they care for. Do not ask a child to type their own story here.

##### What you are asked for

The description you type, plus any age, sex, and place you add. Sex is used only to skip studies
with a sex limit (studies usually mean sex recorded at birth). The place is used to show studies
with a nearby open site first. It does not hide studies that run somewhere else.

Do not type names, birth dates, street addresses, phone numbers, email addresses, medical-record
numbers, insurance numbers, or government ID numbers.

##### Where that text goes

1. **Groq.** Your description is sent to Groq's API so it can be turned into medical search words
   and compared with each study's published rules. {groq}
2. **ClinicalTrials.gov.** The diagnosis and the place are sent to the U.S. National Library of
   Medicine's public registry API. The registry's own privacy policy applies to that request.
3. **This website's memory for the current visit.** The description stays in the browser session
   so the page can show your results. Closer does not write it to a database, and it does not put
   it in the page address.

##### What Closer keeps

Closer does not have user accounts. It does not sell information. It does not show ads. It does
not use an analytics product. Press **Start over**, or close the tab, to drop the description
from this visit.

A phone number or email on a result is copied from ClinicalTrials.gov. The AI is not allowed to
invent contact details.

##### What this is not

Closer is not a doctor's office, health plan, or healthcare clearinghouse, and it is not a HIPAA
business associate. Do not use it as a medical record or as a private message to a clinic.

##### Cookies

The page uses a session cookie so it can remember this visit, including that you checked the
agreement. That cookie is not used to advertise to you.
"""


def terms_document(operator_name: str | None, operator_email: str | None) -> str:
    contact = _operator_sentence(operator_name, operator_email)
    return f"""\
##### Using Closer

You may use Closer to read public clinical-trial listings and to decide whom you might call.
You may not use it for an emergency, to collect other people's identity documents, or to present
its results as proof that someone qualifies for a study.

You must be 18 or older. You are responsible for what you type. If you describe someone else,
type only what you need for a search, and leave out their name and contact details.

##### No warranty

Closer is provided as-is. The registry can be wrong or out of date. The AI can be wrong. Listings
can close without this page knowing. Closer does not promise that a study is still enrolling,
that a phone number works, or that any study is safe or appropriate for you.

##### Independence

Study data comes from ClinicalTrials.gov, U.S. National Library of Medicine. Closer is an
independent project. It is not affiliated with, endorsed by, or operated by the NLM, the NIH,
or Groq.

##### The site can be turned off

The person who runs this copy can change or remove it at any time.

##### These words

This page is a plain-language explanation, not legal advice to you. {contact}
"""


def how_it_works() -> str:
    return """\
##### What happens when you search

1. You describe the situation in your own words and, if you want, add an age, sex, and place.
2. Groq's AI turns that into medical terms and search words.
3. Closer searches ClinicalTrials.gov for studies that are recruiting.
4. Simple rules skip studies whose published age or sex limits do not fit. The AI reads the other
   rules of the studies that remain and sorts them.
5. Names, phone numbers, emails, and addresses on this page are copied from ClinicalTrials.gov.
   The AI does not write them.

##### What a result means

**Worth asking about** means nothing in your description ruled that study out. It does not mean
you qualify.

**Possible match** means something important was missing, or the fit is unclear. The card lists
what to ask the study team.

**Probably not a fit** means the published rules and your description disagree. The AI can still
be wrong. If you are unsure, ask your doctor.

##### What this page does not cover

It reads only a few studies (20 by default) out of what may be hundreds, and it links to the full
ClinicalTrials.gov search for the rest. It does not cover expanded-access programs or studies that
are not listed on ClinicalTrials.gov.
"""


def _operator_sentence(name: str | None, email: str | None) -> str:
    safe_name = escape_md(name) if name else ""
    link = mailto_url(email) if email else None
    if safe_name and link:
        shown = escape_link_text(email or "")
        return (
            f"This copy is run by {safe_name}. For a question about the website itself, write to "
            f"[{shown}]({link}). They cannot tell you whether you qualify for a study."
        )
    if link:
        shown = escape_link_text(email or "")
        return (
            "For a question about the website itself, write to "
            f"[{shown}]({link}). That address cannot tell you whether you qualify for a study."
        )
    return (
        "This copy does not list a person to contact about the website. Questions about a study "
        "go to that study's team. Questions about your health go to your doctor."
    )
