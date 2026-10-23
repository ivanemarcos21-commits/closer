"""Helpers for untrusted text: trial records, patient stories and model output.

The app shows text from three sources it does not control (the ClinicalTrials.gov
registry, the patient's own words, and the language model). These helpers keep that text
from being interpreted as Markdown/LaTeX in the UI, or as structure inside prompts, and
build outbound links only from validated pieces.
"""

from __future__ import annotations

import re

NCT_ID_RE = re.compile(r"^NCT\d{8}$")

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._+'\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
_PHONE_CHARS_RE = re.compile(r"^[0-9\s().+\-]+$")
_PHONE_EXT_RE = re.compile(r"\s*(?:ext\.?|x)\s*[0-9]{1,6}\s*$", re.IGNORECASE)

# Everything Streamlit's Markdown (CommonMark + GFM + LaTeX + directives) treats as syntax.
_MD_SPECIALS_RE = re.compile(r"([\\`*_{}\[\]()<>#+!|~$:])")
_LEADING_NUMBER_RE = re.compile(r"(?m)^([ \t]*[0-9]+)\.")
_LEADING_RULE_RE = re.compile(r"(?m)^([ \t]*)([-=])")
_LINE_BREAKS_RE = re.compile(r"\r\n?|[\u2028\u2029\x0b\x0c\x85]")
_BIDI_CONTROLS_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

# What GitHub-flavoured Markdown auto-links on its own: URLs with a scheme, "www." hosts and
# e-mail addresses. A backslash does NOT stop this in the browser (the escape is decoded first,
# then the text is linked), so such tokens are shown as inert code instead.
_LINKABLE_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)[^\s<>]+|[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", re.IGNORECASE
)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\""

# Tags we use to fence untrusted text inside prompts.
_PROMPT_TAG_RE = re.compile(
    r"</?\s*(?:trial|trials|patient_story|patient_profile)\b[^>]*>", re.IGNORECASE
)


def escape_link_text(text: object) -> str:
    """Escape Markdown syntax in text that will sit inside ``[...](url)`` link text.

    Links cannot nest, so nothing in here can be auto-linked; only syntax needs neutralising.
    """
    if text is None:
        return ""
    escaped = _MD_SPECIALS_RE.sub(r"\\\1", str(text))
    escaped = _LEADING_NUMBER_RE.sub(r"\1\\.", escaped)  # "1. x" must not become a list
    return _LEADING_RULE_RE.sub(r"\1\\\2", escaped)  # "- x" / "====" at the start of a line


def escape_md(text: object) -> str:
    """Make externally supplied text render as plain text in Streamlit's Markdown.

    Handles images and links (``![x](https://evil/?q=...)``), emphasis, headings and lists,
    ``$...$`` math, ``:name[...]`` directives, odd line separators and right-to-left override
    characters. URLs and e-mail addresses are shown as inert code spans, because Markdown
    would otherwise make them clickable. Links we *want* (study page, phone, e-mail) are
    built separately from validated parts.
    """
    if text is None:
        return ""
    cleaned = _BIDI_CONTROLS_RE.sub("", _LINE_BREAKS_RE.sub("\n", str(text)))
    parts: list[str] = []
    position = 0
    for match in _LINKABLE_RE.finditer(cleaned):
        token = match.group()
        core = token.rstrip(_TRAILING_PUNCTUATION) or token
        parts.append(escape_link_text(cleaned[position : match.start()]))
        parts.append("`" + core.replace("`", "'") + "`")
        parts.append(escape_link_text(token[len(core) :]))
        position = match.end()
    parts.append(escape_link_text(cleaned[position:]))
    return "".join(parts)


def contains_link_like(text: str) -> bool:
    """True if ``text`` holds a URL, "www." host or e-mail address."""
    return bool(_LINKABLE_RE.search(text))


def neutralize_delimiters(text: str) -> str:
    """Remove look-alikes of the tags that fence untrusted text inside our prompts."""
    return _PROMPT_TAG_RE.sub("[tag removed]", text)


def collapse_whitespace(text: str) -> str:
    """Turn any run of whitespace (including newlines) into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def shorten(text: str, limit: int) -> str:
    """Trim ``text`` to about ``limit`` characters, cutting at a word boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip(" ,;:.-") + "..."


# --------------------------------------------------------------------------- personal details
_PERSONAL_DETAIL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an e-mail address", re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")),
    (
        "a phone number",
        re.compile(r"(?<![0-9])(?:\+?[0-9]{1,2}[\s.\-]?)?(?:\([0-9]{3}\)|[0-9]{3})[\s.\-]?[0-9]{3}[\s.\-]?[0-9]{4}(?![0-9])"),
    ),
    ("a Social Security number", re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b")),
    (
        "a date of birth",
        re.compile(r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12][0-9]|3[01])[/\-](?:19|20)?[0-9]{2}\b"),
    ),
    (
        "an ID number or birth date label",
        re.compile(r"\b(?:DOB|date of birth|MRN|medical record (?:number|no)|SSN|social security)\b", re.IGNORECASE),
    ),
)


def find_personal_details(text: str) -> list[str]:
    """Plain-English names for kinds of personal details that appear to be in ``text``.

    A nudge, not a filter: it is used to ask people to remove such details before their text
    is sent to an AI service. False alarms are acceptable; silence about a real one is not.
    """
    return [label for label, pattern in _PERSONAL_DETAIL_PATTERNS if pattern.search(text)]


# --------------------------------------------------------------------------- links we build
def clinicaltrials_url(nct_id: str | None) -> str | None:
    """Public study page for an NCT number, or ``None`` if the id is malformed."""
    nct = (nct_id or "").strip().upper()
    return f"https://clinicaltrials.gov/study/{nct}" if NCT_ID_RE.match(nct) else None


def mailto_url(email: str | None) -> str | None:
    """A ``mailto:`` link, only for strings that really look like one e-mail address."""
    candidate = (email or "").strip()
    return f"mailto:{candidate}" if _EMAIL_RE.match(candidate) else None


def tel_url(phone: str | None) -> str | None:
    """A ``tel:`` link built from digits only; ``None`` for anything unusual.

    A trailing extension ("ext. 42") is dropped: it cannot be dialled through a link, and the
    visible text still shows it.
    """
    candidate = _PHONE_EXT_RE.sub("", (phone or "").strip())
    if not candidate or not _PHONE_CHARS_RE.match(candidate):
        return None
    digits = re.sub(r"[^0-9]", "", candidate)
    if not 7 <= len(digits) <= 15:
        return None
    return "tel:" + ("+" if candidate.startswith("+") else "") + digits
