"""The notices have to keep the facts a patient needs, in both the short and full forms."""

from closer.chrome import appearance_css
from closer.legal import (
    MEDICAL_BANNER,
    medical_document,
    privacy_document,
    privacy_summary,
    terms_document,
)


def test_the_short_notices_match_whether_messages_are_kept():
    assert "not medical advice" in MEDICAL_BANNER and "emergency" in MEDICAL_BANNER
    logs = privacy_summary(False)
    kept_off = privacy_summary(True)
    assert "up to 30 days" in logs
    assert "does not use that text to train" in kept_off and "does not keep the messages" in kept_off
    assert "30 days" not in kept_off


def test_the_full_notices_cover_the_limits_of_the_tool():
    medical = medical_document()
    privacy = privacy_document(False)
    terms = terms_document(None, None)
    assert "not a medical device" in medical and "placebo" in medical and "Draft for review" in medical
    assert "under 13" in privacy and "not a HIPAA" in privacy and "18 or older" in privacy
    assert "European Economic Area" not in privacy_document(False)
    assert "European Economic Area" not in privacy_document(True)
    assert "not affiliated" in terms and "as-is" in terms
    assert "does not list a person to contact" in terms


def test_a_publisher_contact_is_shown_only_when_it_is_a_real_email():
    shown = terms_document("River Clinic", "desk@example.org")
    assert "River Clinic" in shown and "mailto:desk@example.org" in shown
    assert "mailto:" not in terms_document("River Clinic", "not an email")


def test_the_page_css_spaces_the_sheet_and_does_not_repaint_the_theme():
    css = appearance_css("Dark")
    assert "max-width: 40rem" in css
    assert "#10161C" not in css and "#F2F4F7" not in css
