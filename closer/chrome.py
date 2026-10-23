"""White page frame. Legal text sits at the bottom, not above the search."""

from __future__ import annotations

import streamlit as st

from .config import Settings
from .legal import (
    how_it_works,
    medical_document,
    privacy_document,
    privacy_summary,
    terms_document,
)

# A narrow white sheet. Corners stay nearly square so the form reads as a document.
_FRAME = """
section.main > div.block-container {
  max-width: 40rem;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
}
[data-testid="stVerticalBlock"] {
  gap: 1.15rem;
}
.closer-mark {
  display: inline-block;
  width: 0.5rem;
  height: 0.5rem;
  margin-right: 0.45rem;
  background: #1E4E78;
  vertical-align: 0.08rem;
}
.closer-kicker {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  margin: 0 0 0.15rem 0;
}
[data-testid="stHeading"] h1 {
  font-size: 2.45rem;
  letter-spacing: -0.03em;
  line-height: 1.05;
  margin: 0.2rem 0 0.4rem 0;
}
[data-testid="stTextArea"] textarea,
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
  border-radius: 2px !important;
}
[data-testid="stFormSubmitButton"] button {
  border-radius: 2px !important;
  min-height: 2.8rem;
  padding-left: 1.35rem;
  padding-right: 1.35rem;
}
.closer-foot {
  margin-top: 2.75rem;
  padding-top: 1.35rem;
  border-top: 1px solid #E4E7EB;
}
[class*="st-key-theme-lock"] {
  display: none !important;
}
footer[data-testid="stFooter"] {
  display: none;
}
"""

# The page stays white even if this browser previously saved a dark theme.
_LOCK = st.components.v2.component(
    "closer_theme_lock",
    js="""
    export default function () {
      const key = `stActiveTheme-${window.location.pathname}-v2`;
      let stored = null;
      try {
        stored = JSON.parse(window.localStorage.getItem(key) || "null");
      } catch (err) {
        stored = null;
      }
      if (stored === "Light") return;
      try {
        window.localStorage.setItem(key, JSON.stringify("Light"));
      } catch (err) {
        return;
      }
      window.location.reload();
    }
    """,
)


def appearance_css(mode: str) -> str:
    """Layout CSS. The page does not paint its own dark palette."""
    del mode
    return _FRAME


def install_style() -> None:
    st.markdown(f"<style>{_FRAME}</style>", unsafe_allow_html=True)
    _LOCK(key="theme-lock")


def render_masthead() -> None:
    st.markdown(
        '<p class="closer-kicker"><span class="closer-mark"></span>Closer</p>',
        unsafe_allow_html=True,
    )


def render_footer(settings: Settings) -> None:
    st.markdown('<div class="closer-foot"></div>', unsafe_allow_html=True)
    st.markdown("## Notices")
    st.markdown("The search box is for a description of the illness. These notices explain the limits.")

    st.markdown("### Medical notice")
    st.markdown(medical_document())

    st.markdown("### Privacy")
    st.markdown(privacy_summary(settings.zero_retention))
    st.markdown(privacy_document(settings.zero_retention))

    st.markdown("### Terms of use")
    st.markdown(terms_document(settings.operator_name, settings.operator_email))

    st.markdown("### How it works")
    st.markdown(how_it_works())

    st.caption(
        "Study data: ClinicalTrials.gov, U.S. National Library of Medicine. "
        "Closer is an independent project and is not affiliated with or endorsed by the NLM, "
        "the NIH, or Groq."
    )
