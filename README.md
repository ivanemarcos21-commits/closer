# Closer

A phone-friendly website that helps patients and families find **recruiting clinical trials** from a plain-English story.

You type something like *"My uncle has stage 3 lung cancer and the first chemotherapy didn't work."* Closer turns that into medical search terms, pulls live studies from [ClinicalTrials.gov](https://clinicaltrials.gov), and sorts the ones that look worth calling about.

**Information only, not medical advice.** Closer cannot say whether anyone qualifies and does not recommend any study. Talk with a doctor. In an emergency, call your local emergency number.

## How it works

1. **Groq** reads the story and extracts a diagnosis, age, sex, place, and extra search words.
2. **ClinicalTrials.gov** (the U.S. government's public registry, free API) is searched for recruiting studies. Place **ranks** results; it never hides studies that run elsewhere.
3. Simple rules skip studies whose published age or sex limits clearly don't fit.
4. Groq reads each remaining study's eligibility text and returns **YES / MAYBE / NO** with reasons. Missing information is never treated as a no.
5. Names, phones, emails, and addresses come **only** from the registry. The AI never writes contact details.

The UI labels a YES as **"Worth asking about"** — not "you qualify."

## Run it on your computer

Python **3.12** (see `.python-version`). Streamlit Community Cloud uses 3.12; 3.14 on your machine is unused.

```bash
cd "SAVE THE WORLD"
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # then paste your key
# Get a free key: https://aistudio.google.com/apikey

streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Without a key, the page still loads and searching is switched off until you add one.

## Try the pipeline without a website

No AI key needed for the registry:

```bash
python fetch_trials.py "non-small cell lung cancer"
python fetch_trials.py "type 1 diabetes" --location "Houston, TX" --limit 5
```

With a key in `.env`:

```bash
python match_trials.py --check
python match_trials.py --story "My uncle is 62 and has stage 3 lung cancer. The first chemotherapy did not work." --location "Boston, MA"
```

`--check` is the fastest way to confirm the Gemini key and model work.

## Tests

```bash
python -m pytest
```

354 tests cover parsing, ranking, Markdown escaping, the Streamlit page, and the CLIs. They do not call Google or ClinicalTrials.gov.

After you add a key, the live smoke tests (a few free-quota requests) are:

```bash
RUN_LIVE=1 python -m pytest -m live -s
```

The live registry test has been run successfully. Live Gemini tests skip until `GEMINI_API_KEY` is set, so the full AI success path has not been exercised against Google in this workspace.

## Put it on the internet (free)

[Streamlit Community Cloud](https://share.streamlit.io/) hosts this kind of app from a GitHub repo at a `*.streamlit.app` URL, no credit card.

1. Push this folder to a GitHub repository (keep `.env` out of git — it is already in `.gitignore`).
2. At share.streamlit.io, deploy with **Main file path:** `app.py`.
3. App → **Settings → Secrets**, paste:

   ```toml
   GROQ_API_KEY = "your-key"
   ```

4. Community Cloud should pick Python 3.12 from `.python-version`. If it doesn't, set 3.12 in the app's Python version setting.

### Before real patients use a public copy

Groq says it does not train on what people type. Unless Zero Data Retention is turned on in the Groq console, Groq may keep that text for up to 30 days. Closer is a hobby project, not a HIPAA covered entity.

For a public copy:

1. In Groq, open Settings → Data Controls and turn on Zero Data Retention.
2. Set `GROQ_ZERO_RETENTION=true` in `.env` or `GROQ_ZERO_RETENTION = "true"` in Streamlit secrets so the privacy notice matches.

The app already warns visitors either way. Don't type names, birth dates, addresses, or ID numbers.

## What it will not do

- Guarantee eligibility. Real screening needs labs, other conditions, and medicines a short story cannot show. The AI can be wrong in either direction.
- Read every matching study. It checks 20 by default (out of what may be hundreds) and links to the full ClinicalTrials.gov search for the rest.
- Cover expanded-access / compassionate-use programs, or studies that are not on ClinicalTrials.gov.
- Promise that a listing is still recruiting. Call the study team; registry contacts can be out of date.

The site also caps searches (about 40 per hour across all visitors, plus a short per-visitor cooldown) so a shared free key is not burned in an afternoon.

## Project layout

```
app.py                 Streamlit website
fetch_trials.py        Phase 1 CLI — live ClinicalTrials.gov only
match_trials.py        Phase 2 CLI — story → Gemini → ranked studies
closer/
  clinicaltrials.py    Registry client (ClinicalTrials.gov API v2)
  profile.py           Story → structured profile
  matcher.py           Eligibility YES / MAYBE / NO
  pipeline.py          End-to-end search
  llm.py               Gemini adapter (default: gemini-3.5-flash-lite)
  ui.py                Result cards, privacy copy, contact links
  text.py              Escaping for untrusted Markdown
.env.example           Key and paid-tier flag (copy to .env)
```

Gemini 1.5 Flash, named in many older tutorials, has been retired. Flash-Lite is the default because the free daily quota is much larger; you can override `GEMINI_MODEL` / `GEMINI_MATCH_MODEL` in `.env`.

Study data: ClinicalTrials.gov, U.S. National Library of Medicine. Closer is independent and is not affiliated with or endorsed by the NLM, the NIH, or Google.
