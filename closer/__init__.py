"""Closer: connect patients with clinical trials that are actively recruiting.

Pipeline (each step lives in its own module):

    patient story --profile.py--> structured profile
        --clinicaltrials.py--> candidate trials (live ClinicalTrials.gov data)
        --matcher.py--> YES / MAYBE / NO verdicts with reasons
        --pipeline.py--> one report for the UI (app.py) or the CLIs
"""

__version__ = "0.1.0"
