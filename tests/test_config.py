"""Tests for closer.config: where settings come from, and that the key never leaks."""

from __future__ import annotations

import logging

import pytest

from closer import config
from closer.config import DEFAULT_MODEL, Settings

NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_MATCH_MODEL", "GEMINI_PAID_TIER")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_nothing_is_set():
    settings = Settings.load(use_dotenv=False)
    assert settings.gemini_api_key is None and not settings.has_ai
    assert settings.model == DEFAULT_MODEL and settings.effective_match_model == DEFAULT_MODEL
    assert settings.paid_tier is False


def test_environment_variables_are_read(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  key-from-env  ")
    monkeypatch.setenv("GEMINI_MODEL", "small")
    monkeypatch.setenv("GEMINI_MATCH_MODEL", "big")
    settings = Settings.load(use_dotenv=False)
    assert settings.gemini_api_key == "key-from-env" and settings.has_ai
    assert (settings.model, settings.effective_match_model) == ("small", "big")


def test_google_api_key_is_accepted_as_a_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert Settings.load(use_dotenv=False).gemini_api_key == "g-key"


def test_a_blank_key_counts_as_no_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")  # e.g. copied from .env.example unchanged
    assert not Settings.load(use_dotenv=False).has_ai


def test_streamlit_secrets_fill_gaps_but_the_environment_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    settings = Settings.load({"GEMINI_API_KEY": "from-secrets", "GEMINI_MODEL": "from-secrets"}, use_dotenv=False)
    assert settings.gemini_api_key == "from-secrets" and settings.model == "from-env"


def test_non_text_secret_values_do_not_crash():
    assert Settings.load({"GEMINI_MODEL": 3}, use_dotenv=False).model == "3"  # e.g. an unquoted TOML number


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("1", True), ("YES", True), ("false", False), ("", False)])
def test_paid_tier_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("GEMINI_PAID_TIER", raw)
    assert Settings.load(use_dotenv=False).paid_tier is expected


def test_the_api_key_never_appears_in_repr_or_str():
    settings = Settings(gemini_api_key="super-secret-key")
    assert "super-secret-key" not in repr(settings) and "super-secret-key" not in f"{settings}"


# --------------------------------------------------------------------------- .env file


def test_the_dotenv_file_is_read(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=key-from-dotenv\nGEMINI_PAID_TIER=true\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    settings = Settings.load()
    assert settings.gemini_api_key == "key-from-dotenv" and settings.paid_tier


def test_real_environment_variables_beat_the_dotenv_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    monkeypatch.setenv("GEMINI_MODEL", "from-shell")
    assert Settings.load().model == "from-shell"


def test_the_project_dotenv_path_is_next_to_the_package():
    assert config._ENV_FILE.name == ".env" and (config._ENV_FILE.parent / "closer").is_dir()


def test_a_missing_dotenv_is_fine(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "nope.env")
    assert not Settings.load().has_ai


def test_a_dotenv_that_cannot_be_read_because_the_library_is_missing_is_reported(monkeypatch, tmp_path, caplog):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)  # makes "import dotenv" fail
    with caplog.at_level(logging.WARNING, logger="closer.config"):
        settings = Settings.load()
    assert not settings.has_ai
    assert any("python-dotenv is not installed" in r.message for r in caplog.records)
