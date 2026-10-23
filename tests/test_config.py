"""Tests for closer.config: where settings come from, and that the key never leaks."""

from __future__ import annotations

import logging

import pytest

from closer import config
from closer.config import DEFAULT_MODEL, Settings

NAMES = ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_MATCH_MODEL", "GROQ_ZERO_RETENTION", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_nothing_is_set():
    settings = Settings.load(use_dotenv=False)
    assert settings.groq_api_key is None and not settings.has_ai
    assert settings.model == DEFAULT_MODEL and settings.effective_match_model == DEFAULT_MODEL
    assert settings.zero_retention is False


def test_environment_variables_are_read(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  key-from-env  ")
    monkeypatch.setenv("GROQ_MODEL", "small")
    monkeypatch.setenv("GROQ_MATCH_MODEL", "big")
    settings = Settings.load(use_dotenv=False)
    assert settings.groq_api_key == "key-from-env" and settings.has_ai
    assert (settings.model, settings.effective_match_model) == ("small", "big")


def test_a_gemini_key_is_not_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert Settings.load(use_dotenv=False).groq_api_key is None


def test_a_blank_key_counts_as_no_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")  # e.g. copied from .env.example unchanged
    assert not Settings.load(use_dotenv=False).has_ai


def test_streamlit_secrets_fill_gaps_but_the_environment_wins(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "from-env")
    settings = Settings.load({"GROQ_API_KEY": "from-secrets", "GROQ_MODEL": "from-secrets"}, use_dotenv=False)
    assert settings.groq_api_key == "from-secrets" and settings.model == "from-env"


def test_non_text_secret_values_do_not_crash():
    assert Settings.load({"GROQ_MODEL": 3}, use_dotenv=False).model == "3"  # e.g. an unquoted TOML number


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("1", True), ("YES", True), ("false", False), ("", False)])
def test_zero_retention_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("GROQ_ZERO_RETENTION", raw)
    assert Settings.load(use_dotenv=False).zero_retention is expected


def test_the_api_key_never_appears_in_repr_or_str():
    settings = Settings(groq_api_key="super-secret-key")
    assert "super-secret-key" not in repr(settings) and "super-secret-key" not in f"{settings}"


# --------------------------------------------------------------------------- .env file


def test_the_dotenv_file_is_read(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=key-from-dotenv\nGROQ_ZERO_RETENTION=true\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    settings = Settings.load()
    assert settings.groq_api_key == "key-from-dotenv" and settings.zero_retention


def test_real_environment_variables_beat_the_dotenv_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    monkeypatch.setenv("GROQ_MODEL", "from-shell")
    assert Settings.load().model == "from-shell"


def test_the_project_dotenv_path_is_next_to_the_package():
    assert config._ENV_FILE.name == ".env" and (config._ENV_FILE.parent / "closer").is_dir()


def test_a_missing_dotenv_is_fine(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "nope.env")
    assert not Settings.load().has_ai


def test_a_dotenv_that_cannot_be_read_because_the_library_is_missing_is_reported(monkeypatch, tmp_path, caplog):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=x\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE", env_file)
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)  # makes "import dotenv" fail
    with caplog.at_level(logging.WARNING, logger="closer.config"):
        settings = Settings.load()
    assert not settings.has_ai
    assert any("python-dotenv is not installed" in r.message for r in caplog.records)
