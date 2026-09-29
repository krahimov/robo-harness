import subprocess

import pytest

from studio import author_providers as providers


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(providers, "_cached_probe", None)
    monkeypatch.delenv("STUDIO_AUTHOR_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STUDIO_AUTHOR_API_KEY", raising=False)
    monkeypatch.delenv("STUDIO_CODEX_PATH", raising=False)
    monkeypatch.setattr(providers, "codex_binary", lambda: "/test/codex")


def test_saved_chatgpt_login_is_default_without_reading_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-api-key")
    monkeypatch.setenv("CODEX_API_KEY", "private-other-key")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command == ["/test/codex", "login", "status"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "CODEX_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "Logged in using ChatGPT")
    monkeypatch.setattr(providers.subprocess, "run", run)
    status = providers.connection_status()
    assert status["provider"] == "codex"
    assert status["configured"] is True
    assert [item["configured"] for item in status["providers"]] == [True, True]
    assert "private-api-key" not in str(status)
    assert len(calls) == 1  # Cache protects repeated status rendering.


@pytest.mark.parametrize("returncode,text", [(1, "Not logged in. Sign in with ChatGPT"), (0, "Logged in using an API key: private-key")])
def test_missing_chatgpt_login_does_not_claim_subscription_access(monkeypatch, returncode, text):
    monkeypatch.setattr(providers.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, text, ""))
    status = providers.codex_status()
    assert status["configured"] is False
    assert "private-key" not in str(status)
    assert providers.default_provider() == "openai"


def test_forced_provider_and_missing_cli_are_explicit(monkeypatch):
    monkeypatch.setattr(providers, "codex_binary", lambda: None)
    monkeypatch.setenv("STUDIO_AUTHOR_PROVIDER", "codex")
    monkeypatch.setenv("OPENAI_API_KEY", "available-api-key")
    status = providers.connection_status()
    assert status["provider"] == "codex" and not status["configured"]
    assert status["providers"][1]["configured"]
    assert "Install the Codex CLI" in status["providers"][0]["message"]
    monkeypatch.setenv("STUDIO_AUTHOR_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="must be auto"):
        providers.default_provider()


def test_login_probe_timeout_returns_safe_disconnected_status(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex login status private-string", 8)
    monkeypatch.setattr(providers.subprocess, "run", run)
    status = providers.codex_status()
    assert not status["configured"]
    assert "could not be checked" in status["message"]
    assert "private-string" not in str(status)
