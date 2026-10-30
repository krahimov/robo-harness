import pytest

from studio import author_providers

REAL_CLAUDE_STATUS = author_providers.claude_status


@pytest.fixture(autouse=True)
def offline_claude(monkeypatch):
    # Never probe the host's Claude Code login from tests; opt in per test.
    monkeypatch.setattr(author_providers, "claude_status", lambda: {"id": "claude", "name": "Claude Code subscription",
        "configured": False, "message": "Test fixture"})


@pytest.fixture
def real_claude_status(monkeypatch):
    monkeypatch.setattr(author_providers, "claude_status", REAL_CLAUDE_STATUS)
    monkeypatch.setattr(author_providers, "_cached_claude", None)
    return REAL_CLAUDE_STATUS
