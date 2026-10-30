"""Discover author connections without reading or returning credentials."""
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Literal

from studio.authoring import api_key

Provider = Literal["codex", "openai", "claude"]
SubscriptionProvider = Literal["codex", "claude"]
_cache_lock = threading.Lock()
_cached_probe: tuple[float, str, dict] | None = None
_cached_claude: tuple[float, str, dict] | None = None

# Each connection pins one exact model. No connection silently substitutes another.
MODELS = {"codex": "gpt-6-astra", "openai": "gpt-6-astra", "claude": "claude-opus-5-5"}
MODEL_NAMES = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5"}
SUBSCRIPTION_PROVIDERS = ("codex", "claude")


def model_for(provider):
    if provider not in MODELS:
        raise ValueError("Choose codex, claude, or openai")
    return MODELS[provider]


def model_name(provider):
    return MODEL_NAMES[model_for(provider)]


def session_factory(provider, reply_model=None):
    """Return the stateless CLI session class for a subscription connection."""
    if provider == "claude":
        from studio.claude_session import ClaudeSession as session
    elif provider == "codex":
        from studio.codex_author import CodexSession as session
    else:
        raise ValueError("Choose codex or claude for a subscription connection")
    if reply_model is None:
        return session
    return lambda folder, instructions: session(folder, instructions, reply_model=reply_model)


def subscription_status(provider):
    if provider == "claude":
        return claude_status()
    if provider == "codex":
        return codex_status()
    raise ValueError("Choose codex or claude for a subscription connection")


def codex_binary():
    configured = os.environ.get("STUDIO_CODEX_PATH", "").strip()
    return shutil.which(configured or "codex")


def claude_binary():
    configured = os.environ.get("STUDIO_CLAUDE_PATH", "").strip()
    return shutil.which(configured or "claude")


def refresh_codex_status():
    global _cached_probe, _cached_claude
    with _cache_lock:
        _cached_probe = None
        _cached_claude = None


def claude_status():
    """Accept only Claude subscription auth: a saved login or CLAUDE_CODE_OAUTH_TOKEN."""
    global _cached_claude
    binary = claude_binary()
    base = {"id": "claude", "name": "Claude Code subscription"}
    if not binary:
        return {**base, "configured": False,
            "message": "Install Claude Code, then run claude auth login or export CLAUDE_CODE_OAUTH_TOKEN from claude setup-token on the backend machine."}
    with _cache_lock:
        if _cached_claude and _cached_claude[0] > time.monotonic() and _cached_claude[1] == binary:
            return dict(_cached_claude[2])
        from studio.claude_session import claude_environment
        try:
            result = subprocess.run([binary, "auth", "status", "--json"], env=claude_environment(),
                capture_output=True, text=True, timeout=8)
            state = json.loads(result.stdout or "{}")
            method = state.get("authMethod") if isinstance(state, dict) else None
            signed_in = (result.returncode == 0 and state.get("loggedIn") is True
                and state.get("apiProvider", "firstParty") == "firstParty" and method not in {None, "none", "api_key"})
            message = ("Using your Claude Code subscription sign-in. Model calls count toward your Claude usage." if signed_in else
                "Run claude auth login, or export CLAUDE_CODE_OAUTH_TOKEN from claude setup-token, on the backend machine. API keys are not used for this connection.")
        except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
            signed_in = False
            message = "Claude Code sign-in status could not be checked. Verify the local CLI and run claude auth login."
        status = {**base, "configured": signed_in, "message": message}
        _cached_claude = (time.monotonic() + 10, binary, status)
        return dict(status)


def codex_status():
    global _cached_probe
    binary = codex_binary()
    if not binary:
        return {"id": "codex", "name": "Codex subscription", "configured": False,
            "message": "Install the Codex CLI, then sign in with codex login on the backend machine."}
    with _cache_lock:
        if _cached_probe and _cached_probe[0] > time.monotonic() and _cached_probe[1] == binary:
            return dict(_cached_probe[2])
        env = dict(os.environ)
        # This connection specifically selects subscription auth, even if an API
        # key is available for the other provider. Codex owns its login cache.
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)
        env.pop("CODEX_ACCESS_TOKEN", None)
        env["RUST_LOG"] = "error"
        try:
            result = subprocess.run([binary, "login", "status"], env=env, capture_output=True, text=True, timeout=8)
            output = (result.stdout + result.stderr).lower()
            signed_in = result.returncode == 0 and "chatgpt" in output
            message = "Using your saved Codex ChatGPT login. Model calls count toward your Codex usage." if signed_in else "Run codex login on the backend machine and sign in with ChatGPT, then check the connection."
        except (OSError, subprocess.TimeoutExpired):
            signed_in = False
            message = "Codex login status could not be checked. Verify the local CLI and run codex login."
        status = {"id": "codex", "name": "Codex subscription", "configured": signed_in, "message": message}
        _cached_probe = (time.monotonic() + 10, binary, status)
        return dict(status)


def provider_status(provider: Provider):
    if provider == "codex":
        return codex_status()
    if provider == "claude":
        return claude_status()
    if provider != "openai":
        raise ValueError("Choose codex, claude, or openai for the scene author")
    configured = bool(api_key())
    return {"id": "openai", "name": "OpenAI API", "configured": configured,
        "message": "Using the backend OpenAI API key." if configured else "Set OPENAI_API_KEY on the backend and restart it to use the OpenAI API connection."}


def default_provider() -> Provider:
    configured = os.environ.get("STUDIO_AUTHOR_PROVIDER", "auto").strip().lower()
    if configured in {"codex", "openai", "claude"}:
        return configured
    if configured != "auto":
        raise ValueError("STUDIO_AUTHOR_PROVIDER must be auto, codex, claude, or openai")
    return "codex" if codex_status()["configured"] else "openai"


def connection_status():
    selected = default_provider()
    providers = [{**provider_status(name), "model": MODELS[name], "model_name": model_name(name)}
        for name in ("codex", "openai", "claude")]
    return {"provider": selected, "configured": next(item["configured"] for item in providers if item["id"] == selected), "providers": providers}
