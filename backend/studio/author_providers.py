"""Discover author connections without reading or returning credentials."""
import os
import shutil
import subprocess
import threading
import time
from typing import Literal

from studio.authoring import api_key

Provider = Literal["codex", "openai"]
_cache_lock = threading.Lock()
_cached_probe: tuple[float, str, dict] | None = None


def codex_binary():
    configured = os.environ.get("STUDIO_CODEX_PATH", "").strip()
    return shutil.which(configured or "codex")


def refresh_codex_status():
    global _cached_probe
    with _cache_lock:
        _cached_probe = None


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
    if provider != "openai":
        raise ValueError("Choose codex or openai for the scene author")
    configured = bool(api_key())
    return {"id": "openai", "name": "OpenAI API", "configured": configured,
        "message": "Using the backend OpenAI API key." if configured else "Set OPENAI_API_KEY on the backend and restart it to use the OpenAI API connection."}


def default_provider() -> Provider:
    configured = os.environ.get("STUDIO_AUTHOR_PROVIDER", "auto").strip().lower()
    if configured in {"codex", "openai"}:
        return configured
    if configured != "auto":
        raise ValueError("STUDIO_AUTHOR_PROVIDER must be auto, codex, or openai")
    return "codex" if codex_status()["configured"] else "openai"


def connection_status():
    selected = default_provider()
    providers = [provider_status("codex"), provider_status("openai")]
    return {"provider": selected, "configured": next(item["configured"] for item in providers if item["id"] == selected), "providers": providers}
