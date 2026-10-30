"""Subscription-backed model calls through the local Claude Code runtime.

Claude Code owns sign-in and token refresh: either its saved login or a
CLAUDE_CODE_OAUTH_TOKEN created with `claude setup-token`. The studio never
reads, stores, or logs that credential. Every call is a stateless `claude -p`
turn with all tools, settings, hooks, MCP servers, and skills disabled, so the
model only returns a schema-bound reply that our simulator validates.
"""

import base64
import json
import os
import subprocess
import time

from studio.authoring import AuthorError, check_cancel
from studio.codex_author import MAX_EVENT_BYTES, MAX_REPLY_BYTES, TURN_TIMEOUT_SECONDS, CodexSession

MODEL = "claude-opus-5-5"


def claude_environment():
    # Keep saved-login discovery, the subscription OAuth token, and platform/network
    # settings. API keys are never inherited, so usage stays on the subscription.
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
        "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR", "APPDATA", "LOCALAPPDATA",
        "USERPROFILE", "XDG_CONFIG_HOME", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "HTTPS_PROXY", "HTTP_PROXY",
        "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "all_proxy", "no_proxy"}
    return {**{key: value for key, value in os.environ.items() if key in allowed},
        "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


def _safe_failure(events):
    # Classify runtime errors without returning raw messages, prompts, or paths.
    lowered = events.lower()
    if any(word in lowered for word in ("rate_limit", "usage limit", "usage_limit", "quota", "429", "overloaded")):
        return "Your Claude subscription reached a usage limit. Retry after the limit resets. No API key was used."
    if any(word in lowered for word in ("not_found_error", "model_not_found", "invalid model", "model not found", "not available for", "does not exist")):
        return "Claude Opus 5.5 is unavailable for this Claude Code login. No alternate model was used."
    if any(word in lowered for word in ("authentication", "not logged in", "401", "oauth", "/login", "unauthorized")):
        return "Claude Code needs a fresh sign-in. Run claude auth login or refresh CLAUDE_CODE_OAUTH_TOKEN on the backend machine, then retry."
    if any(word in lowered for word in ("unknown option", "error: option", "unexpected argument")):
        return "This Claude Code CLI does not support the studio's isolated options. Update Claude Code and retry."
    return "Claude Code could not complete the Opus 5.5 request. Check its sign-in, model access, and network connection, then retry."


def _media_type(data):
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def _usage(value):
    usage = value if isinstance(value, dict) else {}
    counts = {name: usage.get(name, 0) for name in ("input_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens", "output_tokens")}
    counts = {name: count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0
        for name, count in counts.items()}
    return {"input_tokens": counts["input_tokens"] + counts["cache_creation_input_tokens"] + counts["cache_read_input_tokens"],
        "output_tokens": counts["output_tokens"]}


class ClaudeSession(CodexSession):
    """Run bounded, stateless `claude -p` calls using the saved Claude Code login."""

    provider = "claude"
    model = MODEL

    def __enter__(self):
        from studio.author_providers import claude_binary, claude_status
        self.binary = claude_binary()
        if not self.binary or not claude_status()["configured"]:
            raise AuthorError("Sign in with claude auth login, or export CLAUDE_CODE_OAUTH_TOKEN, on the backend machine to use your Claude subscription.")
        return self

    def command(self, *_):
        return [self.binary, "-p", "--model", self.model, "--effort", "medium",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--json-schema", self.schema.read_text(encoding="utf-8"),
            "--system-prompt-file", str(self.instructions), "--tools", "",
            "--setting-sources", "", "--strict-mcp-config", "--safe-mode",
            "--disable-slash-commands", "--no-session-persistence",
            "--permission-mode", "dontAsk", "--permission-prompts", "none"]

    def message(self, prompt, photos):
        content = [{"type": "text", "text": prompt}]
        for index, (label, path) in enumerate(photos):
            data = path.read_bytes()
            content.append({"type": "text", "text": f"Attached image {index + 1}: {label}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": _media_type(data),
                "data": base64.b64encode(data).decode()}})
        return json.dumps({"type": "user", "message": {"role": "user", "content": content}})

    def run(self, prompt, photos, event):
        check_cancel(event)
        self.round += 1
        events_path = self.folder / f"claude-events-{self.round}.jsonl"
        error_path = self.folder / f"claude-error-{self.round}.txt"
        input_path = self.folder / f"claude-input-{self.round}.jsonl"
        input_path.write_text(self.message(prompt, photos) + "\n", encoding="utf-8")
        deadline = min(self.deadline, time.monotonic() + TURN_TIMEOUT_SECONDS)
        timeout = "Claude Opus 5.5 timed out through Claude Code. Your work is saved; retry."
        too_large = "Claude Code exceeded the response limit. Simplify the request and retry."
        if time.monotonic() >= deadline:
            raise AuthorError(timeout)
        try:
            with input_path.open("rb") as source, events_path.open("wb") as output, error_path.open("wb") as errors:
                self.process = subprocess.Popen(self.command(), stdin=source, stdout=output, stderr=errors,
                    cwd=self.folder, env=claude_environment(), start_new_session=os.name == "posix")
                while self.process.poll() is None:
                    check_cancel(event)
                    if time.monotonic() >= deadline:
                        raise AuthorError(timeout)
                    if events_path.stat().st_size + error_path.stat().st_size > MAX_EVENT_BYTES:
                        raise AuthorError(too_large)
                    event.wait(0.1)
                check_cancel(event)
                return_code = self.process.returncode
            if events_path.stat().st_size + error_path.stat().st_size > MAX_EVENT_BYTES:
                raise AuthorError(too_large)
            raw_events = events_path.read_text(encoding="utf-8", errors="replace")
            result = None
            for line in raw_events.splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and item.get("type") == "result":
                    result = item
            if result is None or result.get("is_error") or result.get("subtype") != "success":
                raise AuthorError(_safe_failure(raw_events + error_path.read_text(encoding="utf-8", errors="replace")))
            if return_code:
                raise AuthorError(_safe_failure(raw_events))
            reply = result.get("structured_output")
            if isinstance(reply, (dict, list)):
                reply = json.dumps(reply, ensure_ascii=False)
            elif not isinstance(reply, str):
                reply = result.get("result")
            if not isinstance(reply, str) or not reply.strip():
                raise AuthorError("Claude Code returned no completed response. Your work is saved; retry.")
            if len(reply.encode()) > MAX_REPLY_BYTES:
                raise AuthorError(too_large)
            return reply, _usage(result.get("usage"))
        except OSError:
            raise AuthorError("Could not start Claude Code. Check the backend CLI installation and permissions.") from None
        finally:
            self._stop()
