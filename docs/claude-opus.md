# Claude Opus 5.5 in the studio

Claude Opus 5.5 (`claude-opus-5-5`) is a second LLM next to GPT-6 Astra (`gpt-6-astra`). You can use it anywhere Astra runs, and you choose the model each time. The studio never substitutes one model for the other.

| Phase | Where to switch | What runs |
|---|---|---|
| Scene author / demonstrations | The model menu in the chat composer, or **Scene settings → Author model** | The same bounded build → simulate → inspect → revise loop, with photo references and simulator camera feedback |
| Robot actor | **Workspace → LLM motion programs → Motion-program LLM** | Development proposals, the frozen holdout program, and optional camera checkpoints within each episode |
| Judge | **Judges → Judge connection → Claude Opus 5.5 · Claude Code subscription** | The same evidence-tool loop and verdict validation as the Codex judge; rubric proposals too |

## Sign-in

The connection uses your Claude Code subscription session. On the backend machine, do one of the following:

- run `claude auth login`, or
- run `claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN` in the backend's environment.

Then click **Check connections**. The status probe runs `claude auth status --json`. It accepts only a first-party subscription login (`claude.ai` or `oauth_token`) and rejects API-key auth. `ANTHROPIC_API_KEY` is never passed to the child process, so calls count toward your Claude subscription usage.

The token stays in Claude Code's own store or in the backend environment. It is never written to records, artifacts, logs, or the frontend.

## Isolation

Every model call is one stateless `claude -p` turn:

- `--tools ""`: no shell, file, or web tools. Only the structured-output tool is available.
- `--json-schema`: the same reply schema as the Codex path (scene, motion program, judge, or rubric reply).
- `--system-prompt-file`: the studio's instructions replace Claude Code's default prompt.
- `--safe-mode --setting-sources "" --strict-mcp-config --disable-slash-commands`: no user settings, CLAUDE.md, hooks, skills, plugins, or MCP servers.
- `--no-session-persistence`: runs in a temporary folder that is deleted afterwards.

Images go to the model as base64 image blocks over `--input-format stream-json`. Cancellation, per-turn timeouts, and output limits work the same way as in the Codex session. Errors are reported as safe categories such as sign-in, usage limit, model unavailable, or CLI too old, so raw provider text is never shown.

## Records

Each record keeps the provider and exact model it was created with:

- Drafts store `provider: "claude"` and `model: "claude-opus-5-5"`.
- Experiments store `actor_provider`, and their frozen contract stores `actor` and `actor_provider`.
- Episode artifacts record `model.provider` and `model.model`.
- Checkpoint traces record `provider`.
- Judges pin `model: "claude-opus-5-5"`. Earlier judge versions and evaluations are unchanged.

The model has fixed weights. The motion-program actor is still state-assisted program control, not robotics in-context learning.

Code: `backend/studio/claude_session.py` (the CLI session) and `backend/studio/author_providers.py` (the registry of connections and pinned models).
