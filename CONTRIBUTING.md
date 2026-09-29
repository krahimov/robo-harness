# Contributing

Start with Python 3.12 or 3.13, Node.js 22 or newer, and uv. Run `python3 scripts/dev.py`, then open http://127.0.0.1:5178. The server binds to localhost. Environment variables in `.env.example` are optional; the example workflow needs no provider account.

Run `cd backend && uv run --frozen pytest -q` and `cd web && npm run build` before submitting a change. Tests use temporary data stores and controlled provider fixtures. CI runs without credentials and does not download model weights. For a local workflow check against the running server, use `python3 scripts/smoke.py`.

Keep the scene author, robotics actor, and judge separate. A successful scripted scene is authoring evidence. Claims about in-context learning require a learned actor with fixed weights and a controlled comparison. Name projected or approximate robot controls explicitly.

Task requirements belong to versioned human-authored records. A rubric explains how to inspect evidence; changing it must not change task requirements or prior evaluations. Tests should cover the behavioral risk of a change, including preservation of saved recordings and immutable experiment snapshots.

Add model-specific observations, demonstration preprocessing, and action conversion in an adapter. Document checkpoint provenance, embodiment requirements, action units, camera ordering, model limitations, and any upstream patches. Keep downloaded weights and third-party repositories in `.data/`, outside Git. Retain their licenses; the studio's MIT license does not relicense those materials.

Never commit API keys, saved login credentials, personal factory photos, or generated local recordings. Use a synthetic or clearly licensed reference for a reproducible issue. Include the operation, adapter, task/judge versions, seed, and relevant sanitized error. Avoid provider logs containing private prompts.

Pull requests should explain the observed problem, the resulting behavior, and validation. Existing `AGENTS.md` instructions apply to agent-assisted contributions as well.
