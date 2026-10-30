#!/usr/bin/env python3
"""Run the robomimic-derived evals for each motion-program LLM on a running studio.

Each (task, model) pair is one ordinary experiment: bounded development
revisions, then a frozen program on fresh holdout seeds, scored by the task's
fixed state checks and the chosen judge. Results are also visible in the
Experiments tab. Model calls use the backend's Codex / Claude Code
subscriptions; this script never handles credentials.

  python scripts/run_model_evals.py                      # both models, lift + can
  python scripts/run_model_evals.py --models claude --tasks robomimic-can --holdouts 8
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"codex": "GPT-6 Astra", "claude": "Claude Opus 5.5"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788/api")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=["robomimic-lift", "robomimic-can"])
    parser.add_argument("--holdouts", type=int, default=4, help="final holdout episodes per run (1-8)")
    parser.add_argument("--attempts", type=int, default=3, help="development proposal budget (1-6)")
    parser.add_argument("--feedback", choices=["none", "checkpoints"], default="none")
    parser.add_argument("--judge-id", help="defaults to the local state-check judge")
    parser.add_argument("--demo-ids", nargs="*", default=[], help="up to four demonstrations as sampled context")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per experiment")
    args = parser.parse_args()

    def request(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(args.base_url.rstrip("/") + path, data=body,
            headers={"Content-Type": "application/json"} if body else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Studio returned {error.code}: {error.read().decode()}") from None

    boot = request("/bootstrap")
    actors = {a["id"]: a for a in next(a for a in boot["adapters"] if a["id"] == "astra").get("actors", [])}
    offline = [m for m in args.models if not actors.get(m, {}).get("configured")]
    if offline:
        raise SystemExit("Not signed in on the backend: " + ", ".join(f"{MODELS[m]} ({actors.get(m, {}).get('message', 'unknown')})" for m in offline))
    judge_id = args.judge_id or next(j["id"] for j in boot["judges"] if j["kind"] == "state")
    tasks = {t["id"]: t for t in boot["tasks"]}

    runs = []
    for task_id in args.tasks:
        task = tasks[task_id]
        for model in args.models:
            spec = {"name": f"Eval · {task['name']} · {MODELS[model]}", "adapter": "astra", "actor_provider": model,
                "task_id": task_id, "scene": task["suggested_scene"], "judge_id": judge_id, "demo_ids": args.demo_ids,
                "episodes": args.holdouts, "max_attempts": args.attempts, "feedback_mode": args.feedback,
                "instructions": task["description"]}
            run = request("/experiments", spec)
            print(f"queued {run['id']}  {spec['name']}", flush=True)
            runs.append((task_id, model, run["id"]))

    results = []
    for task_id, model, id in runs:
        deadline = time.monotonic() + args.timeout
        while True:
            run = request(f"/experiments/{id}")
            if run["status"] not in {"queued", "running"} or time.monotonic() > deadline:
                break
            time.sleep(5)
        usage = {"input_tokens": 0, "output_tokens": 0}
        for attempt in run.get("attempts", []):
            for key in usage:
                usage[key] += (attempt.get("usage") or {}).get(key, 0) or 0
        results.append({"task": task_id, "model": run.get("contract", {}).get("actor"), "provider": model, "experiment_id": id,
            "status": run["status"], "outcome": run.get("outcome"), "error": run.get("error"),
            "development_attempts": len(run.get("attempts", [])), "development_passed": run.get("frozen_program") is not None,
            "holdout_passes": run.get("holdout_passes", 0), "holdout_completed": run.get("holdout_completed", 0),
            "holdout_planned": len(run.get("holdout_seeds", [])), "usage": usage})

    out = ROOT / ".data/evals" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"settings": vars(args), "results": results}, indent=2))
    print("\n| Task | Model | Outcome | Dev attempts | Holdout passes | Tokens in/out |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['task']} | {MODELS[r['provider']]} | {r['outcome'] or r['status']} | {r['development_attempts']} | "
              f"{r['holdout_passes']}/{r['holdout_completed']} of {r['holdout_planned']} | {r['usage']['input_tokens']}/{r['usage']['output_tokens']} |")
    print(f"\nSaved {out.relative_to(ROOT)}. Holdout passes need both state checks and the judge; human review is separate. "
          "Scores are robosuite-inspired and not comparable to published robomimic success rates.")


if __name__ == "__main__":
    main()
