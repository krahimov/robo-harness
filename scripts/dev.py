#!/usr/bin/env python3
"""Start both local processes and shut them down together on Ctrl+C."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
children = []


def stop(*args):
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    try:
        subprocess.run(["uv", "sync", "--frozen"], cwd=ROOT / "backend", check=True)
        if not (ROOT / "web/node_modules").exists():
            subprocess.run(["npm", "ci"], cwd=ROOT / "web", check=True)
        children.append(subprocess.Popen([str(ROOT / "backend/.venv/bin/python"), "-m", "uvicorn", "studio.main:app", "--host", "127.0.0.1", "--port", "8788"], cwd=ROOT / "backend", start_new_session=True))
        children.append(subprocess.Popen(["npm", "run", "dev", "--", "--strictPort"], cwd=ROOT / "web", start_new_session=True))
        print("\nStudio: http://127.0.0.1:5178\nAPI: http://127.0.0.1:8788/docs\n", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise SystemExit(next((child.returncode for child in children if child.returncode), 1))
    except KeyboardInterrupt:
        pass
    finally:
        stop()
