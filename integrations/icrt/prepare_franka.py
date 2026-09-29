"""Fetch the Apache-2.0 MuJoCo Menagerie Panda assets outside Git artifacts."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"


def main():
    path = ROOT / ".data/models/menagerie"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://github.com/google-deepmind/mujoco_menagerie.git", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", COMMIT], check=True)
        subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", "franka_emika_panda"], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Use the pinned Menagerie commit before validating the Franka fixture")
    print("Franka Panda assets available; upstream LICENSE remains in the model folder.")


if __name__ == "__main__":
    main()
