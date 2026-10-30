"""Fetch the Apache-2.0 MuJoCo Menagerie UR5e, Robotiq 2F-85, Allegro hand, and TidyBot assets outside Git artifacts.

Uses the same pinned sparse checkout as prepare_franka.py and adds the extra model folders.
"""
from pathlib import Path
import subprocess

from prepare_franka import COMMIT, ROOT, main as prepare_franka

FOLDERS = ["universal_robots_ur5e", "robotiq_2f85", "wonik_allegro", "stanford_tidybot"]


def main():
    prepare_franka()
    path = ROOT / ".data/models/menagerie"
    subprocess.run(["git", "-C", str(path), "sparse-checkout", "add", *FOLDERS], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT or not all((path / folder / "LICENSE").is_file() for folder in FOLDERS):
        raise ValueError("Use the pinned Menagerie commit before validating the arm fixtures")
    print("UR5e, Robotiq 2F-85, Allegro, and TidyBot assets available; upstream LICENSE files remain in each model folder.")


if __name__ == "__main__":
    main()
