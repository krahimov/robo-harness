"""Fetch pinned upstream ICRT files and apply narrow CPU portability fixes.

Downloads stay in .data/models. Original upstream licenses remain in the clone;
this script never evaluates checkpoint YAML or downloads executable model code
through torch.hub. Run from the repository: python integrations/icrt/prepare.py.
"""
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "e0c95887d577c95ef9d0eef5ee23960043d6cb5e"
MODEL_COMMIT = "5cb0391"
FILES = {
    "policy.pth": ("icrt_vitb_droid_pretrained/icrt_vitb_droid_pretrained.pth", "edaf5b17fa3c61e5ff0cf3dc254c8f8573197c607a63a39c7073b6066a69a545"),
    "cross-mae-rtx-vitb.pth": ("crossmae_rtx/cross-mae-rtx-vitb.pth", "2743c5a1ba4cbe296870a2f12a64d2653e7c257c7e8d25c3e2b197ace461f8fe"),
    "run.yaml": ("icrt_vitb_droid_pretrained/run.yaml", "408f0cc1cd1a4c285c2a7f57b1055df7387314d098e04c48dd34b85bd90f4c5e"),
}


def patch_source(source):
    changes = {
        "icrt/models/policy/icrt.py": [
            ("torch.set_default_tensor_type(torch.cuda.HalfTensor)", "# Studio portability: construct on CPU; device is selected by the worker."),
            ("torch.set_default_tensor_type(torch.FloatTensor)", "# Studio portability: retain the caller's default tensor type."),
            ("torch.amp.autocast('cuda', dtype=torch.bfloat16)", "torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=next(self.parameters()).is_cuda)"),
        ],
        "icrt/models/policy/llama.py": [
            (").cuda()", ").to(device=self.wq.weight.device, dtype=self.wq.weight.dtype)"),
        ],
    }
    for relative, replacements in changes.items():
        path = source / relative
        text = path.read_text()
        for old, new in replacements:
            if old not in text and new not in text:
                raise RuntimeError(f"Upstream source changed unexpectedly: {relative}")
            text = text.replace(old, new)
        path.write_text(text)


def main():
    folder = ROOT / ".data/models/icrt"
    source = ROOT / ".data/models/icrt-source"
    folder.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(["git", "clone", "https://github.com/Max-Fu/icrt.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", SOURCE_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected upstream commit {SOURCE_COMMIT}; use a clean checkout of that commit")
    manifest = {"source": "https://github.com/Max-Fu/icrt", "source_commit": SOURCE_COMMIT,
        "model": "mlfu7/ICRT", "model_revision": MODEL_COMMIT, "files": {},
        "modifications": "CPU construction; cache allocation follows model device; disable CUDA autocast on CPU. No weight updates."}
    for name, (remote, expected) in FILES.items():
        path = folder / name
        # Reuse a completed initial download under its temporary descriptive name.
        if name == "cross-mae-rtx-vitb.pth" and not path.exists() and (folder / "vision.pth").exists():
            (folder / "vision.pth").rename(path)
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(f"https://huggingface.co/mlfu7/ICRT/resolve/{MODEL_COMMIT}/{remote}", temporary)
            temporary.rename(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected and actual != expected:
            raise RuntimeError(f"Checkpoint integrity check failed: {name}")
        manifest["files"][name] = {"sha256": actual, "bytes": path.stat().st_size}
    patch_source(source)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("ICRT checkpoints verified and CPU portability changes applied.")


if __name__ == "__main__":
    main()
