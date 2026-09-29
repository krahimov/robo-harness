"""Check a locally generated upload against its transfer manifest."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root, manifest, fixture):
    root = root.resolve()
    data = json.loads(manifest.read_text())
    for name, expected in data['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Invalid manifest path.')
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Manifest file escapes the source directory.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != data['fixture_sha256']:
        raise ValueError('Trial fixture hash mismatch.')
    return len(data['files'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--manifest', type=Path, default=Path('manifest.json'))
    p.add_argument('--fixture', type=Path, default=Path('trial.json'))
    args = p.parse_args()
    print(f'Verified {verify(args.root, args.manifest, args.fixture)} files and the frozen trial.')
