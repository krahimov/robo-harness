"""Build an explicit source + frozen-comparison bundle without local credentials."""
import argparse
import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ('backend/studio', 'integrations/icrt', 'integrations/runpod', 'web/dist')
EXCLUDED = {'.venv', '__pycache__', '.pytest_cache', '.git', 'node_modules'}
SOURCE_FILES = ('backend/pyproject.toml', 'backend/uv.lock', 'LICENSE')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fixture(data_dir, comparison_id):
    con = sqlite3.connect(f'{(data_dir / "studio.sqlite").resolve().as_uri()}?mode=ro', uri=True)
    def get(kind, id):
        row = con.execute('SELECT body FROM records WHERE kind=? AND id=?', (kind, id)).fetchone()
        if row is None:
            raise ValueError(f'Missing {kind}: {id}')
        return json.loads(row[0])
    try:
        comparison = get('comparison', comparison_id)
        experiments = [get('experiment', c['experiment_id']) for c in comparison['conditions']]
        if comparison['adapter'] != 'icrt' or not experiments or any(e['status'] != 'completed' for e in experiments):
            raise ValueError('Select a completed ICRT comparison.')
        first = experiments[0]
        if first['judge']['kind'] != 'state':
            raise ValueError('The first cloud trial requires a numerical state judge.')
        keys = ('scene', 'episodes', 'horizon_steps', 'instructions', 'task', 'judge')
        if any(any(e[k] != first[k] for k in keys) for e in experiments):
            raise ValueError('Comparison snapshots differ in more than context.')
        demos, assets, expected = {}, {}, None
        for e in experiments:
            for d in e['context']:
                artifact = json.loads((data_dir / 'artifacts' / (d['artifact_id']+'.json')).read_text())
                # XML contains machine-specific paths. Rebuild it on the target host.
                artifact = {k: artifact[k] for k in ('scene', 'task', 'frames', 'actions', 'frequency', 'simulator_version') if k in artifact}
                record = {k: d[k] for k in ('id', 'name', 'version', 'scene', 'duration', 'frames', 'source', 'embodiment', 'task', 'status', 'motion_profile', 'task_id') if k in d}
                item = {'record': record, 'artifact': artifact, 'trajectory_sha256': digest(artifact)}
                if d['id'] in demos and demos[d['id']] != item:
                    raise ValueError('The same demo ID has different frozen trajectories.')
                demos[d['id']] = item
        for scene in [comparison['scene'], *[d['record']['scene'] for d in demos.values()]]:
            for prop in scene.get('props', []):
                if prop.get('asset_id'):
                    asset = get('asset', prop['asset_id'])
                    assets[asset['id']] = {k: asset[k] for k in ('id', 'name', 'kind', 'vertices', 'faces', 'size', 'vertex_count', 'face_count') if k in asset}
        experiment_ids = {e['id'] for e in experiments}
        for row in con.execute("SELECT body FROM records WHERE kind='episode'"):
            episode = json.loads(row[0])
            if episode['experiment_id'] not in experiment_ids:
                continue
            artifact = json.loads((data_dir / 'artifacts' / (episode['artifact_id']+'.json')).read_text())
            model = artifact['model']
            provenance = {k: model['model'][k] for k in ('policy_sha256', 'source_commit', 'runtime_adapter_version', 'action_reference')}
            provenance['adapter_version'] = model['adapter_version']
            if expected is not None and provenance != expected:
                raise ValueError('Reference episodes used different model versions.')
            expected = provenance
        if expected is None:
            raise ValueError('Reference comparison has no model provenance.')
        spec = {k: comparison[k] for k in ('name', 'judge_id', 'adapter', 'scene', 'episodes', 'horizon_steps', 'instructions', 'task_id')}
        spec['name'] = ('Cloud / '+comparison['name'])[:100]
        spec['conditions'] = [{k: c[k] for k in ('name', 'demo_ids')} for c in comparison['conditions']]
        return {'format': 'robot-harness-cloud-trial-v1', 'source_comparison_id': comparison_id,
                'spec': spec, 'task': first['task'], 'judge': first['judge'], 'demos': list(demos.values()),
                'assets': list(assets.values()), 'expected_model': expected,
                'note': 'Frozen demonstration states/actions; target recompiles geometry. GPU and renderer differences may change results.'}
    finally:
        con.close()


def source_files(root):
    paths = [root/p for p in SOURCE_FILES]
    for folder in SOURCE_DIRS:
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file()
                     and not p.is_symlink() and not any(x in EXCLUDED for x in p.relative_to(root).parts)
                     and p.suffix not in {'.pyc', '.log'} and not p.name.startswith('.env'))
    return sorted(set(paths))


def build(output, data_dir, comparison_id):
    if not (ROOT/'web/dist/index.html').is_file():
        raise ValueError('Build the frontend first: cd web && npm run build')
    data = fixture(data_dir, comparison_id)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'trial.json'
    target.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    manifest = {}
    with tarfile.open(output/'studio-source.tar.gz', 'w:gz') as archive:
        for path in source_files(ROOT):
            name = path.relative_to(ROOT).as_posix()
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    (output/'manifest.json').write_text(json.dumps({'files': manifest, 'fixture_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))
    print(f'Prepared {len(manifest)} source/build files and {len(data["demos"])} frozen demonstrations in {output}')
    print('Excluded: credentials, photos, local database, model weights, unrelated scenes and conversations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'.data')
    parser.add_argument('--output', type=Path, default=ROOT/'.data/runpod/upload')
    args = parser.parse_args()
    build(args.output, args.data_dir, args.comparison)
