"""Restore a trusted locally exported trial into a NEW isolated data directory."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import digest


def seed_trial(path, target):
    from studio import storage as db, task_api
    from studio.schemas import Scene
    from studio.adapters.simulation import make_simulator
    from studio.comparisons import ComparisonCreate
    target = target.resolve()
    if target.exists():
        raise ValueError('Choose a new data directory; existing data is never overwritten.')
    data = json.loads(path.read_text())
    if data.get('format') != 'robot-harness-cloud-trial-v1':
        raise ValueError('Unsupported cloud trial format.')
    spec = ComparisonCreate(**data['spec'])
    if spec.adapter != 'icrt' or data['judge']['kind'] != 'state':
        raise ValueError('Cloud trial must use ICRT and state checks.')
    previous = db.ROOT
    target.mkdir(parents=True)
    db.ROOT = target
    try:
        for asset in data['assets']:
            db.save('asset', asset)
        db.save('judge', data['judge'])
        if data['task'].get('source') != 'built-in':
            db.save('task', data['task'])
        if task_api.resolve_task(spec.scene, spec.task_id) != data['task']:
            raise ValueError('Task requirements differ from the original frozen experiment.')
        for item in data['demos']:
            artifact = copy.deepcopy(item['artifact'])
            if digest(artifact) != item['trajectory_sha256']:
                raise ValueError('Frozen demonstration integrity check failed.')
            scene = Scene(**item['record']['scene'])
            if Scene(**artifact['scene']) != scene or len(artifact['frames']) != len(artifact['actions'])+1:
                raise ValueError('Demonstration states/actions are inconsistent.')
            sim = make_simulator(scene)
            initial = sim.snapshot()
            original = artifact['frames'][0]
            if len(initial['qpos']) != len(original['qpos']) or [g['name'] for g in initial['geoms']] != [g['name'] for g in original['geoms']]:
                raise ValueError('Target robot geometry does not match the recorded demonstration.')
            if initial.get('robot_asset_commit') != original.get('robot_asset_commit'):
                raise ValueError('Robot asset revision differs from the reference.')
            artifact['xml'] = sim.source
            db.save('demo', {**item['record'], 'artifact_id': db.write_artifact(artifact)})
        expected_ids = {d for c in data['spec']['conditions'] for d in c['demo_ids']}
        if expected_ids != {d['record']['id'] for d in data['demos']}:
            raise ValueError('Comparison demonstration IDs do not match the fixture.')
        receipt = {'source_comparison_id': data['source_comparison_id'],
                   'fixture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'demonstration_hashes': {d['record']['id']: d['trajectory_sha256'] for d in data['demos']}}
        (target/'trial-receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    except Exception:
        shutil.rmtree(target)
        raise
    finally:
        db.ROOT = previous


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture', type=Path)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    seed_trial(args.fixture, args.data_dir)
