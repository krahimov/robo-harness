"""Run or resume the frozen ICRT comparison through the private studio tunnel."""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Client:
    def __init__(self, base):
        parts = urllib.parse.urlsplit(base)
        if parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost'} or parts.username or parts.password:
            raise ValueError('Use a loopback HTTP address through the authenticated SSH tunnel.')
        self.base = base.rstrip('/')

    def request(self, path, data=None):
        raw = None if data is None else json.dumps(data, allow_nan=False).encode()
        req = urllib.request.Request(self.base+path, data=raw, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)


def audit(bundle, artifacts, fixture, require_cuda=True):
    comparison, spec = bundle['comparison'], fixture['spec']
    errors = []
    for key in ('scene', 'episodes', 'horizon_steps', 'instructions', 'adapter', 'task_id'):
        if comparison[key] != spec[key]:
            errors.append(f'Changed experiment field: {key}')
    expected_context = [c['demo_ids'] for c in spec['conditions']]
    if [c['demo_ids'] for c in comparison['conditions']] != expected_context:
        errors.append('Context order changed')
    expected_seeds = list(range(spec['scene']['seed'], spec['scene']['seed']+spec['episodes']))
    if comparison['seeds'] != expected_seeds:
        errors.append('Evaluation seeds changed')
    expected = fixture['expected_model']
    traces = []
    for experiment in bundle['experiments']:
        if experiment['task'] != fixture['task'] or experiment['judge'] != fixture['judge']:
            errors.append('Task or judge snapshot changed')
        if experiment['status'] != 'completed' or len(experiment['results']) != spec['episodes']:
            errors.append('Experiment did not complete every planned episode')
        if [e['scene']['seed'] for e in experiment['results']] != expected_seeds:
            errors.append('Recorded episode seeds differ')
        for episode in experiment['results']:
            artifact = artifacts[episode['artifact_id']]
            model = artifact.get('model', {})
            worker = model.get('model', {})
            if model.get('fixed_weights') is not True or model.get('gradient_updates') != 0:
                errors.append('Missing fixed-weight evidence')
            if model.get('adapter_version') != expected['adapter_version']:
                errors.append('Studio adapter version changed')
            for key in ('policy_sha256', 'source_commit', 'runtime_adapter_version', 'action_reference'):
                if worker.get(key) != expected[key]:
                    errors.append(f'Model provenance changed: {key}')
            if require_cuda and not str(worker.get('device')).startswith('cuda'):
                errors.append('Episode did not use CUDA')
            if len(artifact.get('actions', [])) != spec['horizon_steps'] or len(artifact.get('predictions', [])) != spec['horizon_steps']:
                errors.append('Episode did not execute exactly one prediction per control step')
            if [c['id'] for c in model.get('context_sampling', [])] != experiment['demo_ids']:
                errors.append('Model context order differs from the experiment')
            traces.append({'episode_id': episode['id'], 'seed': episode['scene']['seed'],
                           'device': worker.get('device'), 'predictions': len(artifact.get('predictions', [])),
                           'controller_interventions': model.get('clipped_predictions'),
                           'context_sampling': model.get('context_sampling')})
    return {'valid': not errors, 'errors': sorted(set(errors)), 'traces': traces,
            'note': 'Infrastructure/contract audit. Task failures remain failures; no success-rate gain is assumed.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8789/api')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-cpu', action='store_true', help='Explicit local protocol validation only; not a GPU result')
    p.add_argument('--timeout', type=int, default=1800)
    args = p.parse_args()
    data = json.loads(args.fixture.read_text())
    fixture_hash = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    client = Client(args.base_url)
    receipt = client.request('/cloud/trial')
    if receipt['fixture_sha256'] != fixture_hash:
        raise ValueError('Remote sandbox was seeded from a different trial fixture.')
    bootstrap = client.request('/bootstrap')
    actor = next(a for a in bootstrap['adapters'] if a['id'] == 'icrt')
    if not actor['ready']:
        raise ValueError('ICRT is not ready on the sandbox.')
    if not args.allow_cpu and not str(actor['details']['device']).startswith('cuda'):
        raise ValueError('The sandbox must report CUDA. No CPU fallback was used.')
    for key in ('policy_sha256', 'source_commit', 'runtime_adapter_version', 'action_reference'):
        if actor['details'].get(key) != data['expected_model'][key]:
            raise ValueError(f'The sandbox has different model provenance: {key}')
    state_path = args.output/'run.json'
    if args.resume:
        state = json.loads(state_path.read_text())
        if state['fixture_sha256'] != fixture_hash or state['base_url'] != args.base_url:
            raise ValueError('Resume target or fixture differs from the saved run.')
        comparison_id = state['comparison_id']
    else:
        # Reserve output before POST; an uncertain response must be inspected, not retried blindly.
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'submission.json').write_text(json.dumps({'fixture_sha256': fixture_hash, 'base_url': args.base_url}))
        comparison = client.request('/comparisons', data['spec'])
        comparison_id = comparison['id']
        state_path.write_text(json.dumps({'comparison_id': comparison_id, 'fixture_sha256': fixture_hash, 'base_url': args.base_url}, indent=2))
    print(f'Comparison {comparison_id}. Results stay on the Pod and are downloaded here: {args.output}', flush=True)
    started, last = time.monotonic(), None
    try:
        while time.monotonic()-started < args.timeout:
            comparison = client.request('/comparisons/'+comparison_id)
            progress = [(c['name'], c['recorded'], c['status']) for c in comparison['conditions']]
            if progress != last:
                print(json.dumps(progress), flush=True)
                last = progress
            if comparison['status'] not in {'running', 'queued'}:
                break
            time.sleep(3)
        else:
            client.request('/comparisons/'+comparison_id+'/cancel', {})
            raise TimeoutError('Trial deadline reached; cancellation requested. Use --resume to collect final results.')
    except KeyboardInterrupt:
        client.request('/comparisons/'+comparison_id+'/cancel', {})
        raise SystemExit('Cancellation requested. Use --resume to download the partial results.')
    bundle = client.request('/comparisons/'+comparison_id+'/export')
    (args.output/'comparison.json').write_text(json.dumps(bundle, indent=2))
    artifacts = {}
    folder = args.output/'artifacts'
    folder.mkdir(exist_ok=True)
    for experiment in bundle['experiments']:
        for episode in experiment['results']:
            id = episode['artifact_id']
            artifacts[id] = client.request('/artifacts/'+id)
            (folder/(id+'.json')).write_text(json.dumps(artifacts[id], separators=(',', ':')))
    result = audit(bundle, artifacts, data, require_cuda=not args.allow_cpu)
    result['wall_seconds_this_invocation'] = round(time.monotonic()-started, 3)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2))
    for c in comparison['conditions']:
        print(f'{c["name"]}: {c["counts"]}', flush=True)
    print('Contract audit:', 'PASS' if result['valid'] else result['errors'])
    print('Compute billing continues until the Pod is stopped/terminated in Runpod.')
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
