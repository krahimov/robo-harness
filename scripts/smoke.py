#!/usr/bin/env python3
"""Exercise a running local studio; live authoring only with --author-photo."""
import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8788/api')
    parser.add_argument('--author-photo', type=Path)
    parser.add_argument('--provider', choices=['codex', 'openai'], default='codex')
    args = parser.parse_args()

    def request(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(args.base_url.rstrip('/')+path, data=body,
            headers={'Content-Type': 'application/json'} if body else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Studio returned {error.code}: {error.read().decode()}') from None

    def wait(path, timeout=720):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            record = request(path)
            if record['status'] not in {'running', 'queued'}:
                if record['status'] not in {'completed', 'ready'}:
                    raise RuntimeError(record.get('error') or f"Run ended: {record['status']}")
                return record
            time.sleep(1)
        raise TimeoutError('The run is still active; inspect it in the studio.')

    bootstrap = request('/bootstrap')
    judge = next(j for j in bootstrap['judges'] if j['kind'] == 'state')
    if args.author_photo:
        mime = mimetypes.guess_type(args.author_photo.name)[0]
        if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
            raise ValueError('Choose a JPEG, PNG, or WebP photo')
        raw = args.author_photo.read_bytes()
        if len(raw) > 8*1024*1024:
            raise ValueError('The photo must be at most 8 MiB')
        image = request('/references', {'name': args.author_photo.name,
            'data_url': f'data:{mime};base64,'+base64.b64encode(raw).decode()})
        draft = request('/scenes', {'name': 'Factory photo author check',
            'prompt': (ROOT/'examples/factory-brief.txt').read_text(),
            'provider': args.provider, 'image_ids': [image['id']]})
        request('/scenes/'+draft['id']+'/generate', {})
        result = wait('/scenes/'+draft['id'])
        print(f"Authored scene {result['id']}: {result['verification']['status']}")
        print(result.get('summary', ''))
    demos = [request('/demos', {'name': f'Smoke context {i+1}',
        'scene': {'seed': 17+i, 'cube_x': x}}) for i, x in enumerate([-.22, -.29])]
    comparison = request('/comparisons', {'name': 'Replay context smoke', 'adapter': 'replay',
        'judge_id': judge['id'], 'episodes': 4, 'horizon_steps': 180,
        'scene': {'seed': 1000, 'variation': .04},
        'conditions': [{'name': 'One demonstration', 'demo_ids': [demos[0]['id']]},
            {'name': 'Two demonstrations', 'demo_ids': [d['id'] for d in demos]}]})
    finished = wait('/comparisons/'+comparison['id'], timeout=180)
    bundle = request('/comparisons/'+comparison['id']+'/export')
    folder = ROOT/'.data/smoke'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/(comparison['id']+'.json')
    output.write_text(json.dumps(bundle, indent=2))
    for c in finished['conditions']:
        print(f"{c['name']}: {c['counts']['pass']}/{c['resolved']} resolved passes, {c['unresolved']} unresolved")
    print(f'Evidence: {output}')
    print('Trajectory-replay workflow check; no learned robotics result is claimed.')


if __name__ == '__main__':
    main()
