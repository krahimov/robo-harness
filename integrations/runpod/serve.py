"""Supervise private model + simulation processes on the SAME GPU Pod."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.data_dir/'trial-receipt.json').exists():
        parser.error('Run seed_trial.py into this data directory first.')
    for port in (8788, 8790):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                parser.error(f'Port {port} is already occupied; no process was changed.')
    # Do not pass local/provider account credentials into either service.
    env = {k: v for k, v in os.environ.items() if k in {
        'PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES', '__EGL_VENDOR_LIBRARY_FILENAMES'}}
    env.update({'PYTHONPATH': str(ROOT/'backend')+os.pathsep+str(ROOT), 'PYTHONUNBUFFERED': '1',
        'STUDIO_DATA_DIR': str(args.data_dir.resolve()), 'MUJOCO_GL': 'egl',
        'STUDIO_ICRT_ENDPOINT': 'http://127.0.0.1:8790', 'STUDIO_UI_ORIGIN': 'http://127.0.0.1:8789'})
    children, logs = [], []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for name, command in [
            ('icrt', [str(ROOT/'integrations/icrt/.venv/bin/python'), str(ROOT/'integrations/icrt/worker.py'), '--device', 'cuda', '--port', '8790']),
            ('studio', [str(ROOT/'backend/.venv/bin/python'), '-m', 'uvicorn', 'integrations.runpod.cloud_app:app', '--host', '127.0.0.1', '--port', '8788'])]:
            log = (args.data_dir/(name+'.log')).open('ab')
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in children):
                raise RuntimeError('A service stopped during startup. Read the logs in the trial data directory.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8790/health', timeout=2) as r:
                    health = json.load(r)
                with urllib.request.urlopen('http://127.0.0.1:8788/api/bootstrap', timeout=2) as r:
                    json.load(r)
                if health.get('ready') and str(health.get('device')).startswith('cuda'):
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('CUDA model/API startup timed out.')
        print('Ready: CUDA model + simulation on this Pod. Access studio through an SSH tunnel.', flush=True)
        while all(p.poll() is None for p in children):
            time.sleep(1)
        raise RuntimeError('A service exited; stopping its sibling. Inspect the trial logs.')
    except KeyboardInterrupt:
        pass
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()
        print('Services stopped. Pod billing continues until you stop/terminate it in Runpod.', flush=True)


if __name__ == '__main__':
    main()
