"""Auditable fixed-weight ICRT experiments with separate diagnostic tasks.

Run on the GPU host with backend/.venv/bin/python. The worker must be ready.
Each candidate writes new demonstrations, rollout traces, and numerical verdicts.
No model weights or task thresholds are modified. Teacher controls are labeled.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
import httpx
import numpy as np
from studio.schemas import Scene, task_for_scene
from studio.adapters.simulation import make_simulator, default_waypoints, record
from studio.adapters.icrt import pose, demonstration_indices, ADAPTER_VERSION
from studio.adapters.icrt_frames import flange_to_tip
from studio.adapters.icrt_render import Cameras
from studio.judges import state_judge


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def camera_xml(source, view):
    if view == 'original':
        return source
    root = ET.fromstring(source)
    cam = next(c for c in root.iter('camera') if c.get('name') == 'overview')
    position = np.array([.65, -.75, .65]) if view == 'close' else np.array([.25, -.8, .5])
    target = np.array([0., 0., .05])
    z = position-target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam.set('pos', ' '.join(map(str, position)))
    cam.set('xyaxes', ' '.join(map(str, np.r_[x,y])))
    cam.set('fovy', '50')
    return ET.tostring(root, encoding='unicode')


def metrics(artifact):
    frames = artifact['frames']
    return {'verdict': state_judge(artifact, {'require_review': True}),
        'grasp_frames': sum(bool(f['attached']) for f in frames),
        'max_cube_z': max(f['cube'][2] for f in frames),
        'min_gripper_distance': min(float(np.linalg.norm(np.array(f['ee'])-f['cube'])) for f in frames),
        'final_cube': frames[-1]['cube'],
        'max_ik_error': max(f.get('ik_error',{}).get('position', 0) if isinstance(f.get('ik_error'),dict) else f.get('ik_position_error',0) for f in frames)}


def experiment(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['studio_adapter_version'] = ADAPTER_VERSION
    config['model_pose_frame'] = 'panda_link8'
    config['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['output'] = str(output)
    if config.get('demo_artifacts'):
        config['demo_artifacts'] = [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in config['demo_artifacts']]
    fixture_path = ROOT/'trial.json'
    if not fixture_path.exists():
        fixture_path = ROOT/'.data/runpod/gpu-trial-1/trial.json'
    fixture = json.loads(fixture_path.read_text()) if args.task == 'factory' else None
    if fixture:
        scene = Scene.model_validate(fixture['spec']['scene'])
    else:
        scene = Scene(embodiment='franka-panda-v1', variation=.02, cube_x=getattr(args,'cube_x',-.22), cube_y=getattr(args,'cube_y',-.08),
            table_color=[.7,.65,.55,1] if args.view != 'original' else [.23,.29,.26,1])
    task = fixture['task'] if fixture else task_for_scene(scene)
    config.update(task=task, task_sha256=digest(task), fixed_weights=True, actor='icrt', gradient_updates=0)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    camera = Cameras()
    client = httpx.Client(base_url=args.endpoint, timeout=180)
    session = None
    try:
        converted, demo_results = [], []
        for i in range(args.demos):
            ds = scene.model_copy(update={'seed': 42+i, 'variation': args.demo_variation})
            waypoints = default_waypoints(ds)
            for w in waypoints:
                w['seconds'] *= args.pace
            artifact = json.loads(args.demo_artifacts[i].read_text()) if getattr(args, 'demo_artifacts', []) else record(ds, waypoints)
            if getattr(args, 'demo_artifacts', []):
                artifact['xml'] = make_simulator(Scene.model_validate(artifact['scene'])).source
            artifact['task'] = task
            artifact['xml'] = camera_xml(artifact['xml'], args.view)
            measured = metrics(artifact)
            demo_results.append(measured)
            (output/f'teacher-{i}.json').write_text(json.dumps(artifact))
            if measured['verdict']['status'] != 'pass':
                report = {'candidate':config,'teacher':demo_results,'results':[],'successes':0,'episodes':0,'status':'rejected_demo','reason':f'Teacher {i} failed; actor not run'}
                (output/'report.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(report),flush=True)
                return report
            indexes = demonstration_indices(artifact, 256//args.demos)
            frames = [artifact['frames'][j] for j in indexes]
            images = camera.render({'xml': artifact['xml'], 'frames': frames})
            observations = [{**im, 'proprio': pose(f, base=f['robot_base'], projected=False)} for f,im in zip(frames,images)]
            actions = []
            for j in indexes:
                f = artifact['frames'][j]
                a = artifact['actions'][j]
                if args.actions == 'realized':
                    af = {**artifact['frames'][j+1], 'gripper':float(a.get('gripper',a['closed']))}
                else:
                    af = {'ee':[a[k] for k in ('x','y','z')], 'euler_xyz':a.get('euler_xyz',[np.pi,0,0]), 'closed':a['closed'], 'gripper':float(a.get('gripper',a['closed']))}
                actions.append(pose(af, base=f['robot_base'], projected=False))
            converted.append({'id':f'teacher-{i}', 'frames':observations, 'actions':actions})
        results=[]
        for seed in args.seeds:
            sim = make_simulator(scene.model_copy(update={'seed':seed}))
            sim.source = camera_xml(sim.source, args.view)
            frames,actions,predictions = [sim.snapshot()],[],[]
            response = client.post('/sessions', json={'protocol':'robot-harness-icrt-v1','demonstrations':converted})
            response.raise_for_status()
            payload = response.json()
            session=payload['session_id']
            start=time.monotonic()
            for step in range(args.steps):
                observation=frames[-1]
                cameras=camera.render({'xml':sim.source,'frames':[observation]})[0]
                if step == 0:
                    for key,value in cameras.items():
                        (output/f'{seed}-{key}.jpg').write_bytes(base64.b64decode(value))
                response=client.post(f'/sessions/{session}/act',json={'observation':{**cameras,'proprio':pose(observation,base=observation['robot_base'],projected=False)}})
                response.raise_for_status()
                raw=np.array(response.json()['action'])
                xyz, rotation=flange_to_tip(raw[:3]+np.array(observation['robot_base']),raw[3:6])
                current=np.array(observation['ee'])
                xyz=np.clip(np.clip(xyz,[-.5,-.35,.025],[.5,.35,.45]),current-.04,current+.04)
                a=dict(zip(['x','y','z'],xyz.tolist()))
                a.update(euler_xyz=rotation if not args.lock_rotation else [np.pi,0,0],
                    closed=bool(raw[6]>.5),gripper=float(raw[6]>.5) if args.binary_gripper else float(np.clip(raw[6],0,1)))
                actions.append(a)
                predictions.append({'raw':raw.tolist(),'executed':a})
                frames.append(sim.step(a))
            client.delete(f'/sessions/{session}').raise_for_status()
            session=None
            artifact={'scene':sim.scene.model_dump(), 'task':task, 'xml':sim.source,'frames':frames,'actions':actions,'predictions':predictions,'model':payload['metadata'],'candidate':config,'frequency':sim.frequency}
            (output/f'rollout-{seed}.json').write_text(json.dumps(artifact))
            result={'seed':seed,**metrics(artifact),'seconds':time.monotonic()-start,'artifact_sha256':digest(artifact)}
            results.append(result)
            report={'candidate':config,'teacher':demo_results,'results':results,'successes':sum(r['verdict']['status']=='pass' for r in results),'episodes':len(results)}
            (output/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'output':output.name,**{k:v for k,v in result.items() if k!='verdict'},'status':result['verdict']['status']}),flush=True)
        return report
    finally:
        if session:
            client.delete(f'/sessions/{session}')
        client.close()
        camera.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=['stationary','factory'],default='stationary')
    p.add_argument('--cube-x',type=float,default=-.22)
    p.add_argument('--cube-y',type=float,default=-.08)
    p.add_argument('--view',choices=['original','close','front'],default='original')
    p.add_argument('--actions',choices=['commanded','realized'],default='commanded')
    p.add_argument('--demos',type=int,choices=range(5),default=1)
    p.add_argument('--demo-artifacts',type=Path,nargs='*',default=[])
    p.add_argument('--demo-variation',type=float,default=.02)
    p.add_argument('--pace',type=float,default=1)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--seeds',type=int,nargs='+',default=[2000])
    p.add_argument('--endpoint',default='http://127.0.0.1:8790')
    p.add_argument('--lock-rotation',action='store_true')
    p.add_argument('--binary-gripper',action='store_true')
    args=p.parse_args()
    if not 1<=args.steps<=254 or not .5<=args.pace<=2 or not 0<=args.demo_variation<=.15:
        p.error('Use 1–254 steps, pace .5–2, and demo variation 0–.15')
    if args.demo_artifacts and len(args.demo_artifacts)!=args.demos:
        p.error('Supply exactly one artifact per demonstration')
    experiment(args)
