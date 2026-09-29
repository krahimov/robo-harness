"""Compare model predictions to a demonstrated trajectory without claiming rollout success."""
import argparse,base64,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import io
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
import httpx
from studio.adapters.icrt import pose,demonstration_indices
from studio.adapters.icrt_render import Cameras

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('artifact',type=Path)
 p.add_argument('--endpoint',default='http://127.0.0.1:8790')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(); art=json.loads(a.artifact.read_text()); cam=Cameras()
 frames=art['frames'][:-1]; images=cam.render({'xml':art['xml'],'frames':frames})
 obs=[{**im,'proprio':pose(f,base=f['robot_base'],projected=False)} for f,im in zip(frames,images)]
 acts=[pose({'ee':[cmd[k] for k in ('x','y','z')],'euler_xyz':cmd.get('euler_xyz',[np.pi,0,0]),'closed':cmd['closed'],'gripper':float(cmd.get('gripper',cmd['closed']))},base=f['robot_base'],projected=False) for f,cmd in zip(frames,art['actions'])]
 with httpx.Client(base_url=a.endpoint,timeout=180) as c:
  sid=None
  try:
   payload=c.post('/sessions',json={'protocol':'robot-harness-icrt-v1','demonstrations':[{'id':'probe','frames':obs,'actions':acts}]}).raise_for_status().json();sid=payload['session_id']
   results=[]
   for i,o in enumerate(obs):
    raw=c.post(f'/sessions/{sid}/act',json={'observation':o}).raise_for_status().json()['action']
    results.append({'i':i,'time':frames[i]['t'],'observed':o['proprio'],'expected':acts[i],'predicted':raw,'translation_error':float(np.linalg.norm(np.array(raw[:3])-acts[i][:3])),'gripper_match':bool((raw[-1]>.5)==(acts[i][-1]>.5))})
   a.output.write_text(json.dumps({'kind':'teacher-observation probe; not a rollout; previous actions are model predictions','metadata':payload['metadata'],'results':results},indent=2))
   print('Mean translation error',np.mean([r['translation_error'] for r in results]),'gripper agreement',np.mean([r['gripper_match'] for r in results]))
  finally:
   if sid:c.delete(f'/sessions/{sid}')
   cam.close()
