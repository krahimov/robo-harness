"""Rigid-coordinate interventions: fixed weights/images, transformed poses/actions."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from runtime import ICRTRuntime,ROOT
from reference_eval import load_episode,metrics
from transfer_probe import load

def transform(values,rotation,translation):
 result=np.array(values,dtype=float,copy=True)
 result[...,:3]=rotation.apply(result[...,:3])+translation
 result[...,3:6]=(rotation*Rotation.from_euler('XYZ',result[...,3:6])).as_euler('XYZ')
 return result

def episode_transform(episode,r,t):
 p=transform(episode['proprio'],r,t)
 return {**episode,'proprio':p,'actions':transform(episode['actions'],r,t),'frames':[{**f,'proprio':v} for f,v in zip(episode['frames'],p)]}

def run(folder):
 out=folder/'coordinates';out.mkdir(exist_ok=False)
 ref=ROOT/'.data/validation/icrt-reference-1';m=json.loads((ref/'manifest.json').read_text())
 rc=load_episode(ref,m['episodes'][0]);rq=load_episode(ref,m['episodes'][1]);sc=load(folder/'context.npz');sq=load(folder/'query.npz')
 # One fixed transform from context initial poses; no query actions define it.
 rot=Rotation.from_euler('XYZ',rc['proprio'][0,3:6])*Rotation.from_euler('XYZ',sc['proprio'][0,3:6]).inv()
 shift=rc['proprio'][0,:3]-rot.apply(sc['proprio'][0,:3])
 identity=Rotation.identity();zero=np.zeros(3)
 cond=[('sim_original',sc,sq,identity,zero),('sim_shift',sc,sq,identity,rc['proprio'][0,:3]-sc['proprio'][0,:3]),('sim_rigid',sc,sq,rot,shift),('real_original',rc,rq,identity,zero),('real_to_sim',rc,rq,rot.inv(),-rot.inv().apply(shift))]
 plan={'conditions':[c[0] for c in cond]+[f'sim_wrist_{a}' for a in [90,180,270]],'rotation_matrix':rot.as_matrix().tolist(),'translation':shift.tolist(),'kind':'offline coordinate sensitivity; images fixed; no query target used to choose transform; not physically calibrated camera extrinsics or a rollout'}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));rt=ICRTRuntime(device='cpu');rows=[]
 for angle in [90,180,270]:
  def rotate_wrist(e):
   return {**e,'frames':[{**f,'wrist':f['wrist'].rotate(angle,expand=True)} for f in e['frames']]}
  cond.append((f'sim_wrist_{angle}',rotate_wrist(sc),rotate_wrist(sq),identity,zero))
 for name,c,q,r,t in cond:
  c=episode_transform(c,r,t);q=episode_transform(q,r,t);rt.reset([c]);start=time.monotonic();pred=[rt.act(f) for f in q['frames']]
  row={'condition':name,'metrics':metrics(pred,q['actions'],q['proprio']),'seconds':time.monotonic()-start};rows.append(row)
  (out/(name+'.json')).write_text(json.dumps({**row,'predictions':pred}));(out/'report.json').write_text(json.dumps({'plan':plan,'model':rt.metadata(),'results':rows},indent=2));print(json.dumps(row),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)
