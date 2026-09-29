"""Offline simulator/reference probes. These do not execute or score robot tasks."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import ICRTRuntime,ROOT
from reference_eval import metrics,recorded_previous_action

def load(path):
 with np.load(path,allow_pickle=False) as x:
  p=x['proprio'].copy();a=x['actions'].copy()
  return {'frames':[{'side':Image.fromarray(s),'wrist':Image.fromarray(w),'proprio':v} for s,w,v in zip(x['side'],x['wrist'],p)],'proprio':p,'actions':a}

def run(folder):
 output=folder/'offline';output.mkdir(exist_ok=False)
 context=load(folder/'context.npz');query=load(folder/'query.npz')
 conditions=[('same_episode','own',context),('same_episode','recorded_past',context),('separate_episode','own',query),('separate_episode','recorded_past',query)]
 plan={'conditions':[(name,history) for name,history,_ in conditions],'kind':'offline; complete sampled demonstration; same-episode diagnostic explicitly includes query in context, separate episode uses another saved program; not unseen-task evaluation','inputs':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['context.npz','query.npz']}}
 (output/'plan.json').write_text(json.dumps(plan,indent=2));rt=ICRTRuntime(device='cpu');rows=[]
 for name,history,q in conditions:
  rt.reset([context]);pred=[];start=time.monotonic()
  for i,f in enumerate(q['frames']):
   if i and history=='recorded_past':rt.model.last_action=recorded_previous_action(rt,q['frames'][i-1],q['actions'][i-1])
   pred.append(rt.act(f))
  row={'condition':name,'history':history,'metrics':metrics(pred,q['actions'],q['proprio']),'hold_current':metrics(q['proprio'],q['actions'],q['proprio']),'seconds':time.monotonic()-start}
  (output/(name+'-'+history+'.json')).write_text(json.dumps({**row,'predictions':pred},indent=2));rows.append(row);(output/'report.json').write_text(json.dumps({'plan':plan,'model':rt.metadata(),'results':rows},indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)
