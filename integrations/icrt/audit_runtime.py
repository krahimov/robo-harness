"""Read-only weight coverage and chunked-prompt equivalence checks on real weights."""
import base64,contextlib,io,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from runtime import ICRTRuntime,ROOT


def run():
 r=ICRTRuntime(device='cpu')
 state=torch.load(r.folder/'policy.pth',map_location='cpu',weights_only=True)['model']
 actual=r.model.state_dict()
 policy={k:torch.equal(actual[k].cpu(),v) for k,v in state.items()}
 vision=torch.load(r.folder/'cross-mae-rtx-vitb.pth',map_location='cpu',weights_only=True)
 vision=vision.get('model',vision.get('state_dict',vision))
 vt={k:torch.equal(v.cpu(),vision[k]) for k,v in r.model.vision_encoder.model.state_dict().items() if k in vision}
 missing=[k for k in r.model.vision_encoder.model.state_dict() if k not in vision]
 p=ROOT/'.data/runpod/iteration-local/smoke'
 art=json.loads((p/'teacher-0.json').read_text())
 # Identical known camera frames suffice to audit chunk/cache equivalence.
 side=Image.open(ROOT/'.data/runpod/iteration-local/original-side.jpg')
 wrist=Image.open(ROOT/'.data/runpod/iteration-local/original-wrist.jpg')
 from scipy.spatial.transform import Rotation
 frames=[];actions=[]
 for f,a in zip(art['frames'][:24],art['actions'][:24]):
  base=np.array(f['robot_base'])
  frames.append({'side':side,'wrist':wrist,'proprio':[*list(np.array(f['ee'])-base),*f['euler_xyz'],f['gripper']]})
  actions.append([*list(np.array([a[k] for k in ['x','y','z']])-base),*a.get('euler_xyz',f['euler_xyz']),float(a['closed'])])
 with torch.inference_mode():
  r.reset([{'frames':frames,'actions':actions}]);chunk=r.act(frames[0])
  r.reset([]);r.context_frames=len(frames);r.model.prompt(r._prepare(frames,actions));whole=r.act(frames[0])
 report={'policy_tensors':len(policy),'policy_tensors_exact':sum(policy.values()),'vision_tensors_checked':len(vt),'vision_tensors_exact':sum(vt.values()),'unmatched_vision_tensors':missing,'chunked_vs_whole_max_action_difference':float(np.max(np.abs(np.array(chunk)-whole))),'chunked_action':chunk,'whole_action':whole,'transform':str(r.wrapper.preprocess_PIL),'metadata':r.metadata()}
 out=ROOT/'.data/validation/icrt-diagnosis/runtime-audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':run()
