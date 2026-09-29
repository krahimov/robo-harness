"""Offline real-data action probe; never scored as closed-loop task success."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import ICRTRuntime,ROOT

if __name__=='__main__':
    folder=ROOT/'.data/validation/icrt-diagnosis'
    data=np.load(folder/'real-sample.npz')
    frames=[{'side':Image.fromarray(s),'wrist':Image.fromarray(w),'proprio':p} for s,w,p in zip(data['side'],data['wrist'],data['proprio'])]
    frames[0]['side'].save(folder/'real-side.jpg');frames[0]['wrist'].save(folder/'real-wrist.jpg')
    runtime=ICRTRuntime(device='cpu')
    results=[]
    for context in [False,True]:
        runtime.reset([{'frames':frames,'actions':data['actions']}] if context else [])
        predictions=np.array([runtime.act(f) for f in frames])
        truth=data['actions']
        r={'condition':'same-episode context' if context else 'no context','frames':len(frames),
            'translation_mae_m':float(np.linalg.norm(predictions[:,:3]-truth[:,:3],axis=1).mean()),
            'gripper_agreement':float(((predictions[:,-1]>.5)==(truth[:,-1]>.5)).mean()),
            'gripper_close_fraction':float((truth[:,-1]>.5).mean()),
            'hold_pose_translation_mae_m':float(np.linalg.norm(data['proprio'][:,:3]-truth[:,:3],axis=1).mean()),
            'proprio_min':data['proprio'].min(axis=0).tolist(),'proprio_max':data['proprio'].max(axis=0).tolist(),
            'predictions':predictions.tolist()}
        results.append(r);print({k:v for k,v in r.items() if k not in ['predictions','proprio_min','proprio_max']},flush=True)
    (folder/'real-probe.json').write_text(json.dumps({'note':'Offline observations from one released real demonstration; own predicted action history, same-episode context; not a rollout, held-out evaluation, or calibrated benchmark.','results':results},indent=2))
