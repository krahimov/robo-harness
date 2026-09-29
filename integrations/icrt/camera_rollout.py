"""Controlled wrist-image intervention on the unchanged simulated pen task."""
import argparse,base64,io,json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'backend'))
from studio import storage as db
from studio.schemas import Scene
from studio.adapters.simulation import make_simulator
from studio.adapters import icrt
from studio.judges import state_judge

def run(output,angle,seed):
    output.mkdir(parents=True,exist_ok=False)
    source=db.get('experiment','de8c6487b6f44edeb759ecd80aea468c')
    demo=db.get('demo','f72fc7aee300431f9823dbbf3bc11f5e');demo={**demo,'artifact':db.read_artifact(demo['artifact_id'])}
    scene=Scene(**{**source['scene'],'seed':seed});task=source['task'];sim=make_simulator(scene)
    plan={'source':source['id'],'demo':demo['id'],'scene':scene.model_dump(),'task':task,'seed':seed,'steps':415,'wrist_rotation_degrees':angle,'kind':'Actual ICRT control; camera rotation only; fixed weights, no oracle or replay','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    original=icrt.CameraClient.render
    def render(self,source,frames):
        images=original(self,source,frames)
        for image in images:
            photo=Image.open(io.BytesIO(base64.b64decode(image['wrist']))).rotate(angle,expand=True)
            buffer=io.BytesIO();photo.save(buffer,format='PNG');image['wrist']=base64.b64encode(buffer.getvalue()).decode()
        return images
    icrt.CameraClient.render=render
    frames=[sim.snapshot()];actions=[];actor=None;error=None
    try:
        actor=icrt.ICRTModel([demo],frames[0],sim,horizon_steps=415)
        for i in range(415):
            a=actor.act(frames[-1]);actions.append(a);frames.append(sim.step(a))
            if i%100==0:print('step',i,flush=True)
    except Exception as exc:error=repr(exc)
    finally:
        if actor:actor.close()
        icrt.CameraClient.render=original
    artifact={'scene':scene.model_dump(),'task':task,'xml':sim.source,'frames':frames,'actions':actions,'frequency':sim.frequency,'model':actor.metadata if actor else {},'predictions':actor.predictions if actor else [],'intervention':plan,'execution_error':error}
    (output/'rollout.json').write_text(json.dumps(artifact))
    v=state_judge(artifact,{'require_review':True})
    summary={'verdict':v,'steps':len(actions),'error':error,'grasp_frames':sum(f['objects']['pen']['attached'] for f in frames),'closest_fingertip_distance_m':min(float(np.linalg.norm(np.array(f['ee'])-f['objects']['pen']['position'])) for f in frames),'clipped_predictions':artifact['model'].get('clipped_predictions')}
    (output/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--angle',type=int,choices=[90,180,270],required=True);p.add_argument('--seed',type=int,default=62000);a=p.parse_args();run(a.output,a.angle,a.seed)
