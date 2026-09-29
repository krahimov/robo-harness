"""Export complete, evenly sampled simulator recordings for offline ICRT diagnosis."""
import argparse,base64,hashlib,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from studio import storage as db
from studio.adapters.icrt import pose,demonstration_indices
from studio.adapters.icrt_render import Cameras

def prepare(demo_id,episode_id,output,budget=120):
    if not 16<=budget<=250:raise ValueError('Budget must leave space for both context and query')
    output.mkdir(parents=True,exist_ok=False)
    records={'context':db.get('demo',demo_id),'query':db.get('episode',episode_id)}
    plan={'demo_id':demo_id,'episode_id':episode_id,'budget':budget,'kind':'Offline sampled observation/action prediction; no robot execution','artifacts':{n:r['artifact_id'] for n,r in records.items()}}
    (output/'input-plan.json').write_text(json.dumps(plan,indent=2));camera=Cameras()
    try:
        for name,record in records.items():
            artifact=db.read_artifact(record['artifact_id']);indexes=demonstration_indices(artifact,budget)
            frames=[artifact['frames'][i] for i in indexes]
            images=camera.render({'xml':artifact['xml'],'frames':frames})
            proprio=[pose(f,base=f['robot_base'],projected=False) for f in frames];actions=[]
            for index,frame in zip(indexes,frames):
                command=artifact['actions'][index]
                actions.append(pose({'ee':[command[k] for k in ('x','y','z')],'euler_xyz':command.get('euler_xyz',[np.pi,0,0]),'closed':command['closed'],'gripper':float(command.get('gripper',command['closed']))},base=frame['robot_base'],projected=False))
            cameras={key:np.array([np.array(Image.open(io.BytesIO(base64.b64decode(image[key])))) for image in images]) for key in ['side','wrist']}
            path=output/(name+'.npz');np.savez_compressed(path,proprio=proprio,actions=actions,indices=indexes,**cameras)
            plan[name]={'source_frequency':artifact['frequency'],'source_actions':len(artifact['actions']),'indices':indexes,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (output/'input-plan.json').write_text(json.dumps(plan,indent=2))
    finally:camera.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--demo',required=True);p.add_argument('--episode',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=120);a=p.parse_args();prepare(a.demo,a.episode,a.output,a.budget)
