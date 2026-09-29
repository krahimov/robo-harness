"""Teacher-gated ICRT pick/place tests with a separate Robotiq research fixture."""
import argparse,base64,hashlib,json,sys,time,io
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'backend'))
from robotiq_fixture import RobotiqFranka,pose,action
from studio.schemas import Scene,task_for_scene
from studio.adapters.icrt_render import Cameras
from studio.adapters.simulation import interpolate_orientation
from studio.judges import state_judge
import httpx

def scene(seed):
    return Scene(embodiment='franka-panda-v1',seed=seed,variation=.008,cube_x=-.07,cube_y=-.12,tray_x=.06,tray_y=.10,table_color=[.85,.85,.82,1],cube_color=[.65,.45,.25,1],contact_solver='precision')

def teacher(spec):
    sim=RobotiqFranka(spec);frames=[sim.snapshot()];actions=[]
    x,y,z=frames[0]['cube'];points=[(x,y,.19,False,1),(x,y,z,False,1),(x,y,z,True,1),(x,y,.19,True,1),(spec.tray_x,spec.tray_y,.19,True,1.5),(spec.tray_x,spec.tray_y,.07,True,1),(spec.tray_x,spec.tray_y,.07,False,.7),(spec.tray_x,spec.tray_y,.24,False,1)]
    for x,y,z,closed,duration in points:
        start=frames[-1];count=round(duration*sim.frequency)
        for i in range(count):
            frac=(i+1)/count;cmd=dict(zip(['x','y','z'],np.array(start['ee'])+frac*(np.array([x,y,z])-start['ee'])))
            cmd.update(closed=closed,euler_xyz=interpolate_orientation(start['euler_xyz'],[np.pi,0,-np.pi/4],frac))
            frames.append(sim.step(cmd));actions.append(cmd)
    for _ in range(15):frames.append(sim.step(actions[-1]));actions.append(actions[-1].copy())
    return {'scene':spec.model_dump(),'xml':sim.source,'task':task_for_scene(spec),'frames':frames,'actions':actions,'frequency':sim.frequency,'fixture':'franka-robotiq-2f85-research-v1','teacher':True}

def rotate(images,angle):
    if not angle:return images
    for image in images:
        im=Image.open(io.BytesIO(base64.b64decode(image['wrist']))).rotate(angle,expand=True);buf=io.BytesIO();im.save(buf,format='PNG');image['wrist']=base64.b64encode(buf.getvalue()).decode()
    return images

def metrics(a):
    return {'verdict':state_judge(a,{'require_review':True}),'grasp_frames':sum(f['attached'] for f in a['frames']),'closest_m':min(float(np.linalg.norm(np.array(f['ee'])-f['cube'])) for f in a['frames']),'max_cube_z':max(f['cube'][2] for f in a['frames'])}

def run(out,teacher_only=False,angle=0,demo_count=1,seeds=(72000,),binary=False,feedback=False,endpoint="http://127.0.0.1:8790"):
    out.mkdir(parents=True,exist_ok=False)
    plan={'fixture':'Franka with Menagerie Robotiq 2F-85, approximate mounting/cameras; not calibrated DROID','actor':'icrt','teacher_seeds':[71,72][:demo_count or 1],'query_seeds':list(seeds),'wrist_rotation':angle,'binary_gripper':binary,'executed_action_feedback':feedback,'steps':220,'fixed_weights':True,'demo_count':demo_count,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));camera=Cameras();converted=[]
    try:
        for i,seed in enumerate(plan['teacher_seeds']):
            a=teacher(scene(seed));(out/f'teacher-{i}.json').write_text(json.dumps(a));measured=metrics(a);(out/f'teacher-{i}-metrics.json').write_text(json.dumps(measured,indent=2));print('teacher',i,measured,flush=True)
            ims=rotate(camera.render({'xml':a['xml'],'frames':[a['frames'][0],a['frames'][30],a['frames'][60]]}),angle)
            for j,im in enumerate(ims):
                for key in ['side','wrist']:(out/f'teacher-{i}-{j}-{key}.jpg').write_bytes(base64.b64decode(im[key]))
            if measured['verdict']['status']!='pass':raise ValueError('Teacher failed: no model trial will run')
            if teacher_only:continue
            # Full, uncompressed single demo; two demos are uniformly sampled to 128 each.
            ix=np.linspace(0,len(a['actions'])-1,min(len(a['actions']),256//max(1,demo_count))).round().astype(int)
            fs=[a['frames'][j] for j in ix];ims=rotate(camera.render({'xml':a['xml'],'frames':fs}),angle)
            acts=[]
            for j in ix:
                cmd=a['actions'][j];acts.append(pose({'ee':[cmd[k] for k in ['x','y','z']],'euler_xyz':cmd['euler_xyz'],'closed':cmd['closed'],'gripper':float(cmd['closed'])}))
            converted.append({'id':f'robotiq-teacher-{i}','frames':[{**im,'proprio':pose(f)} for f,im in zip(fs,ims)],'actions':acts})
        if teacher_only:return
        with httpx.Client(base_url=endpoint,timeout=180) as client:
            for seed in seeds:
                sim=RobotiqFranka(scene(seed));fs=[sim.snapshot()];actions=[];pred=[];sid=None
                try:
                    response=client.post('/sessions',json={'protocol':'robot-harness-icrt-v1','demonstrations':converted[:demo_count]}).raise_for_status().json();sid=response['session_id']
                    for i in range(plan['steps']):
                        im=rotate(camera.render({'xml':sim.source,'frames':[fs[-1]]}),angle)[0]
                        request={'observation':{**im,'proprio':pose(fs[-1])}}
                        if feedback and actions:
                            last=actions[-1];request['previous_action']=pose({'ee':[last[k] for k in ['x','y','z']],'euler_xyz':last['euler_xyz'],'closed':last['closed'],'gripper':last['gripper']})
                        raw=client.post(f'/sessions/{sid}/act',json=request).raise_for_status().json()['action'];cmd,clipped=action(raw,fs[-1])
                        if binary:cmd['gripper']=float(raw[6]>.5)
                        fs.append(sim.step(cmd));actions.append(cmd);pred.append({'raw':raw,'executed':cmd,'clipped':clipped})
                        if i%75==0:print(seed,'step',i,flush=True)
                finally:
                    if sid:client.delete(f'/sessions/{sid}')
                a={'scene':sim.scene.model_dump(),'task':task_for_scene(sim.scene),'xml':sim.source,'frames':fs,'actions':actions,'frequency':sim.frequency,'predictions':pred,'model':response['metadata'],'fixture':plan['fixture'],'plan':plan}
                (out/f'rollout-{seed}.json').write_text(json.dumps(a));m=metrics(a);m['clipped_predictions']=sum(p['clipped'] for p in pred);(out/f'report-{seed}.json').write_text(json.dumps(m,indent=2));print('result',seed,m,flush=True)
    finally:camera.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-only',action='store_true');p.add_argument('--angle',type=int,choices=[0,90,180,270],default=0);p.add_argument('--demos',type=int,choices=[0,1,2],default=1);p.add_argument('--seeds',type=int,nargs='+',default=[72000]);p.add_argument('--binary',action='store_true');p.add_argument('--executed-feedback',action='store_true');p.add_argument('--endpoint',default='http://127.0.0.1:8790');a=p.parse_args();run(a.output,a.teacher_only,a.angle,a.demos,a.seeds,a.binary,a.executed_feedback,a.endpoint)
