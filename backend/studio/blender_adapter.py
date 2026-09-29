"""Blender authoring assets and MuJoCo execution with immutable build evidence."""
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from studio import storage as db
from studio.assets import parse_obj, public_asset
from studio.authoring import AuthorError, SceneProgram, check_cancel, compile_program
from studio.blender_schema import BlenderProgram
from studio.schemas import Scene, SceneProp

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT/'integrations/blender/worker.py'
router = APIRouter(prefix='/api/blender')
_probe = None


def python_binary():
    configured=os.environ.get('STUDIO_BLENDER_PYTHON')
    return str(Path(configured).expanduser()) if configured else str(ROOT/'integrations/blender/.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))


def worker_environment():
    return {key:value for key,value in os.environ.items() if key in {'PATH','TMPDIR','TEMP','TMP','SYSTEMROOT','WINDIR','LANG','LC_ALL'}}


def status():
    global _probe
    binary=python_binary()
    if _probe and _probe[0]>time.monotonic() and _probe[1]==binary:
        return dict(_probe[2])
    ready,version=False,None
    if Path(binary).is_file():
        try:
            probe=subprocess.run([binary,'-c','import bpy; print("STUDIO_BLENDER="+bpy.app.version_string)'],
                capture_output=True,text=True,timeout=12,env=worker_environment())
            version=next((line.split('=',1)[1] for line in probe.stdout.splitlines() if line.startswith('STUDIO_BLENDER=')),None)
            ready=probe.returncode==0 and version is not None
        except (OSError,subprocess.TimeoutExpired):
            pass
    result={'configured':ready,'version':version,'message':'Blender is ready for scene generation.' if ready else
        'Install the Blender worker with uv sync --project integrations/blender from the repository root.'}
    _probe=(time.monotonic()+30,binary,result)
    return result


def build_folder(id):
    if len(id)!=32 or any(c not in '0123456789abcdef' for c in id):
        raise ValueError('Invalid Blender build ID')
    return db.ROOT/'blender'/id


@router.get('/status')
def blender_status():
    return status()


@router.get('/builds/{id}/{filename}')
def download(id: str, filename: str):
    if filename not in {'preview.png','scene.blend','bundle.zip','manifest.json','design.json'}:
        raise HTTPException(404,'Blender file not found')
    try:
        folder=build_folder(id)
    except ValueError:
        raise HTTPException(404,'Blender file not found') from None
    path=folder/filename
    # Publication is atomic at the end of a successful compile.
    if not (folder/'complete').exists() or not path.is_file():
        raise HTTPException(404,'Blender file not found')
    return FileResponse(path,media_type='image/png' if filename=='preview.png' else 'application/octet-stream',
        filename=None if filename=='preview.png' else filename,
        headers={'X-Content-Type-Options':'nosniff'})


def run_worker(mode, folder, event):
    check_cancel(event)
    try:
        with (folder/(mode+'.log')).open('wb') as output:
            process=subprocess.Popen([python_binary(),str(WORKER),mode,str(folder)],cwd=folder,
                env=worker_environment(),stdout=output,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+180
            try:
                while process.poll() is None:
                    check_cancel(event)
                    if time.monotonic()>deadline:
                        raise AuthorError('Blender timed out. Simplify the scene and retry.')
                    if output.tell()>4*1024*1024:
                        raise AuthorError('Blender exceeded the output limit. Simplify the scene and retry.')
                    event.wait(.1)
                if process.returncode:
                    raise ValueError('Blender could not build or render this geometry. Simplify its parts.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=3)
    except OSError:
        raise AuthorError('Could not start Blender. Check the local Blender worker installation.') from None


def fixture_geometries(artifact, generated_names):
    import mujoco
    import numpy as np
    model=mujoco.MjModel.from_xml_string(artifact['xml'])
    data=mujoco.MjData(model)
    data.qpos[:]=artifact['frames'][0]['qpos']
    mujoco.mj_forward(model,data)
    result=[]
    for i in range(model.ngeom):
        name=model.geom(i).name
        if name in generated_names or model.geom_rgba[i,3]==0:
            continue
        if artifact['scene']['embodiment']=='franka-panda-v1' and model.geom_bodyid[i]!=0 and model.geom_group[i]==3:
            continue
        kind={int(mujoco.mjtGeom.mjGEOM_MESH):'mesh',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder',int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere'}.get(int(model.geom_type[i]),'box')
        geom={'name':name,'type':kind,'position':data.geom_xpos[i].tolist(),'matrix':data.geom_xmat[i].tolist(),
            'size':model.geom_size[i].tolist(),'color':model.geom_rgba[i].tolist()}
        if model.geom_matid[i]>=0:
            geom['color']=model.mat_rgba[model.geom_matid[i]].tolist()
        if kind=='mesh':
            mid=model.geom_dataid[i]
            start,count=model.mesh_vertadr[mid],model.mesh_vertnum[mid]
            fstart,fcount=model.mesh_faceadr[mid],model.mesh_facenum[mid]
            geom.update(vertices=model.mesh_vert[start:start+count].tolist(),faces=model.mesh_face[fstart:fstart+fcount].tolist())
        result.append(geom)
    return result


def compile_blender(program: BlenderProgram, task=None, embodiment=None, *, event=None, progress=None):
    event=event or threading.Event()
    progress=progress or (lambda message:None)
    if not status()['configured']:
        raise AuthorError(status()['message'])
    if embodiment:
        program=program.model_copy(update={'scene':Scene.model_validate({**program.scene.model_dump(),'embodiment':embodiment})})
    id=uuid.uuid4().hex
    folder=build_folder(id)
    folder.mkdir(parents=True)
    created=[]
    try:
        design=program.model_dump(exclude_none=True)
        (folder/'design.json').write_text(json.dumps(design,indent=2))
        progress('Building machinery meshes in Blender.')
        run_worker('build',folder,event)
        check_cancel(event)
        manifest=json.loads((folder/'manifest.json').read_text())
        if [m['name'] for m in manifest['meshes']] != [a.name for a in program.assemblies]:
            raise ValueError('Blender export does not match the requested assemblies')
        props=list(program.scene.props)
        generated_names=set()
        for mesh in manifest['meshes']:
            geometry=parse_obj((folder/(mesh['name']+'.obj')).read_text(),max_vertices=1200,max_faces=2400,max_bytes=512*1024)
            asset=db.save('asset',{'name':mesh['name'],'kind':'mesh','source':'blender','status':'staging','build_id':id,**geometry})
            created.append(asset['id'])
            prop=SceneProp(name='bv_'+mesh['name'],shape='mesh',asset_id=asset['id'],position=mesh['position'],size=mesh['size'],color=mesh['color'],collidable=False)
            props.append(prop); generated_names.add('prop_'+prop.name)
            for index,collision in enumerate(mesh['collisions']):
                props.append(SceneProp(name=f"bc_{mesh['name']}_{index}",position=collision['position'],
                    size=[max(.005,s) for s in collision['size']],color=[.3,.5,.8,0],collidable=True))
        scene=Scene.model_validate({**program.scene.model_dump(),'props':[p.model_dump() for p in props]})
        plain=SceneProgram.model_validate({k:v for k,v in design.items() if k!='assemblies'}).model_copy(update={'scene':scene})
        progress('Importing separate visual and collision geometry into MuJoCo; running the selected robot.')
        spec,artifact,verification=compile_program(plain,task,embodiment)
        check_cancel(event)
        (folder/'fixture.json').write_text(json.dumps(fixture_geometries(artifact,generated_names)))
        if artifact['scene']['embodiment']=='franka-panda-v1':
            from studio.adapters.franka import asset_folder, ASSET_COMMIT
            license_path=asset_folder()/'LICENSE'
            if not license_path.is_file():
                raise ValueError('The Panda asset license is missing. Reinstall the pinned robot assets.')
            (folder/'ROBOT_ASSETS_LICENSE.txt').write_text('MuJoCo Menagerie Franka Panda assets\nSource: https://github.com/google-deepmind/mujoco_menagerie\nCommit: '+ASSET_COMMIT+'\n\n'+license_path.read_text())
        progress('Rendering the Blender layout with the actual robot and starting state.')
        run_worker('preview',folder,event)
        if not (folder/'preview.png').is_file():
            raise ValueError('Blender returned no layout image')
        metadata={'build_id':id,'version':manifest['blender_version'],'program':design,'asset_ids':created,
            'visual_meshes':len(created),'collision_shapes':sum(len(m['collisions']) for m in manifest['meshes']),
            'preview_url':f'/api/blender/builds/{id}/preview.png','blend_url':f'/api/blender/builds/{id}/scene.blend',
            'bundle_url':f'/api/blender/builds/{id}/bundle.zip','collision_method':manifest['collision_method']}
        artifact['blender']=metadata
        (folder/'scene.xml').write_text(artifact['xml'])
        (folder/'simulation.json').write_text(json.dumps({'spec':spec,'task':artifact['task'],'verification':verification,
            'frequency':artifact['frequency'],'actions':artifact['actions'],'qpos':[f['qpos'] for f in artifact['frames']],
            'blender':metadata},indent=2))
        with zipfile.ZipFile(folder/'bundle.zip','w',zipfile.ZIP_DEFLATED) as bundle:
            for filename in ['scene.blend','preview.png','manifest.json','design.json','simulation.json','scene.xml',*[m['name']+'.obj' for m in manifest['meshes']],*(['ROBOT_ASSETS_LICENSE.txt'] if (folder/'ROBOT_ASSETS_LICENSE.txt').exists() else [])]:
                bundle.write(folder/filename,filename)
        (folder/'complete').write_text('ready')
        for asset_id in created:
            db.update('asset',asset_id,status='ready')
        return spec,artifact,verification
    except BaseException:
        # Candidate assets are hidden from the library until compile completes. Remove
        # records from failed candidates so the library does not fill with debris.
        for asset_id in created:
            db.delete('asset',asset_id)
        shutil.rmtree(folder,ignore_errors=True)
        raise
