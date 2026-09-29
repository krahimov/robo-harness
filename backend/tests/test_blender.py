import json
import threading
import zipfile
from pathlib import Path

import mujoco
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from studio import blender_adapter as adapter, storage, codex_author, authoring_api, assets
from studio.authoring import AuthorCancelled, AuthorError
from studio.blender_schema import BlenderProgram, BLENDER_RULES
from studio.schemas import Scene
from studio.task_api import get_task
from studio.main import app

TETRA='v -1 -1 -1\nv 1 -1 -1\nv -1 1 -1\nv -1 -1 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'


def design():
    return BlenderProgram.model_validate({'name':'Machine','description':'A machine next to the robot',
        'assemblies':[{'name':'housing','parts':[{'position':[-.4,.3,.1],'size':[.05,.05,.1]}]}]})


@pytest.fixture
def fake_blender(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'ROOT',tmp_path)
    monkeypatch.setattr(adapter,'status',lambda:{'configured':True,'version':'test'})
    def worker(mode,folder,event):
        if mode=='build':
            (folder/'housing.obj').write_text(TETRA)
            (folder/'scene.blend').write_bytes(b'test-blend')
            (folder/'manifest.json').write_text(json.dumps({'format':'studio-blender-v1','blender_version':'test',
                'collision_method':'separate part AABBs','meshes':[{'name':'housing','color':[.2,.4,.6,1],
                'position':[-.4,.3,.1],'size':[.05,.05,.1],
                'collisions':[{'position':[-.4,.3,.1],'size':[.05,.05,.1]}]}]}))
        else:
            assert not assets.list_assets()
            (folder/'preview.png').write_bytes(b'layout-image')
    monkeypatch.setattr(adapter,'run_worker',worker)
    return worker


def test_blender_import_separates_visual_contact_geometry_and_exports_evidence(fake_blender):
    spec,artifact,checks=adapter.compile_blender(design(),get_task('block-to-tray'))
    assert checks['status']=='pass'
    m=mujoco.MjModel.from_xml_string(artifact['xml'])
    assert m.geom('prop_bv_housing').contype==0
    assert m.geom('prop_bc_housing_0').contype==1
    assert m.geom('prop_bc_housing_0').rgba[3]==0
    assert artifact['task']['schema_version']==2
    meta=artifact['blender']
    assert meta['visual_meshes']==meta['collision_shapes']==1
    assert len(assets.list_assets())==1
    folder=adapter.build_folder(meta['build_id'])
    with zipfile.ZipFile(folder/'bundle.zip') as z:
        assert {'scene.blend','housing.obj','scene.xml','simulation.json','design.json'}<=set(z.namelist())
        assert not any(name.endswith('.log') for name in z.namelist())
    assert adapter.download(meta['build_id'],'preview.png').path==folder/'preview.png'
    fixture=json.loads((folder/'fixture.json').read_text())
    assert not any(g['name'].startswith('prop_bv_') or g['name'].startswith('prop_bc_') for g in fixture)


def test_failed_render_removes_partial_assets_and_build(fake_blender,monkeypatch):
    def fail(mode,folder,event):
        if mode=='preview':raise ValueError('Render failed')
        fake_blender(mode,folder,event)
    monkeypatch.setattr(adapter,'run_worker',fail)
    with pytest.raises(ValueError):adapter.compile_blender(design())
    assert storage.all_records('asset')==[]
    assert list((storage.ROOT/'blender').iterdir())==[]


def test_cancel_cleans_partial_blender_assets(fake_blender,monkeypatch):
    event=threading.Event()
    def cancel(mode,folder,stop):
        fake_blender(mode,folder,stop)
        event.set()
    monkeypatch.setattr(adapter,'run_worker',cancel)
    with pytest.raises(AuthorCancelled):adapter.compile_blender(design(),event=event)
    assert storage.all_records('asset')==[]


@pytest.mark.parametrize('mutation',[lambda d:d.update(python='import os'),
    lambda d:d['assemblies'][0].update(name='../escape'),
    lambda d:d['assemblies'][0]['parts'][0].update(size=[float('nan'),.1,.1]),
    lambda d:d.update(assemblies=d['assemblies']*2)])
def test_blender_program_rejects_code_paths_nonfinite_and_duplicate_names(mutation):
    data=design().model_dump();mutation(data)
    with pytest.raises(ValidationError):BlenderProgram.model_validate(data)


def test_mujoco_mesh_and_material_survive_blender_fixture_export(fake_blender):
    spec,artifact,_=adapter.compile_blender(design())
    fixture=adapter.fixture_geometries(artifact,set())
    mesh=next(g for g in fixture if g['name']=='prop_bv_housing')
    assert mesh['type']=='mesh' and len(mesh['vertices'])==4 and len(mesh['faces'])==4
    assert any(g['type']=='cylinder' for g in fixture)


def test_codex_blender_loop_supplies_layout_and_rollout_evidence(fake_blender,monkeypatch):
    monkeypatch.setattr(codex_author,'rendered_frames',lambda *args:[])
    calls=[]
    class Session:
        def __init__(self,folder,instructions):assert 'assemblies' in instructions and 'Blender' in instructions
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def run(self,prompt,photos,event):
            calls.append((json.loads(prompt),photos))
            reply={'action':'build','program_json':design().model_dump_json(),'summary':'Build'} if len(calls)==1 else {'action':'finish','program_json':'','summary':'Reviewed layout and numerical checks.'}
            return json.dumps(reply),{}
    result=codex_author.author_scene_codex({'name':'Scene','prompt':'Build machinery','image_ids':[]},None,lambda _:None,threading.Event(),
        session_factory=Session,program_model=BlenderProgram,compiler=adapter.compile_blender,extra_instructions=BLENDER_RULES)
    assert result['artifact']['blender']['visual_meshes']==1
    assert len(calls[1][1])==1 and 'static layout' in calls[1][1][0][0]
    assert calls[1][0]['history'][0]['simulation']['blender']['collision_shapes']==1
    assert result['camera_available'] is False  # Layout rendering cannot substitute for motion camera evidence.


def test_blender_draft_preflight_and_file_allowlist(fake_blender,monkeypatch):
    monkeypatch.setattr(authoring_api,'provider_status',lambda _: {'configured':True,'name':'test'})
    with TestClient(app) as client:
        draft=client.post('/api/scenes',json={'name':'Photo scene','prompt':'Build this factory','provider':'codex','builder':'blender'}).json()
        assert draft['builder']=='blender'
        monkeypatch.setattr(adapter,'status',lambda:{'configured':False,'message':'Install Blender'})
        response=client.post('/api/scenes/'+draft['id']+'/generate')
        assert response.status_code==409 and response.json()['detail']=='Install Blender'
        assert client.get('/api/blender/builds/'+'a'*32+'/build.log').status_code==404
        assert client.get('/api/blender/builds/invalid/preview.png').status_code==404


def test_worker_environment_excludes_provider_secrets(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-private')
    monkeypatch.setenv('STUDIO_ICRT_TOKEN','test-private')
    assert 'OPENAI_API_KEY' not in adapter.worker_environment()
    assert 'STUDIO_ICRT_TOKEN' not in adapter.worker_environment()
