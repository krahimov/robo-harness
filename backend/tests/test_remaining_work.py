import copy,json,math,threading
import numpy as np
import pytest
from studio import storage,calibration_sets
from studio.calibration import correction_evidence
from studio.stress import CASES,case_scene
from studio.workflows import pen_pouch_scene
from studio.task_api import bind_task,get_task
from studio.adapters import astra
from studio.adapters.icrt import context_budget,demonstration_indices
from studio.adapters.simulation import xml
from studio.schemas import ExperimentCreate,JudgeSpec


def test_stress_case_rotation_aligns_pen_and_supports_and_keeps_input_fixed():
    import mujoco
    base=pen_pouch_scene();task=bind_task(base,get_task('pen-into-open-pouch'));before=base.model_dump();task_before=copy.deepcopy(task)
    for case in CASES:
        scene,resolved=case_scene(base,task,case,44)
        assert scene.seed==44
        if case.get('yaw'):
            m=mujoco.MjModel.from_xml_string(xml(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            axis=d.geom('object_pen_geom').xmat.reshape(3,3)[:,2]
            props={p.name:p for p in scene.props}
            direction=np.array(props['pen_rest_far'].position)-props['pen_rest_near'].position
            direction/=np.linalg.norm(direction)
            assert abs(np.dot(axis,direction))>0.99999
        if case.get('pouch_dx'):
            old=next(r['bounds'] for r in task['requirements'] if r.get('bounds'))
            new=next(r['bounds'] for r in resolved['requirements'] if r.get('bounds'))
            assert new[0]-old[0]==pytest.approx(.04)
            assert new[1]-new[0]==pytest.approx(old[1]-old[0])
    assert base.model_dump()==before and task==task_before


def test_icrt_reserves_query_space_and_preserves_demo_ending():
    assert context_budget(415,1)==95
    assert context_budget(180,1)==256
    assert context_budget(510,0)==0
    with pytest.raises(ValueError):context_budget(510,1)
    with pytest.raises(ValueError):context_budget(511,0)
    artifact={'actions':[{}]*414,'frames':[{}]*415,'frequency':15}
    indices=demonstration_indices(artifact,context_budget(415,1))
    assert len(indices)==95 and indices[0]==0 and indices[-1]==413
    assert len(indices)+415+2<=512


def test_checkpoint_can_replace_remaining_motions_without_reset(monkeypatch):
    instances=[]
    class Sim:
        frequency=15;source='<mujoco/>'
        def __init__(self):self.t=0;self.ee=[0,0,.2];instances.append(self)
        def snapshot(self):return {'t':self.t,'ee':self.ee,'euler_xyz':[math.pi,0,0],'closed':False,'cube':[0,0,.04],'objects':{'pen':{'position':[0,0,.04],'attached':False}}}
        def step(self,a):self.t+=1/15;self.ee=[a[k] for k in ('x','y','z')];return self.snapshot()
    monkeypatch.setattr(astra,'make_simulator',lambda scene:Sim())
    scene=pen_pouch_scene();task=bind_task(scene,get_task('pen-into-open-pouch'))
    base={'target':'gripper','offset':[0,0,0],'yaw':0,'closed':False,'seconds':.2}
    initial=astra.Program(motions=[{**base,'closed':True}]+[base]*3)
    calls=[]
    def feedback(artifact,pending,reason):
        calls.append((artifact['frames'][-1]['t'],reason))
        replacement=astra.Program(motions=[{**base,'offset':[.01,0,0]}]*4)
        return replacement,{'decision':'replace','t':artifact['frames'][-1]['t']}
    a=astra.execute(scene,task,initial,threading.Event(),checkpoint=feedback)
    assert len(instances)==1 and len(calls)==1
    assert a['frames'][-1]['ee'][0]==pytest.approx(.04)
    assert a['feedback_trace'][0]['decision']=='replace'
    assert len(a['actions'])+1==len(a['frames']) and a['execution_error'] is None


def test_feedback_call_limit_and_snapshot_error_stop(monkeypatch):
    from studio.adapters.astra_feedback import Feedback
    scene=pen_pouch_scene();task=bind_task(scene,get_task('pen-into-open-pouch'))
    feedback=Feedback(scene,task,threading.Event(),limit=1);feedback.count=1
    assert feedback({},[],'after_grasp')==(None,None)
    with pytest.raises(ValueError):ExperimentCreate(name='test',demo_ids=[],judge_id='j',max_checkpoints=20)


def test_calibration_set_never_invents_labels_and_excludes_holdouts_from_rubric(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'ROOT',tmp_path)
    judge=storage.save('judge',{**JudgeSpec(name='Judge',rubric='Inspect each outcome.').model_dump(),'version':1})
    task={'requirements':[{'id':'placed','required':True,'definition':'Object inside target'}]}
    for status in ['pass','pass','fail','fail']:
        episode=storage.save('episode',{'index':1,'artifact_id':'unused'})
        storage.save('evaluation',{'episode_id':episode['id'],'judge':judge,'task':task,'status':status,'outcomes':[{'id':'placed','status':status}]})
    sample=calibration_sets.create_set(judge['id'])
    assert sample['status']=='awaiting_human_labels'
    assert all(m['agreement'] is None for m in sample['metrics'].values())
    assert not storage.all_records('outcome_label')
    for part,ids in sample['splits'].items():
        for eid in ids:
            storage.save('outcome_label',{'judge_id':judge['id'],'evaluation_id':eid,'outcomes':[{'id':'placed','status':'fail'}]})
    result=calibration_sets.detail(sample['id'])
    assert result['status']=='labelled'
    assert len(correction_evidence(judge['id']))==len(sample['splits']['development'])
    assert not {e['label']['evaluation_id'] for e in correction_evidence(judge['id'])}&set(sample['splits']['holdout'])
