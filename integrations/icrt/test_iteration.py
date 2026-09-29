"""Experiment boundaries: proposals cannot redefine outcomes or evaluation seeds."""
import json
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backend'))
from improve import Candidate


def test_optimizer_rejects_goal_changes_and_actor_substitution():
    base=dict(view='close',actions='commanded',demos=1,pace=1,demo_variation=.02,lock_rotation=False,binary_gripper=False)
    assert Candidate.model_validate(base).demos==1
    for extra in ({'task':{}},{'actor':'replay'},{'seeds':[42]},{'success':'true'}):
        with pytest.raises(ValidationError):
            Candidate.model_validate({**base,**extra})
    for changes in ({'demos':5},{'pace':0},{'demo_variation':float('nan')}):
        with pytest.raises(ValidationError):
            Candidate.model_validate({**base,**changes})


def test_failed_teacher_is_feedback_and_never_runs_actor(tmp_path,monkeypatch):
    import argparse
    import iterate
    class Camera:
        def close(self):pass
    class Client:
        def __init__(self,**kwargs):pass
        def post(self,*args,**kwargs):raise AssertionError('Actor must not see a failed teacher')
        def close(self):pass
    monkeypatch.setattr(iterate,'Cameras',Camera)
    monkeypatch.setattr(iterate.httpx,'Client',Client)
    monkeypatch.setattr(iterate,'default_waypoints',lambda scene:[])
    monkeypatch.setattr(iterate,'record',lambda *args:{'xml':'<mujoco/>','frames':[],'actions':[]})
    monkeypatch.setattr(iterate,'metrics',lambda artifact:{'verdict':{'status':'fail'}})
    args=argparse.Namespace(output=tmp_path/'candidate',task='stationary',view='original',actions='commanded',demos=1,demo_variation=.02,pace=1,steps=20,seeds=[2000],endpoint='http://127.0.0.1:8790',lock_rotation=False,binary_gripper=False)
    report=iterate.experiment(args)
    assert report['status']=='rejected_demo'
    assert report['episodes']==0
    assert (args.output/'report.json').exists()
    with pytest.raises(FileExistsError):iterate.experiment(args)


def test_motion_program_cannot_change_geometry_and_rejects_nonfinite_offsets():
    from tool_actor import Program
    motion=dict(target='cube',offset=[0,0,.1],closed=False,seconds=1)
    assert Program.model_validate({'motions':[motion,motion]}).motions[0].target=='cube'
    with pytest.raises(ValidationError):
        Program.model_validate({'motions':[motion,motion],'scene':{'friction':2}})
    with pytest.raises(ValidationError):
        Program.model_validate({'motions':[{**motion,'offset':[float('nan'),0,0]},motion]})
