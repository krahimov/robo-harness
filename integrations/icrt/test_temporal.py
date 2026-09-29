"""Ensembled predictions must retain their original world-frame anchors."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from temporal_runtime import TemporalRuntime


def test_ensemble_does_not_reanchor_old_chunks_to_new_robot_pose():
    source=Path(__file__).resolve().parents[2]/'.data/models/icrt-source'
    if not source.exists():pytest.skip('Pinned upstream source not installed')
    sys.path.insert(0,str(source))
    from icrt.models.policy.icrt import ICRT
    class Policy:
        reset=ICRT.reset
        num_pred_steps=16
        def forward_inference(self,obs,start_pos):
            a=torch.tensor([.1,0,0,1,0,0,0,1,0,.4])[None,None].repeat(1,16,1)
            a[0,:,0]=torch.arange(1,17)*.1
            return a,None
    r=TemporalRuntime.__new__(TemporalRuntime)
    r.model=Policy();r.device=torch.device('cpu');r.horizon=2;r.reset([])
    def prepare(frames):
        x=frames[0]['x']
        return {'observation':torch.zeros(1,1,1),'proprio':torch.tensor([x,0,0,1,0,0,0,1,0,0],dtype=torch.float32)[None,None],'action':None}
    r._prepare=prepare
    assert r.act({'x':0})[0]==pytest.approx(.1)
    expected=(.2+np.exp(.05)*1.1)/(1+np.exp(.05))
    result=r.act({'x':1})
    assert result[0]==pytest.approx(expected)
    assert result[-1]==pytest.approx(.4)
    assert r.model.last_action[0,0,0].item()==pytest.approx(expected-1)
    r.reset([])
    assert len(r.chunks)==0
    assert r.act({'x':2})[0]==pytest.approx(2.1)
