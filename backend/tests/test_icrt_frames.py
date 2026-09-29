import math
import numpy as np
import pytest
from studio.adapters.icrt_frames import tip_to_flange, flange_to_tip
from studio.adapters.franka import available, FrankaSimulator, rotation_matrix
from studio.schemas import Scene


def test_known_downward_pose_flange_is_above_tip():
    p,r=tip_to_flange([.4,.1,.025],[math.pi,0,0])
    np.testing.assert_allclose(p,[.4,.1,.1284],atol=1e-9)
    assert r[2]==pytest.approx(math.pi/4)
    q,e=flange_to_tip(p,r)
    np.testing.assert_allclose(q,[.4,.1,.025],atol=1e-9)
    np.testing.assert_allclose(rotation_matrix(e),rotation_matrix([math.pi,0,0]),atol=1e-9)


@pytest.mark.skipif(not available(),reason='Menagerie assets not installed')
def test_converted_site_matches_independent_link7_forward_kinematics():
    sim=FrankaSimulator(Scene(embodiment='franka-panda-v1'))
    for target in [[-.22,-.08,.28],[-.1,.03,.2]]:
        f=sim.step(dict(zip(['x','y','z'],target))|{'closed':False,'euler_xyz':[2.9,.1,-.2]})
        p,r=tip_to_flange(f['ee'],f['euler_xyz'])
        link=sim.data.body('link7')
        R=link.xmat.reshape(3,3)
        expected=link.xpos+R@np.array([0,0,.107])
        np.testing.assert_allclose(p,expected,atol=1e-5)
        np.testing.assert_allclose(rotation_matrix(r),R,atol=1e-5)
