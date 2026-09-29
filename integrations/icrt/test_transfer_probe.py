import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from coordinate_probe import transform,episode_transform

def test_rigid_change_preserves_local_action_and_round_trips():
    p=np.array([[.4,-.2,.3,2.9,.2,-.1,.1],[.5,-.1,.2,3.,-.1,.3,.8]])
    a=p.copy();a[:,:3]+=[.01,.02,-.03];a[:,6]=[0.,1.]
    r=Rotation.from_euler('XYZ',[.2,-.3,.7]);t=np.array([-.2,.4,.1])
    pp,aa=transform(p,r,t),transform(a,r,t)
    before=Rotation.from_euler('XYZ',p[:,3:6]).inv().apply(a[:,:3]-p[:,:3])
    after=Rotation.from_euler('XYZ',pp[:,3:6]).inv().apply(aa[:,:3]-pp[:,:3])
    np.testing.assert_allclose(before,after,atol=1e-12)
    restored=transform(pp,r.inv(),-r.inv().apply(t))
    np.testing.assert_allclose(restored[:,:3],p[:,:3],atol=1e-12)
    np.testing.assert_allclose(Rotation.from_euler('XYZ',restored[:,3:6]).as_matrix(),Rotation.from_euler('XYZ',p[:,3:6]).as_matrix(),atol=1e-12)
    np.testing.assert_array_equal(aa[:,6],a[:,6])

def test_intervention_keeps_images_and_source_unchanged():
    p=np.array([[.5,.1,.3,3.,0.,0.,.1]])
    image=Image.new('RGB',(8,8),'white')
    e={'frames':[{'side':image,'wrist':image,'proprio':p[0]}],'proprio':p,'actions':p.copy()}
    changed=episode_transform(e,Rotation.identity(),[.2,0,0])
    np.testing.assert_allclose(e['proprio'],p)
    assert changed['frames'][0]['side'] is image
    np.testing.assert_allclose(changed['frames'][0]['proprio'][:3],[.7,.1,.3])
