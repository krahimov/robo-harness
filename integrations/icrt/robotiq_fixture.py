"""Research Franka/2F-85 fixture. Approximate mounting and cameras, not calibrated DROID."""
import math,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'backend'))
from studio.adapters.franka import FrankaSimulator,franka_xml,rotation_matrix,euler_xyz,BASE
from studio.adapters.icrt_frames import FLANGE_FROM_HAND

TIP_OFFSET=np.array([0.,0.,.1488])

def aimed(camera,position,target,fovy=55):
    z=np.array(position)-target;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    camera.set('pos',' '.join(map(str,position)));camera.set('xyaxes',' '.join(map(str,np.r_[x,y])));camera.set('fovy',str(fovy))

def fixture_xml(scene,close_view=True):
    root=ET.fromstring(franka_xml(scene));folder=ROOT/'.data/models/menagerie/robotiq_2f85'
    grip=ET.parse(folder/'2f85.xml').getroot()
    # Keep imported names separate from the Panda arm assets and defaults.
    names={n.get('name'):'rq_'+n.get('name') for n in grip.iter() if n.get('name')}
    names.update({n.get('class'):'rq_'+n.get('class') for n in grip.iter('default') if n.get('class')})
    for mesh in grip.findall('asset/mesh'):
        name=Path(mesh.get('file')).stem;mesh.set('name',name);names[name]='rq_'+name
        mesh.set('file',str(folder/'assets'/mesh.get('file')))
    names.update(base_mount='hand',right_driver_joint='finger_joint1',left_driver_joint='finger_joint2',left_pad='left_finger',right_pad='right_finger',pinch='ee')
    refs={'name','class','childclass','mesh','material','joint','joint1','joint2','tendon','body1','body2'}
    for node in grip.iter():
        for key,value in list(node.attrib.items()):
            if key in refs and value in names:node.set(key,names[value])
    world=root.find('worldbody');world.find("body[@name='cube']").set('quat',f'{math.cos(math.pi/8)} 0 0 {math.sin(math.pi/8)}');parent=root.find(".//body[@name='link7']");old=parent.find("body[@name='hand']")
    replacement=grip.find('worldbody/body');replacement.set('pos',old.get('pos'));replacement.set('quat',old.get('quat'))
    # 2F-85 fingers close along the inherited hand's Y axis.
    replacement.find('body').set('quat','1 0 0 0')
    parent.remove(old);parent.append(replacement)
    for section in ['tendon','equality','contact']:
        old_section=root.find(section)
        if old_section is not None:root.remove(old_section)
        addition=grip.find(section)
        if addition is not None:root.append(addition)
    root.find('actuator').remove(root.find("actuator/general[@name='actuator8']"))
    actuator=grip.find('actuator/general')
    # Preserve upstream stiffness/force bound; reverse UI control so 255 opens.
    actuator.set('gainprm','-0.3137255 0 0');actuator.set('biasprm','80 -100 -10');root.find('actuator').append(actuator)
    for section in ['asset','default']:
        for node in grip.find(section):root.find(section).append(node)
    root.find('option').set('impratio','10');root.find('option').set('cone','elliptic')
    hand=root.find(".//body[@name='hand']")
    wrist=ET.SubElement(hand,'camera',name='wrist')
    aimed(wrist,[.12,0,.065],TIP_OFFSET,65)
    if close_view:
        aimed(next(c for c in world.iter('camera') if c.get('name')=='overview'),[.32,-.55,.48],np.array([-.075,0,.035]),55)
    for i,g in enumerate(world.iter('geom')):
        if not g.get('name'):g.set('name','rq_fixture_geom_'+str(i))
    return ET.tostring(root,encoding='unicode')

class RobotiqFranka(FrankaSimulator):
    id='mujoco-franka-robotiq-research-v1'
    def build_source(self,scene):return fixture_xml(scene)
    def __init__(self,scene):
        super().__init__(scene)
        # Approximate the released reference's starting flange pose, not its actions.
        initial=self.inverse_kinematics(np.array([.0473,-.1487,.2128]),[math.pi,0,-math.pi/4])
        self.data.qpos[self.arm_qpos]=initial;self.data.qvel[:]=0;self.data.ctrl[:7]=initial
        mujoco.mj_forward(self.model,self.data)
        for _ in range(120):
            self.data.qfrc_applied[self.arm_dofs]=self.data.qfrc_bias[self.arm_dofs]
            mujoco.mj_step(self.model,self.data)
        self.data.time=0
    def snapshot(self):
        state=super().snapshot()
        # Actual pad separation, normalized by nominal 85 mm opening.
        left=self.data.geom('rq_left_pad1').xpos;right=self.data.geom('rq_right_pad1').xpos
        width=max(0.,float(np.linalg.norm(left-right))-.008)
        state.update(gripper=float(np.clip(1-width/.085,0,1)),embodiment='franka-robotiq-2f85-research-v1',gripper_width_m=width)
        return state

def pose(frame):
    rot=rotation_matrix(frame['euler_xyz'])
    return [*(np.array(frame['ee'])-BASE-rot@TIP_OFFSET),*euler_xyz(rot@FLANGE_FROM_HAND),float(frame.get('gripper',frame['closed']))]

def action(raw,frame):
    rot=rotation_matrix(raw[3:6])@FLANGE_FROM_HAND.T
    xyz=np.array(raw[:3])+BASE+rot@TIP_OFFSET
    bounded=np.clip(np.clip(xyz,[-.5,-.35,.025],[.5,.35,.45]),np.array(frame['ee'])-.04,np.array(frame['ee'])+.04)
    return dict(zip(['x','y','z'],bounded.tolist()))|{'euler_xyz':euler_xyz(rot),'closed':bool(raw[6]>.5),'gripper':float(np.clip(raw[6],0,1))},bool(not np.allclose(xyz,bounded))
