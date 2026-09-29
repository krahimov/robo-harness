"""Versioned pen fixture variations, fixed before any evaluation."""
from copy import deepcopy
import math
from studio.schemas import Scene

CASES = [
    {'id':'baseline','label':'Baseline'},
    {'id':'position_left','label':'Pen 20 mm left','pen_dx':-.02},
    {'id':'position_right','label':'Pen 20 mm right','pen_dx':.02},
    {'id':'angle_plus','label':'Pen rotated +15 degrees','yaw':math.pi/12},
    {'id':'angle_minus','label':'Pen rotated -15 degrees','yaw':-math.pi/12},
    {'id':'thin_pen','label':'12 mm diameter pen','radius':.006},
    {'id':'shifted_pouch','label':'Pouch shifted 40 mm','pouch_dx':.04},
    {'id':'obstacle','label':'Obstacle near transfer path','obstacle':True},
]


def case_scene(scene, task, case, seed):
    value=scene.model_dump();value['seed']=seed;resolved=deepcopy(task)
    pen=next(o for o in value['objects'] if o['id']=='pen')
    dx=case.get('pen_dx',0);pen['position'][0]+=dx
    for prop in value['props']:
        if prop['name'].startswith('pen_rest_'):prop['position'][0]+=dx
    if case.get('yaw'):
        # XYZ intrinsic roll=-pi/2 rotates the cylinder onto the table; pitch
        # rotates its long axis in the XY plane. Match its rests to that axis.
        pen['euler_xyz']=[-math.pi/2,-case['yaw'],0]
        c,s=math.cos(case['yaw']),math.sin(case['yaw'])
        for prop in value['props']:
            if prop['name'].startswith('pen_rest_'):
                x,y=prop['position'][0]-pen['position'][0],prop['position'][1]-pen['position'][1]
                prop['position'][:2]=[pen['position'][0]+c*x-s*y,pen['position'][1]+s*x+c*y]
    if case.get('radius'):
        pen['size'][:2]=[case['radius']]*2
        pen['position'][2]=.033+case['radius']
    shift=case.get('pouch_dx',0)
    if shift:
        for prop in value['props']:
            if prop['name'].startswith(('pouch_','handle_')):prop['position'][0]+=shift
        for requirement in resolved['requirements']:
            if requirement.get('bounds'):
                requirement['bounds'][0]+=shift;requirement['bounds'][1]+=shift
        resolved['fixture_transform']={'translation':[shift,0,0],'reason':'Predeclared pouch relocation; dimensions and success predicates unchanged'}
    if case.get('obstacle'):
        value['props'].append({'name':'transfer_obstacle','shape':'box','position':[-.075,.015,.07],
            'size':[.025,.025,.07],'color':[.65,.3,.1,1],'collidable':True})
    return Scene.model_validate(value),resolved
