"""A model-free Blender conveyor example for checking the import contract."""
import math
from studio.blender_schema import BlenderProgram


def conveyor_design():
    rails=[{'position':[-.22,y,.025],'size':[.205,.009,.025]} for y in [-.165,.005]]
    rollers=[{'shape':'cylinder','position':[-.39+i*.045,-.08,.025],'size':[.013,.013,.07],
        'euler_xyz':[math.pi/2,0,0],'collision':False,'bevel':.001} for i in range(9)]
    cabinet=[{'position':[-.36,.30,.10],'size':[.10,.055,.10],'bevel':.005},
        {'position':[-.36,.242,.12],'size':[.075,.005,.065],'bevel':.002,'collision':False}]
    return BlenderProgram.model_validate({'name':'Blender conveyor / Franka','description':'Blender-built frame, rollers, and control cabinet with a contact-based robot transfer.',
        'assumptions':['Procedural example, not reconstructed from a photograph.',
            'Rollers are static visual geometry below an approximate MuJoCo conveyor surface.',
            'Separate axis-aligned collision boxes approximate each solid machinery part. Robot and box dynamics remain in MuJoCo.'],
        'scene':{'embodiment':'franka-panda-v1','conveyor':{'x':-.22,'y':-.08,'length':.38,'width':.14,'height':.04,'speed':.025,'direction':'x'}},
        'assemblies':[{'name':'blue_frame','color':[.045,.21,.40,1],'parts':rails},
            {'name':'steel_rollers','color':[.6,.65,.68,1],'metallic':.8,'parts':rollers},
            {'name':'control_cabinet','color':[.30,.40,.44,1],'parts':cabinet}]})
