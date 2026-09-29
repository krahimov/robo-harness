"""Trusted bpy operations. Input is validated by the studio, never executed as code."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def material(name, color, metallic=.2, roughness=.4):
    item = bpy.data.materials.new(name)
    item.diffuse_color = color
    item.use_nodes = True
    shader = item.node_tree.nodes.get("Principled BSDF")
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Roughness'].default_value = roughness
    return item


def primitive(name, shape, size, position, rotation=(0, 0, 0), bevel=0):
    if shape == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=size[0], depth=size[2]*2)
    elif shape == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=size[0])
    else:
        bpy.ops.mesh.primitive_cube_add(size=2)
        bpy.context.object.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    item = bpy.context.object
    item.name = name
    item.location = position
    item.rotation_euler = rotation
    if bevel:
        mod = item.modifiers.new('Manufactured edges', 'BEVEL')
        mod.width = min(bevel, min(size)/3)
        mod.segments = 1
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return item


def build(folder, program):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1
    meshes = []
    for assembly in program['assemblies']:
        mat = material(assembly['name'], assembly['color'], assembly['metallic'], assembly['roughness'])
        objects, collisions = [], []
        for index, part in enumerate(assembly['parts']):
            item = primitive(f"{assembly['name']}_{index}", part['shape'], part['size'], part['position'], part['euler_xyz'], part['bevel'])
            item.data.materials.append(mat)
            bpy.context.view_layer.update()
            if part['collision']:
                # Each separate part keeps gaps and openings. This is an AABB,
                # not a convex hull of the entire machine or bin.
                corners = [item.matrix_world @ Vector(c) for c in item.bound_box]
                low = [min(c[i] for c in corners) for i in range(3)]
                high = [max(c[i] for c in corners) for i in range(3)]
                collisions.append({'position': [(a+b)/2 for a,b in zip(low, high)],
                    'size': [(b-a)/2 for a,b in zip(low, high)]})
            objects.append(item)
        vertices, faces = [], []
        for item in objects:
            offset = len(vertices)
            item.data.calc_loop_triangles()
            vertices.extend([list(item.matrix_world @ v.co) for v in item.data.vertices])
            faces.extend([[offset+i for i in t.vertices] for t in item.data.loop_triangles])
        if len(vertices) > 1200 or len(faces) > 2400:
            raise ValueError('Generated mesh exceeds geometry budget')
        low = [min(v[i] for v in vertices) for i in range(3)]
        high = [max(v[i] for v in vertices) for i in range(3)]
        obj = '\n'.join(['v '+' '.join(f'{x:.8f}' for x in v) for v in vertices]+['f '+' '.join(str(i+1) for i in f) for f in faces])+'\n'
        (folder/(assembly['name']+'.obj')).write_text(obj)
        meshes.append({'name': assembly['name'], 'color': assembly['color'], 'position': [(a+b)/2 for a,b in zip(low,high)],
            'size': [(b-a)/2 for a,b in zip(low,high)], 'collisions': collisions,
            'vertices': len(vertices), 'triangles': len(faces)})
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'scene.blend'))
    (folder/'manifest.json').write_text(json.dumps({'format':'studio-blender-v1','blender_version':bpy.app.version_string,
        'units':'metres','up_axis':'Z','collision_method':'separate part AABBs','meshes':meshes},indent=2))


def preview(folder, geoms):
    bpy.ops.wm.open_mainfile(filepath=str(folder/'scene.blend'))
    for geom in geoms:
        if geom['color'][3] == 0:
            continue
        if geom['type'] == 'mesh':
            data=bpy.data.meshes.new(geom['name'])
            data.from_pydata(geom['vertices'], [], geom['faces'])
            for polygon in data.polygons:
                polygon.use_smooth=True
            data.update()
            item=bpy.data.objects.new(geom['name'],data)
            bpy.context.collection.objects.link(item)
        else:
            size=geom['size']
            if geom['type']=='cylinder':
                size=[size[0],size[0],size[1]]
            item=primitive(geom['name'],geom['type'],size,[0,0,0])
        transform=Matrix([geom['matrix'][i:i+3] for i in range(0,9,3)]).to_4x4()
        transform.translation=Vector(geom['position'])
        item.matrix_world=transform
        item.data.materials.append(material(geom['name'],geom['color']))
    scene=bpy.context.scene
    if (folder/'ROBOT_ASSETS_LICENSE.txt').is_file():
        bpy.data.texts.new('ROBOT_ASSETS_LICENSE.txt').write((folder/'ROBOT_ASSETS_LICENSE.txt').read_text())
    bpy.ops.object.camera_add(location=(1.35,-1.65,1.25))
    camera=bpy.context.object
    camera.rotation_euler=(Vector((0,0,.14))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.lens=45
    scene.camera=camera
    for position,power,size in [((0,-.5,2.5),180,3),((-.9,.6,1.8),120,2)]:
        bpy.ops.object.light_add(type='AREA',location=position)
        lamp=bpy.context.object
        lamp.data.energy=power
        lamp.data.shape='DISK'
        lamp.data.size=size
    scene.world=bpy.data.worlds.new('Studio world')
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.18,.21,.24,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.5
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=12
    scene.render.threads_mode='FIXED'
    scene.render.threads=6
    scene.render.resolution_x=768
    scene.render.resolution_y=576
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath=str(folder/'preview.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'scene.blend'))
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    mode, directory = sys.argv[1:3]
    folder=Path(directory).resolve()
    if mode=='build':
        build(folder,json.loads((folder/'design.json').read_text()))
    elif mode=='preview':
        preview(folder,json.loads((folder/'fixture.json').read_text()))
    else:
        raise ValueError('Unsupported Blender worker operation')
