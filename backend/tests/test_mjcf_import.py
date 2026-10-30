import math
import time

import pytest
from fastapi.testclient import TestClient

from studio import authoring_api, storage
from studio.adapters.arms import ur5e_available
from studio.adapters.franka import available
from studio.adapters.simulation import Simulator, record
from studio.articulated import CABINET_MJCF, open_drawer_scene
from studio.authoring import SceneProgram, compile_program
from studio.main import app
from studio.mjcf_import import MjcfError, convert
from studio.schemas import Scene
from studio.skills import compile_skills
from studio.task_api import bind_task, get_task, measure_task

franka = pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py")
ur5e = pytest.mark.skipif(not ur5e_available(), reason="Run integrations/icrt/prepare_arms.py")
LID = """<mujoco><worldbody><body name="base"><geom type="box" size=".05 .05 .01"/>
<body name="lid" pos="0 0 .05"><joint name="hinge" axis="0 1 0" range="-90 0"/><geom type="box" size=".05 .05 .002"/></body>
</body></worldbody></mujoco>"""


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)


@pytest.mark.parametrize("xml, message", [
    ('<mujoco><include file="/etc/passwd"/></mujoco>', "Unsupported MJCF elements: include"),
    ('<mujoco><asset><mesh file="a.stl"/></asset></mujoco>', "a.stl was not uploaded"),
    ('<mujoco><asset><texture name="t" builtin="checker"/></asset></mujoco>', "Unsupported MJCF elements: texture"),
    ('<mujoco><worldbody><geom type="mesh" mesh="a"/></worldbody></mujoco>', "could not compile"),
    ('<mujoco><tendon><fixed name="t"/></tendon></mujoco>', "path cannot be empty"),
    ('<mujoco><worldbody><geom type="plane" size="1 1 1"/></worldbody></mujoco>', "Geom type plane is unsupported"),
    ('<mujoco><actuator><motor joint="j"/></actuator></mujoco>', "unknown transmission target"),
    ('<mujoco><compiler meshdir="/etc"/></mujoco>', "unsupported attributes: meshdir"),
    ('<!DOCTYPE m [<!ENTITY x "y">]><mujoco/>', "DOCTYPE or entity"),
    ('<robot name="urdf"/>', "root element must be <mujoco>"),
    ("<mujoco><worldbody>", "not well-formed"),
    ('<mujoco><worldbody><body><joint type="hinge"/></body></worldbody></mujoco>', "could not compile"),
])
def test_sanitizer_rejects_files_assets_actuators_and_bad_xml(xml, message):
    with pytest.raises(MjcfError, match=message):
        convert(xml, "obj")


def test_conversion_resolves_degrees_defaults_and_names():
    result = convert(CABINET_MJCF.read_text(), "cab", (.05, .1, 0))
    cabinet = result["bodies"][0]
    drawer, door = cabinet["children"]
    assert cabinet["name"] == "cab_cabinet" and cabinet["position"] == [.05, .1, 0]
    assert drawer["joints"][0] == {**drawer["joints"][0], "name": "cab_drawer_slide", "type": "slide", "range": [0, .14], "damping": 1.5}
    assert door["joints"][0]["range"] == pytest.approx([-math.radians(100), 0], abs=1e-5)
    handle = next(g for g in drawer["geoms"] if g.get("name") == "bar")
    assert handle["color"] == pytest.approx([.25, .25, .27, 1]) and handle["friction"] == pytest.approx(1.2)
    assert [s["name"] for s in drawer["sites"]] == ["cab_drawer_pull"]
    assert any("Ignored <light>" in w for w in result["warnings"])
    Scene(bodies=result["bodies"])


def test_world_geoms_frame_bodies_and_yaw_placement():
    xml = """<mujoco><worldbody><geom name="slab" type="box" size=".1 .1 .01" pos="0 0 .01"/>
      <body name="arm" pos=".1 0 .05"><body name="tip" pos=".05 0 0"><joint name="j" type="ball"/>
      <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01"/></body></body></worldbody></mujoco>"""
    roots = convert(xml, "kit", (0, 0, 0), math.pi / 2)["bodies"]
    assert [r["name"] for r in roots] == ["kit_static", "kit_arm"]
    assert roots[1]["position"] == pytest.approx([0, .1, .05], abs=1e-6)
    assert roots[1]["euler_xyz"][2] == pytest.approx(math.pi / 2, abs=1e-5)
    assert roots[1]["geoms"][0]["collidable"] is False  # Frame-only body gets a tiny contact-free geom.
    tip = roots[1]["children"][0]
    assert tip["joints"][0]["type"] == "ball" and tip["geoms"][0]["shape"] == "capsule"
    assert tip["geoms"][0]["size"] == pytest.approx([.01, .01, .05], abs=1e-6)
    frame = Simulator(Scene(bodies=roots)).snapshot()
    assert set(frame["joints"]) == {"kit_j"}


def test_imported_hinge_starts_where_declared():
    roots = convert(LID, "box")["bodies"]
    assert roots[0]["children"][0]["joints"][0]["range"] == pytest.approx([-math.pi / 2, 0], abs=1e-6)
    frame = Simulator(Scene(bodies=roots)).snapshot()
    assert frame["joints"]["box_hinge"]["value"] == pytest.approx(0, abs=.02)


@pytest.mark.parametrize("embodiment", [pytest.param("franka-panda-v1", marks=franka), pytest.param("ur5e-2f85-v1", marks=ur5e)])
def test_open_drawer_preset_passes_on_each_contact_arm(embodiment):
    task = get_task("open-drawer")
    program = SceneProgram(name="Drawer", description="Open the drawer.", scene=open_drawer_scene())
    spec, artifact, verification = compile_program(program, task, embodiment)
    assert artifact["frames"][0]["joints"]["cabinet_drawer_slide"]["value"] == pytest.approx(0, abs=.005)
    assert verification["status"] == "pass", verification["outcomes"]


@franka
def test_open_drawer_is_not_passed_without_pulling():
    scene = open_drawer_scene()
    idle = compile_skills(scene, [{"kind": "move_to", "target": {"site": "cabinet_drawer_pull"}, "offset": [0, 0, .12]}, {"kind": "grasp"}])
    result = measure_task({**record(scene, idle), "task": bind_task(scene, get_task("open-drawer"))}, {"require_review": False})
    assert {o["id"]: o["status"] for o in result["outcomes"]}["drawer_open"] == "fail"


def test_api_previews_and_inserts_mjcf_as_a_revision():
    with TestClient(app) as client:
        preview = client.post("/api/mjcf/preview", json={"name": "lidbox", "xml": LID, "position": [.1, .1, 0]})
        assert preview.status_code == 200 and preview.json()["summary"]["joints"] == 1
        bad = client.post("/api/mjcf/preview", json={"name": "x", "xml": '<mujoco><include file="a.xml"/></mujoco>'})
        assert bad.status_code == 422 and "include" in bad.json()["detail"]
        parent = client.post("/api/scenes/template").json()
        child = client.post(f"/api/scenes/{parent['id']}/mjcf", json={"name": "lidbox", "xml": LID, "position": [.1, .1, 0]})
        assert child.status_code == 202
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = client.get("/api/scenes/" + child.json()["id"]).json()
            if result["status"] not in {"queued", "running"} and child.json()["id"] not in authoring_api.jobs:
                break
            time.sleep(.05)
        assert result["status"] == "ready", result.get("error")
        assert [b["name"] for b in result["spec"]["scene"]["bodies"]] == ["lidbox_base"]
        objects = client.get(f"/api/scenes/{result['id']}/objects").json()["objects"]
        assert "body:lidbox_base" in {o["id"] for o in objects}


CUBE_MESH = 'vertex="-.02 -.02 0 .02 -.02 0 .02 .02 0 -.02 .02 0 -.02 -.02 .04 .02 -.02 .04 .02 .02 .04 -.02 .02 .04" ' \
    'face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"'


def test_inline_and_stl_meshes_become_stored_scene_meshes():
    import struct
    xml = f"""<mujoco><asset><mesh name="block" {CUBE_MESH}/><mesh name="wedge" file="wedge.stl" scale="2 2 2"/></asset>
      <worldbody><body name="lid" pos=".1 0 .05"><joint name="h" type="hinge" axis="0 1 0" range="-1 0"/>
      <geom type="mesh" mesh="block" pos=".03 0 0"/><geom type="mesh" mesh="wedge"/></body></worldbody></mujoco>"""
    triangles = [((0, 0, 0), (.01, 0, 0), (0, .01, 0)), ((0, 0, 0), (0, .01, 0), (0, 0, .01)),
        ((0, 0, 0), (0, 0, .01), (.01, 0, 0)), ((.01, 0, 0), (0, 0, .01), (0, .01, 0))]
    stl = b"\0" * 80 + struct.pack("<I", 4) + b"".join(struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0) for a, b, c in triangles)
    stored = {}
    def store(normalized):
        stored[str(len(stored)).rjust(32, "a")] = normalized
        return str(len(stored) - 1).rjust(32, "a")
    result = convert(xml, "kit", files={"wedge.stl": stl}, store=store)
    block, wedge = result["bodies"][0]["geoms"]
    assert block["shape"] == "mesh" and block["size"] == pytest.approx([.02, .02, .02], abs=1e-6)
    assert block["position"] == pytest.approx([.03, 0, .02], abs=1e-6)  # Body-frame box center of the mesh.
    assert wedge["size"] == pytest.approx([.01, .01, .01], abs=1e-6)  # Scale 2 applied to the 1 cm STL.
    assert len(stored) == 2 and stored[block["asset_id"]]["vertex_count"] == 8


def test_mjcf_meshes_are_stored_simulated_and_recorded():
    from studio import storage as db
    from studio.mjcf_import import database_store
    xml = f"""<mujoco><asset><mesh name="block" {CUBE_MESH}/></asset><worldbody><body name="slab" pos="0 .1 0">
      <geom type="mesh" mesh="block"/></body></worldbody></mujoco>"""
    result = convert(xml, "slab", (0, 0, 0), files={}, store=database_store("slab"))
    asset_id = result["bodies"][0]["geoms"][0]["asset_id"]
    assert db.get("asset", asset_id)["source"] == "mjcf"
    frame = Simulator(Scene(bodies=result["bodies"])).snapshot()
    geom = next(g for g in frame["geoms"] if g["name"] == "geom_slab_0")
    assert geom["type"] == "mesh" and len(geom["mesh_vertices"]) == 8
    assert frame["bodies"]["slab"]["extent"] == pytest.approx([.02, .02, .02], abs=1e-4)


FOUR_BAR = """<mujoco><worldbody>
  <body name="ground" pos="0 0 .1"><geom type="box" size=".08 .01 .01"/>
    <body name="crank" pos="-.06 0 0"><joint name="crank_hinge" axis="0 1 0" damping=".1"/><geom type="capsule" fromto="0 0 0 0 0 .05" size=".006"/>
      <body name="coupler" pos="0 0 .05"><joint name="coupler_hinge" axis="0 1 0" damping=".1"/><geom type="capsule" fromto="0 0 0 .12 0 0" size=".006"/>
        <site name="tip" pos=".12 0 0"/></body></body>
    <body name="rocker" pos=".06 0 0"><joint name="rocker_hinge" axis="0 1 0" damping=".1"/><geom type="capsule" fromto="0 0 0 0 0 .05" size=".006"/></body>
  </body></worldbody>
  <equality><connect body1="coupler" body2="rocker" anchor=".12 0 0"/></equality>
  <contact><exclude body1="coupler" body2="rocker"/></contact></mujoco>"""


@pytest.mark.parametrize("embodiment", ["cartesian-gripper-v1", pytest.param("franka-panda-v1", marks=franka), pytest.param("ur5e-2f85-v1", marks=ur5e)])
def test_closed_linkage_constraint_survives_every_robot_and_gripper_release(embodiment):
    import numpy as np
    from studio.adapters.simulation import make_simulator
    result = convert(FOUR_BAR, "bar", (.05, .1, 0))
    assert result["equalities"][0]["type"] == "connect" and result["contact_excludes"] == [["bar_coupler", "bar_rocker"]]
    scene = Scene(embodiment=embodiment, include_cube=False, include_tray=False, bodies=result["bodies"],
        equalities=result["equalities"], contact_excludes=result["contact_excludes"])
    sim = make_simulator(scene)
    assert "scene_eq_0" in sim.source and 'body1="body_bar_coupler"' in sim.source
    for closed in [True, False, True, False]:
        sim.step({"x": -.22, "y": -.08, "z": .28, "closed": closed})
    tip = np.array(sim.snapshot()["sites"]["bar_tip"])
    rocker = sim.data.body("body_bar_rocker")
    rocker_top = rocker.xpos + rocker.xmat.reshape(3, 3) @ np.array([0, 0, .05])
    eq = sim.model.eq("scene_eq_0").id
    assert sim.data.eq_active[eq] and np.linalg.norm(tip - rocker_top) < .005


def test_constraints_must_reference_scene_parts():
    from pydantic import ValidationError
    roots = convert(LID, "box")["bodies"]
    with pytest.raises(ValidationError, match="unknown bodies or hinge/slide joints: ghost"):
        Scene(bodies=roots, equalities=[{"type": "connect", "body1": "ghost"}])
    with pytest.raises(ValidationError, match="Contact exclusion references unknown bodies"):
        Scene(bodies=roots, contact_excludes=[["box_base", "ghost"]])


def test_import_summary_uses_singular_counts():
    from studio.scene_editing import MjcfInsert, insert_mjcf
    from studio.authoring import SceneProgram
    scene = {"spec": SceneProgram(name="Empty", description="Empty.", scene=Scene()).model_dump(exclude_none=True), "task": None}
    _, summary, _ = insert_mjcf(scene, MjcfInsert(name="bar", xml=FOUR_BAR))
    assert summary == "Imported bar from MJCF: 4 bodies, 3 joints, 1 site, 1 constraint."


DOOR = """<mujoco><worldbody><body name="frame"><geom type="box" size=".01 .01 .08" pos="0 0 .08"/><site name="anchor" pos="0 .1 .15"/>
 <body name="door" pos="0 0 .08"><joint name="hinge" axis="0 0 1" range="-90 0" damping=".5"/><geom type="box" size=".005 .045 .07" pos="0 .06 0"/>
 <site name="handle" pos="0 .09 0"/></body></body></worldbody>
 <tendon><spatial name="spring" stiffness="5" springlength=".05"><site site="anchor"/><site site="handle"/></spatial>
   <fixed name="pair"><joint joint="hinge" coef="1"/></fixed></tendon>
 <actuator><position name="opener" joint="hinge" kp="2" kv=".2" ctrlrange="-1.6 0"/><motor name="pull" tendon="spring" gear="3" ctrlrange="0 1"/>
   <velocity name="spin" joint="hinge" kv=".4" ctrlrange="-1 1"/></actuator></mujoco>"""


def test_tendons_and_actuators_import_with_their_parameters():
    result = convert(DOOR, "door", (.1, .1, 0))
    spring, pair = result["tendons"]
    assert spring == {**spring, "name": "door_spring", "type": "spatial", "sites": ["door_anchor", "door_handle"], "stiffness": 5, "rest_length": .05}
    assert pair["type"] == "fixed" and pair["joints"] == [{"joint": "door_hinge", "coef": 1}]
    opener, pull, spin = result["actuators"]
    assert opener == {**opener, "type": "position", "joint": "door_hinge", "kp": 2, "kv": pytest.approx(.2), "ctrl_range": [-1.6, 0], "schedule": [[0, 0]]}
    assert pull == {**pull, "type": "motor", "tendon": "door_spring", "gear": 3}
    assert spin == {**spin, "type": "velocity", "kv": pytest.approx(.4)}


@pytest.mark.parametrize("xml, message", [
    ('<mujoco><worldbody><body><joint name="j"/><geom size=".02"/></body></worldbody><actuator><general joint="j"/></actuator></mujoco>', "Unsupported MJCF elements: general"),
    ('<mujoco><worldbody><body><joint name="j"/><geom size=".02"/><site name="s"/></body></worldbody><actuator><motor site="s"/></actuator></mujoco>', "unsupported transmission"),
    ('<mujoco><worldbody><body><joint name="j"/><geom name="g" size=".02"/><site name="a"/><site name="b"/></body></worldbody><tendon><spatial><site site="a"/><geom geom="g"/><site site="b"/></spatial></tendon></mujoco>', "only pass through sites"),
    ('<mujoco><worldbody><body><joint name="j"/><geom size=".02"/></body></worldbody><actuator><position joint="j" kp="1" gear="2"/></actuator></mujoco>', "gear other than 1"),
])
def test_unsupported_drives_are_rejected(xml, message):
    with pytest.raises(MjcfError, match=message):
        convert(xml, "x")


@pytest.mark.parametrize("embodiment", ["cartesian-gripper-v1", pytest.param("franka-panda-v1", marks=franka), pytest.param("ur5e-2f85-v1", marks=ur5e)])
def test_scheduled_actuator_drives_scene_without_disturbing_the_robot(embodiment):
    from studio.adapters.simulation import make_simulator
    result = convert(DOOR, "door", (.1, .1, 0))
    actuators = [{**result["actuators"][0], "schedule": [[0, 0], [1, -1.2]]}]
    scene = Scene(embodiment=embodiment, include_cube=False, include_tray=False, bodies=result["bodies"], tendons=result["tendons"][:1], actuators=actuators)
    sim = make_simulator(scene)
    for _ in range(int(3 * sim.frequency)):
        frame = sim.step({"x": -.22, "y": -.08, "z": .28, "closed": False})
    assert frame["actuators"]["door_opener"]["ctrl"] == -1.2
    assert frame["joints"]["door_hinge"]["value"] == pytest.approx(-1.2, abs=.02)
    assert frame["ee"] == pytest.approx([-.22, -.08, .28], abs=.003)
    assert frame["tendons"]["door_spring"]["length"] > .1


def test_actuator_schedule_validation():
    from pydantic import ValidationError
    roots = convert(LID, "box")["bodies"]
    with pytest.raises(ValidationError, match="schedule starts at 0 s and increases"):
        Scene(bodies=roots, actuators=[{"name": "a", "joint": "box_hinge", "schedule": [[1, 0]]}])
    with pytest.raises(ValidationError, match="inside its control range"):
        Scene(bodies=roots, actuators=[{"name": "a", "joint": "box_hinge", "schedule": [[0, 5]]}])
    with pytest.raises(ValidationError, match="unknown joint or tendon"):
        Scene(bodies=roots, actuators=[{"name": "a", "joint": "ghost"}])


def test_site_based_equalities_become_anchored_body_constraints():
    from studio.adapters.simulation import make_simulator
    xml = """<mujoco><worldbody><body name="post"><geom type="box" size=".01 .01 .05" pos="0 0 .05"/><site name="top" pos="0 0 .1"/>
      <body name="arm" pos="0 0 .1"><joint name="swing" axis="0 1 0" damping=".1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".005"/>
        <site name="tip" pos=".1 0 0"/></body></body>
      <body name="rod" pos=".1 0 .1"><freejoint name="loose"/><geom type="box" size=".005 .005 .03"/>
        <site name="eye" pos="0 0 0"/></body></worldbody>
      <equality><connect site1="tip" site2="eye"/></equality></mujoco>"""
    result = convert(xml, "kit", (.05, .1, 0))
    assert result["equalities"][0] == {**result["equalities"][0], "type": "connect", "body1": "kit_arm", "body2": "kit_rod", "anchor": [.1, 0, 0]}
    scene = Scene(include_cube=False, include_tray=False, bodies=result["bodies"], equalities=result["equalities"])
    sim = make_simulator(scene)
    for _ in range(40):
        frame = sim.step({"x": -.22, "y": -.08, "z": .28, "closed": False})
    import numpy as np
    # MuJoCo's default soft constraint stretches a few millimetres while the linkage swings.
    assert np.linalg.norm(np.array(frame["sites"]["kit_tip"]) - np.array(frame["sites"]["kit_eye"])) < .01
