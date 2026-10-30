import pytest
from pydantic import ValidationError

from studio.adapters.franka import available
from studio.adapters.simulation import Simulator, record
from studio.articulated import close_box_scene, close_box_skills
from studio.authoring import SceneProgram, compile_program
from studio.robots import catalog
from studio.schemas import Scene
from studio.skills import compile_skills
from studio.task_api import TaskCreate, bind_task, get_task, measure_task

franka = pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py to install the licensed assets")
JUDGE = {"require_review": False}


def drawer(**changes):
    return {"name": "cabinet", "position": [.1, .1, 0], "geoms": [{"size": [.08, .08, .01], "position": [0, 0, .01]}],
        "children": [{"name": "drawer", "position": [0, 0, .05],
            "joints": [{"name": "drawer_slide", "type": "slide", "axis": [0, -1, 0], "range": [0, .1], "initial": .04, "damping": 2}],
            "geoms": [{"size": [.05, .05, .02]}], "sites": [{"name": "handle", "position": [0, -.06, 0]}]}], **changes}


def test_scene_graph_rejects_invalid_structure():
    with pytest.raises(ValidationError, match="Body names must be unique"):
        Scene(bodies=[drawer(), drawer()])
    with pytest.raises(ValidationError, match="initial value must lie inside"):
        Scene(bodies=[drawer(children=[{"name": "d", "joints": [{"name": "j", "type": "slide", "range": [0, .1], "initial": .2}], "geoms": [{"size": [.02, .02, .02]}]}])])
    with pytest.raises(ValidationError, match="Only root bodies can use a free joint"):
        Scene(bodies=[drawer(children=[{"name": "d", "joints": [{"name": "j", "type": "free"}], "geoms": [{"size": [.02, .02, .02]}]}])])
    with pytest.raises(ValidationError, match="must sit on the table"):
        Scene(bodies=[drawer(position=[.9, 0, 0])])
    with pytest.raises(ValidationError, match="clash with existing objects"):
        Scene(bodies=[drawer(name="cube")])
    with pytest.raises(ValidationError, match="Extra inputs"):
        Scene(bodies=[drawer(mjcf="<include file='/etc/passwd'/>")])


def test_joint_state_is_recorded_and_legacy_frames_are_unchanged():
    frame = Simulator(Scene(bodies=[drawer()])).snapshot()
    assert frame["joints"]["drawer_slide"]["type"] == "slide"
    assert frame["joints"]["drawer_slide"]["value"] == pytest.approx(.04, abs=.005)
    assert set(frame["bodies"]) == {"cabinet", "drawer"}
    assert frame["sites"]["handle"][1] == pytest.approx(.1 - .04 - .06, abs=.006)
    assert {"geom_drawer_0", "geom_cabinet_0"} <= {g["name"] for g in frame["geoms"]}
    legacy = Simulator(Scene()).snapshot()
    assert not {"joints", "bodies", "sites"} & set(legacy)


def test_joint_range_check_passes_fails_and_reports_missing_state():
    requirement = {"id": "shut", "label": "Shut", "definition": "Drawer closed.", "required": True,
        "check": "joint_range", "object_id": "cube", "joint": "drawer_slide", "min_value": 0, "max_value": .005}
    task = {"name": "Drawer", "version": 1, "requirements": [requirement]}
    frames = [{"t": 0, "joints": {"drawer_slide": {"value": .04}}}, {"t": 1, "joints": {"drawer_slide": {"value": .002}}}]
    assert measure_task({"frames": frames, "task": task}, JUDGE)["status"] == "pass"
    frames[-1]["joints"]["drawer_slide"]["value"] = .03
    assert measure_task({"frames": frames, "task": task}, JUDGE)["status"] == "fail"
    assert measure_task({"frames": [{"t": 0}], "task": task}, JUDGE)["outcomes"][0]["status"] == "unknown"


def test_task_validation_for_joints_and_skills():
    base = {"name": "Drawer", "description": "Close the drawer."}
    with pytest.raises(ValidationError, match="joint_range requirement needs"):
        TaskCreate(**base, requirements=[{"id": "a", "label": "a", "definition": "a", "check": "joint_range", "joint": "j"}])
    with pytest.raises(ValidationError, match="either teacher operations or teacher skills"):
        TaskCreate(**base, requirements=[{"id": "a", "label": "a", "definition": "a", "check": "released"}],
            operations=[{"destination": "tray"}], teacher_skills=[{"kind": "grasp"}])
    task = {**TaskCreate(**base, requirements=[{"id": "a", "label": "a", "definition": "a", "check": "joint_range",
        "joint": "missing", "min_value": 0, "max_value": 1}]).model_dump(), "schema_version": 2}
    with pytest.raises(ValueError, match="missing task joints: missing"):
        bind_task(Scene(bodies=[drawer()]), task)


def test_cartesian_gripper_cannot_push_articulated_parts():
    scene = Scene(bodies=[drawer()])
    skills = [{"kind": "move_to", "target": {"site": "handle"}}, {"kind": "follow_joint", "joint": "drawer_slide", "site": "handle", "to": 0}]
    with pytest.raises(ValueError, match="Cartesian gripper cannot push"):
        compile_skills(scene, skills)
    with pytest.raises(ValueError, match="unknown site nowhere"):
        compile_skills(scene, [{"kind": "move_to", "target": {"site": "nowhere"}}, {"kind": "grasp"}])
    with pytest.raises(ValueError, match="outside the robot workspace"):
        compile_skills(scene, [{"kind": "move_to", "target": {"point": [0, 0, .6]}}, {"kind": "grasp"}])
    assert {r["id"]: r["capabilities"] for r in catalog()}["cartesian-gripper-v1"] == ["idealized_cloth_pinch", "weld_grasp"]


@franka
def test_close_box_preset_teacher_closes_both_flaps_and_saves_skills():
    task = get_task("close-box-flaps")
    program = SceneProgram(name="Close box", description="Close both flaps.", scene=task["suggested_scene"])
    spec, artifact, verification = compile_program(program, task, "franka-panda-v1")
    assert verification["status"] == "pass", verification["outcomes"]
    assert [(s["kind"], s.get("target")) for s in spec["skills"]] == [(s["kind"], s.get("target")) for s in close_box_skills()]
    assert "waypoints" not in spec
    first = artifact["frames"][0]["joints"]
    assert abs(first["flap_left_hinge"]["value"]) > 1.5 and abs(first["flap_right_hinge"]["value"]) > 1.5
    # The saved program replays the same demonstration.
    again = compile_program(SceneProgram.model_validate(spec), task, "franka-panda-v1")[1]
    assert again["actions"] == artifact["actions"]


@franka
def test_close_box_is_not_passed_by_an_unrelated_motion():
    scene = close_box_scene()
    task = bind_task(scene, get_task("close-box-flaps"))
    idle = compile_skills(scene, [{"kind": "move_to", "target": {"point": [-.2, -.2, .3]}}, {"kind": "grasp"}])
    result = measure_task({**record(scene, idle), "task": task}, JUDGE)
    statuses = {o["id"]: o["status"] for o in result["outcomes"]}
    assert statuses["flap_left_closed"] == statuses["flap_right_closed"] == "fail"
    assert result["status"] == "fail"


def test_round_geoms_get_round_bounds():
    knob = {"name": "knob", "position": [0, 0, .1], "geoms": [
        {"shape": "capsule", "size": [.01, .01, .03]}, {"shape": "sphere", "size": [.02, .02, .02], "position": [.1, 0, 0]}]}
    frame = Simulator(Scene(bodies=[knob])).snapshot()
    assert frame["bodies"]["knob"]["extent"] == pytest.approx([.065, .02, .04], abs=1e-4)
    assert {g["name"]: g["type"] for g in frame["geoms"] if g["name"].startswith("geom_knob")} == {"geom_knob_0": "capsule", "geom_knob_1": "sphere"}


@franka
def test_articulated_body_is_listed_and_movable_with_its_teacher():
    from studio.scene_editing import SceneEdit, edited_program, scene_objects
    task = bind_task(close_box_scene(), get_task("close-box-flaps"))
    program = SceneProgram(name="Close box", description="Close both flaps.", scene=close_box_scene(), skills=close_box_skills())
    scene = {"spec": program.model_dump(exclude_none=True), "task": None}
    listed = {o["id"]: o for o in scene_objects(scene)["objects"]}["body:box"]
    assert listed["kind"] == "articulated" and "geom_flap_left_panel" in listed["geom_names"]
    moved, summary = edited_program(scene, SceneEdit(object_id="body:box", position=[-.03, .04, 0]))
    assert moved.scene.bodies[0].position == [-.03, .04, 0] and moved.skills
    task["requirements"] = [r for r in task["requirements"] if r["check"] == "joint_range"]
    assert compile_program(moved, task, "franka-panda-v1")[2]["status"] == "pass"


ur5e = pytest.mark.skipif(not __import__("studio.adapters.arms", fromlist=["x"]).ur5e_available(),
    reason="Run integrations/icrt/prepare_arms.py to install the UR5e and Robotiq assets")


def test_hinge_ranges_are_radians_on_every_robot():
    # MuJoCo defaults to degrees; a -2.4 rad range read as degrees would snap the flap shut.
    lid = {"name": "base", "geoms": [{"size": [.05, .05, .01]}], "children": [{"name": "lid", "position": [0, 0, .05],
        "joints": [{"name": "lid_hinge", "axis": [0, 1, 0], "range": [-2.4, 0], "initial": -2.0}],
        "geoms": [{"size": [.05, .05, .002], "position": [.05, 0, 0]}]}]}
    frame = Simulator(Scene(bodies=[lid])).snapshot()
    assert frame["joints"]["lid_hinge"]["value"] < -1.5


def test_close_box_task_requires_contact_fingers_not_a_specific_robot():
    task = get_task("close-box-flaps")
    with pytest.raises(ValueError, match="needs a robot with contact fingers"):
        bind_task(close_box_scene().model_copy(update={"embodiment": "cartesian-gripper-v1"}), task)
    assert {r["id"] for r in catalog()} >= {"cartesian-gripper-v1", "franka-panda-v1", "ur5e-2f85-v1"}


@ur5e
def test_same_close_box_skills_run_on_the_ur5e():
    task = get_task("close-box-flaps")
    program = SceneProgram(name="Close box", description="Close both flaps.", scene=close_box_scene())
    spec, artifact, verification = compile_program(program, task, "ur5e-2f85-v1")
    first, last = artifact["frames"][0], artifact["frames"][-1]
    assert abs(first["joints"]["flap_left_hinge"]["value"]) > 1.5 and abs(first["joints"]["flap_right_hinge"]["value"]) > 1.5
    assert verification["status"] == "pass", verification["outcomes"]
    assert last["embodiment"] == "ur5e-2f85-v1" and spec["scene"]["embodiment"] == "ur5e-2f85-v1"
    assert {"wrist", "overview"} <= {artifact["xml"].split('camera name="')[i].split('"')[0] for i in range(1, artifact["xml"].count('camera name="') + 1)}
    assert all(g["name"] for g in last["geoms"]) and {"capsule", "cylinder"} <= {g["type"] for g in last["geoms"]}
    scene = close_box_scene().model_copy(update={"embodiment": "ur5e-2f85-v1"})
    idle = compile_skills(scene, [{"kind": "move_to", "target": {"point": [-.2, -.2, .3]}}, {"kind": "grasp"}])
    assert measure_task({**record(scene, idle), "task": bind_task(scene, task)}, JUDGE)["status"] == "fail"


@ur5e
def test_ur5e_scene_xml_exports_with_its_menagerie_assets_only():
    from studio.scene_packages import _portable_xml
    artifact = record(Scene(embodiment="ur5e-2f85-v1"), [{"x": -.22, "y": -.08, "z": .28, "closed": False, "seconds": .2}] * 2)
    xml, files = _portable_xml(artifact["xml"], "ur5e-2f85-v1")
    assert b'file="/' not in xml and "robot/SOURCE.txt" in files
    assert {"robot/universal_robots_ur5e/LICENSE", "robot/robotiq_2f85/LICENSE"} <= set(files)
    assert all(name.startswith("robot/") for name in files)
    import mujoco, tempfile, pathlib
    with tempfile.TemporaryDirectory() as folder:
        for name, data in files.items():
            (pathlib.Path(folder) / name).parent.mkdir(parents=True, exist_ok=True)
            (pathlib.Path(folder) / name).write_bytes(data)
        (pathlib.Path(folder) / "scene.xml").write_bytes(xml)
        assert mujoco.MjModel.from_xml_path(str(pathlib.Path(folder) / "scene.xml")).nu == 7


def test_scenes_can_leave_out_the_legacy_cube_and_tray():
    bare = Scene(include_cube=False, include_tray=False, objects=[{"id": "blue", "position": [-.2, .1, .026]}])
    frame = Simulator(bare).snapshot()
    assert not {"cube", "cube_speed", "cube_extent"} & set(frame) and frame["attached"] is False
    assert not any(g["name"] in {"block", "tray_base"} for g in frame["geoms"])
    # With no human task there is nothing to pass.
    program = SceneProgram(name="Bare", description="No task.", scene=bare, waypoints=[
        {"x": -.2, "y": .1, "z": .2, "closed": False, "seconds": .5}, {"x": -.2, "y": .1, "z": .2, "closed": False, "seconds": .5}])
    verification = compile_program(program)[2]
    assert verification["status"] == "unknown" and verification["needs_review"]
    # Grasp welds shift without the cube; the remaining object is still picked up.
    task = {**TaskCreate(name="Move blue", description="Move the blue block.", requirements=[
        {"id": "moved", "label": "Moved", "definition": "Blue in region.", "check": "inside_region", "object_id": "blue", "bounds": [.0, .2, .0, .25, -.01, .1]}],
        operations=[{"object_id": "blue", "destination": "region", "target": [.1, .12, .045]}]).model_dump(), "schema_version": 2, "version": 1}
    spec, artifact, verification = compile_program(SceneProgram(name="Blue", description="Move blue.", scene=bare), task)
    assert verification["status"] == "pass", verification["outcomes"]
    with pytest.raises(ValueError, match="uses the destination tray"):
        bind_task(bare, get_task("block-to-tray"))
    with pytest.raises(ValueError, match="need a human task teacher"):
        compile_program(SceneProgram(name="Bare", description="No task.", scene=bare))


def test_articulated_presets_leave_out_the_cube_and_tray():
    for id in ["close-box-flaps", "open-drawer"]:
        scene = get_task(id)["suggested_scene"]
        assert scene["include_cube"] is False and scene["include_tray"] is False


def test_task_located_body_is_locked_like_the_tray():
    from studio.scene_editing import SceneEdit, edited_program, scene_objects
    program = SceneProgram(name="Close box", description="Close both flaps.", scene=close_box_scene(), skills=close_box_skills())
    task = bind_task(close_box_scene(), get_task("close-box-flaps"))
    scene = {"spec": program.model_dump(exclude_none=True), "task": task}
    box = {o["id"]: o for o in scene_objects(scene)["objects"]}["body:box"]
    assert box["editable"] is False and "fixes this body's region" in box["constraints"][0]
    with pytest.raises(ValueError, match="fixes this body's region"):
        edited_program(scene, SceneEdit(object_id="body:box", position=[.03, .02, 0]))
