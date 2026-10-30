"""Human-owned task ports of robomimic / RoboTurk manipulation evals.

robomimic's Lift and PickPlaceCan datasets were collected on robosuite, the
multi-human ("mh") sets through RoboTurk. These presets port each task's goal
geometry and success condition into this studio's scene language so any robot
actor (GPT-6 Astra, Claude Opus 5.5, ICRT, replay) is scored by the same fixed
numerical checks. They are robosuite-inspired, not the benchmark itself:
scores are not comparable to published robomimic success rates. Each preset
lists its deviations so reports can disclose them.
"""
from studio.schemas import Scene

SUITE = "robomimic"
REFERENCE = "https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation"

# PickPlaceCan target: one bin quadrant, modelled as a walled compartment.
CAN_BIN_CENTER = (.1, .12)
CAN_BIN_HALF = (.0975, .1225)
CAN_BIN_BOUNDS = [CAN_BIN_CENTER[0] - CAN_BIN_HALF[0], CAN_BIN_CENTER[0] + CAN_BIN_HALF[0],
    CAN_BIN_CENTER[1] - CAN_BIN_HALF[1], CAN_BIN_CENTER[1] + CAN_BIN_HALF[1], -.005, .16]


def lift_scene():
    # robosuite samples the cube within ±3 cm of the table centre.
    return Scene(embodiment="franka-panda-v1", finger_contact_dim=6, contact_solver="precision",
        cube_x=-.12, cube_y=-.1, variation=.03, tray_x=.38, tray_y=.22, cube_color=[.85, .2, .2, 1])


def lift_requirements():
    return [
        {"id": "cube_lifted", "label": "Cube lifted 2 cm clear", "check": "lifted_once", "object_id": "cube", "required": True,
         "min_height": .045, "definition": "The cube is held and its underside is at least 2 cm above the table at least once "
            "(center z ≥ 0.045 m for the 2.5 cm half-size block). Mirrors robosuite Lift: cube center above table + 0.04 m for its 2 cm half-size cube."},
        {"id": "cube_no_drops", "label": "No off-table drops", "check": "no_drops", "object_id": "cube", "required": True,
         "definition": "The cube center never falls below z=-0.02 m."},
    ]


def can_scene():
    wall = [.55, .5, .42, 1]
    props = [dict(name="bin_floor_mark", position=[*CAN_BIN_CENTER, .0005], size=[CAN_BIN_HALF[0], CAN_BIN_HALF[1], .005],
        color=[.78, .7, .52, 1], collidable=False)]
    for label, x in [("left", CAN_BIN_CENTER[0] - CAN_BIN_HALF[0] - .005), ("right", CAN_BIN_CENTER[0] + CAN_BIN_HALF[0] + .005)]:
        props.append(dict(name="bin_wall_" + label, position=[x, CAN_BIN_CENTER[1], .02], size=[.005, CAN_BIN_HALF[1] + .01, .02], color=wall))
    for label, y in [("front", CAN_BIN_CENTER[1] - CAN_BIN_HALF[1] - .005), ("back", CAN_BIN_CENTER[1] + CAN_BIN_HALF[1] + .005)]:
        props.append(dict(name="bin_wall_" + label, position=[CAN_BIN_CENTER[0], y, .02], size=[CAN_BIN_HALF[0], .005, .02], color=wall))
    return Scene(embodiment="franka-panda-v1", finger_contact_dim=6, contact_solver="precision",
        cube_x=-.36, cube_y=.25, tray_x=.38, tray_y=-.22, props=props,
        objects=[dict(id="can", shape="cylinder", position=[-.2, -.1, .06], size=[.025, .025, .06],
            mass=.08, color=[.75, .12, .12, 1], variation=.02)])


def can_requirements():
    def requirement(id, label, check, definition, **fields):
        return dict(id=id, label=label, check=check, definition=definition, object_id="can", required=True, **fields)
    return [
        requirement("can_in_bin", "Can inside target bin", "inside_region",
            "The complete can bounding box lies inside the target bin compartment at episode end. Stricter than robosuite, which tests only the can center.",
            bounds=CAN_BIN_BOUNDS),
        requirement("can_released", "Can released", "released",
            "At episode end the can is not held. robosuite requires the gripper to be away from the can."),
        requirement("can_settled", "Can at rest", "settled", "Can speed stays below 0.03 m/s during the final half-second."),
        requirement("can_no_drops", "No off-table drops", "no_drops", "The can center never falls below z=-0.02 m."),
    ]


BENCHMARKS = {
    "robomimic-lift": {
        "name": "robomimic Lift", "scene": lift_scene, "requirements": lift_requirements,
        "description": "Grasp the red cube and lift it clear of the table. Ported from robosuite Lift, the robomimic lift task.",
        "benchmark": {"suite": SUITE, "task": "lift", "robosuite_env": "Lift", "datasets": ["lift/ph", "lift/mh"],
            "reference": REFERENCE + "/lift.py", "deviations": [
                "Cube half-size 2.5 cm instead of 2 cm; the lift threshold keeps robosuite's 2 cm underside clearance.",
                "Success needs an active grasp while lifted; robosuite counts any lift.",
                "Task horizon follows this studio's motion-program limits, not robosuite's 500 steps at 20 Hz."]},
    },
    "robomimic-can": {
        "name": "robomimic Can (PickPlaceCan)", "scene": can_scene, "requirements": can_requirements,
        "description": "Pick the red can from the table, place it upright or lying inside the walled target bin, then release it. Ported from robosuite PickPlaceCan, the robomimic can task.",
        "benchmark": {"suite": SUITE, "task": "can", "robosuite_env": "PickPlaceCan", "datasets": ["can/ph", "can/mh"],
            "reference": REFERENCE + "/pick_place.py", "deviations": [
                "Only the can's target bin quadrant is modelled, as a 19.5 × 24.5 cm compartment with 4 cm walls; the source bin is open table.",
                "Can radius 2.5 cm (the scene language's cylinder limit) instead of robosuite's larger can mesh.",
                "The complete can must lie inside the bin; robosuite checks only its center.",
                "No other clutter objects (milk, bread, cereal)."]},
    },
}


def presets(make_preset):
    result = []
    for id, spec in BENCHMARKS.items():
        scene = spec["scene"]()
        result.append({**make_preset(id, spec["name"], spec["description"], spec["requirements"](), [], scene.model_dump()),
            "required_embodiment": "franka-panda-v1", "suggested_builder": "mujoco", "benchmark": spec["benchmark"]})
    return result
