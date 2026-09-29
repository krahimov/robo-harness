"""Contact-manipulation fixtures with explicit physical approximations."""
import math
from studio.schemas import Scene

POUCH_BOUNDS = [.005, .195, .04, .16, .012, .11]


def pen_pouch_scene():
    cloth = [.20, .24, .42, 1]
    rim = [.42, .49, .68, 1]
    props = []
    def box(name, position, size, color=cloth, collidable=True):
        props.append(dict(name=name, position=position, size=size, color=color, collidable=collidable))
    # Keep the supporting surface 2 mm inside the fixed outcome region so
    # finite contact penetration cannot make a resting pen fail containment.
    box("pouch_floor", [.1, .1, .007], [.105, .07, .007])
    for label, y in [("front", .035), ("back", .165)]:
        box("pouch_" + label, [.1, y, .056], [.105, .005, .054])
        box("pouch_rim_" + label, [.1, y, .108], [.105, .006, .005], rim)
    for label, x in [("left", 0), ("right", .2)]:
        box("pouch_" + label, [x, .1, .056], [.005, .06, .054])
    # Decorative handles stay outside the opening; the support is an assumption.
    for side, y in [("front", .025), ("back", .175)]:
        for j, x in enumerate([.065, .135]):
            box(f"handle_{side}_{j}", [x, y, .139], [.006, .005, .028], rim, False)
        box("handle_" + side + "_top", [.1, y, .167], [.041, .005, .006], rim, False)
    for name, y in [("near", -.152), ("far", -.048)]:
        box("pen_rest_" + name, [-.2, y, .016], [.035, .012, .016], [.65, .68, .65, 1])
        for side, x in [("left", -.219), ("right", -.181)]:
            box(f"pen_rest_{name}_{side}", [x, y, .03], [.005, .012, .012], [.65, .68, .65, 1])
    return Scene(embodiment="franka-panda-v1", finger_contact_dim=6, contact_solver="precision", cube_x=-.36, cube_y=.25,
        tray_x=.38, tray_y=-.22, table_color=[.65, .67, .64, 1], props=props,
        objects=[dict(id="pen", shape="cylinder", position=[-.2, -.1, .041],
            size=[.008, .008, .07], euler_xyz=[-math.pi/2, 0, 0],
            mass=.018, color=[.10, .35, .85, 1], variation=.004)])


def pen_pouch_requirements():
    def requirement(id, label, check, definition, **fields):
        return dict(id=id, label=label, check=check, definition=definition,
                    object_id="pen", required=True, **fields)
    return [
        requirement("pen_lifted", "Pen lifted clear", "lifted_once", "The pen is held by both fingers with its center above 0.14 m at least once.", min_height=.14),
        requirement("pen_inserted", "Pen inserted while held", "inside_while_held", "The complete pen lies within the pouch interior while the two-finger grasp is still active, before release.", bounds=POUCH_BOUNDS),
        requirement("pen_inside", "Pen remains inside pouch", "inside_region", "The complete pen bounding box is within the declared pouch interior at episode end.", bounds=POUCH_BOUNDS),
        requirement("pen_released", "Pen released", "released", "At episode end the pen is no longer held between the fingers."),
        requirement("pen_settled", "Pen settled", "settled", "Pen speed stays below 0.03 m/s during the final half-second."),
        requirement("pen_no_drops", "No off-table pen drops", "no_drops", "The pen center never falls below z=-0.02 m."),
    ]


def pen_pouch_waypoints(scene):
    """Scripted teacher for the fixed fixture; model execution is separate."""
    from studio.adapters.simulation import make_simulator
    if scene.embodiment != "franka-panda-v1":
        raise ValueError("Pen insertion requires the Franka embodiment")
    state = make_simulator(scene).snapshot()
    x, y, z = state["objects"]["pen"]["position"]
    points = []
    def move(position, closed, seconds, yaw):
        points.append(dict(zip(["x", "y", "z"], position)) | dict(closed=closed, seconds=seconds, euler_xyz=[math.pi, 0, yaw]))
    move([x, y, .24], False, 2, math.pi/2)
    move([x, y, max(.025, z-.006)], False, 2, math.pi/2)
    move([x, y, max(.025, z-.006)], True, 1, math.pi/2)
    move([x, y, .22], True, 2.5, math.pi/2)
    move([x, y, .22], True, 1, 0)
    move([.12, .1, .22], True, 1.5, 0)
    move([.12, .1, .08], True, 1.2, 0)
    move([.12, .1, .08], False, 1.5, 0)
    move([.12, .1, .25], False, 1, 0)
    return points
