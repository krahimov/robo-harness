"""Articulated task fixtures built from the scene graph.

Cardboard is modelled as rigid panels on damped hinges. Paper stiffness, creases,
and flap friction are simulation assumptions, not measured material properties.
"""
from pathlib import Path

from studio.schemas import Scene

CARDBOARD = [.72, .56, .36, 1]
FLAP = [.80, .64, .42, 1]
# Box footprint and hinge height, metres. Flaps start folded outward past vertical.
BOX_CENTER = (0., .02)
HALF_X, HALF_Y, HEIGHT, WALL = .08, .06, .10, .004
FLAP_OPEN = 2.1
CLOSED_TOLERANCE = .2


def _flap(side):
    """Hinged panel covering half of the box top; +1 is the far (+x) side."""
    sign = 1 if side == "right" else -1
    hinge = [sign * HALF_X, 0, HEIGHT]
    # Closed, the panel extends from its hinge toward the box center. It opens by
    # swinging upward and outward about the Y axis, which needs the opposite sign per side.
    return {"name": f"flap_{side}", "position": hinge, "joints": [{
            "name": f"flap_{side}_hinge", "type": "hinge", "axis": [0, 1, 0],
            "range": [0, 2.4] if sign > 0 else [-2.4, 0], "initial": sign * FLAP_OPEN,
            "damping": .02, "armature": .0005}],
        "geoms": [{"name": "panel", "size": [HALF_X / 2 - .001, HALF_Y - .002, .002],
            "position": [-sign * (HALF_X / 2), 0, .002], "color": FLAP, "mass": .015}],
        # Push point on the panel's outer face, close enough that closed fingertips make contact.
        "sites": [{"name": f"flap_{side}_push", "position": [-sign * .06, 0, .008]}]}


def close_box_scene():
    walls = [
        {"name": "floor", "size": [HALF_X, HALF_Y, WALL / 2], "position": [0, 0, WALL / 2]},
        {"name": "wall_left", "size": [WALL / 2, HALF_Y, HEIGHT / 2], "position": [-HALF_X, 0, HEIGHT / 2]},
        {"name": "wall_right", "size": [WALL / 2, HALF_Y, HEIGHT / 2], "position": [HALF_X, 0, HEIGHT / 2]},
        {"name": "wall_front", "size": [HALF_X, WALL / 2, HEIGHT / 2], "position": [0, -HALF_Y, HEIGHT / 2]},
        {"name": "wall_back", "size": [HALF_X, WALL / 2, HEIGHT / 2], "position": [0, HALF_Y, HEIGHT / 2]},
    ]
    box = {"name": "box", "position": [*BOX_CENTER, 0], "geoms": [{**wall, "color": CARDBOARD} for wall in walls],
        "children": [_flap("left"), _flap("right")]}
    return Scene(embodiment="franka-panda-v1", finger_contact_dim=4, include_cube=False, include_tray=False,
        table_color=[.62, .64, .62, 1], bodies=[box])


def close_box_requirements():
    def closed(side):
        low, high = (-CLOSED_TOLERANCE, .01) if side == "left" else (-.01, CLOSED_TOLERANCE)
        return {"id": f"flap_{side}_closed", "label": f"{side.title()} flap closed", "check": "joint_range",
            "joint": f"flap_{side}_hinge", "min_value": low, "max_value": high, "required": True,
            "definition": f"At episode end the {side} flap hinge is within {CLOSED_TOLERANCE} rad of flat closed."}
    return [closed("left"), closed("right"),
        {"id": "box_in_place", "label": "Box stays in place", "check": "inside_region", "object_id": "box", "required": True,
         "bounds": [round(v, 4) for v in [BOX_CENTER[0] - HALF_X - .03, BOX_CENTER[0] + HALF_X + .03, BOX_CENTER[1] - HALF_Y - .03,
            BOX_CENTER[1] + HALF_Y + .03, -.005, HEIGHT + .03]],
         "definition": "The box body's bounding box stays within 3 cm of its starting footprint at episode end."}]


def close_box_skills():
    skills = [{"kind": "grasp", "seconds": .6}]
    for side, sign in [("left", -1), ("right", 1)]:
        site = f"flap_{side}_push"
        skills += [
            # Come down outside the open flap, then approach its outer face horizontally.
            {"kind": "move_to", "target": {"site": site}, "offset": [sign * .07, 0, 0], "via_height": .3, "seconds": 3, "tool": "push"},
            {"kind": "move_to", "target": {"site": site}, "seconds": 1, "tool": "push"},
            {"kind": "follow_joint", "joint": f"flap_{side}_hinge", "site": site, "to": 0, "steps": 10, "seconds": 4},
            {"kind": "move_to", "target": {"site": site}, "offset": [0, 0, .12], "seconds": 1.2, "tool": "push"},
        ]
    return skills


CABINET_MJCF = Path(__file__).resolve().parents[2] / "examples/mjcf/cabinet.xml"
CABINET_AT = (.05, .1, 0.)
DRAWER_OPEN = .09


def open_drawer_scene():
    """Cabinet imported from MJCF; its drawer starts closed and slides out toward the robot."""
    from studio.mjcf_import import convert
    bodies = convert(CABINET_MJCF.read_text(), "cabinet", CABINET_AT)["bodies"]
    return Scene(embodiment="franka-panda-v1", include_cube=False, include_tray=False,
        table_color=[.62, .64, .62, 1], bodies=bodies)


def open_drawer_requirements():
    x, y, _ = CABINET_AT
    return [
        {"id": "drawer_open", "label": "Drawer pulled open", "check": "joint_range", "joint": "cabinet_drawer_slide",
         "min_value": DRAWER_OPEN, "max_value": .14, "required": True,
         "definition": f"At episode end the drawer slide is at least {DRAWER_OPEN} m out of its 0.14 m travel."},
        {"id": "door_closed", "label": "Upper door left closed", "check": "joint_range", "joint": "cabinet_door_hinge",
         "min_value": -.15, "max_value": .01, "required": True,
         "definition": "The upper door hinge stays within 0.15 rad of closed at episode end."},
        {"id": "cabinet_in_place", "label": "Cabinet stays in place", "check": "inside_region", "object_id": "cabinet", "required": True,
         "bounds": [round(v, 4) for v in [x - .13, x + .13, y - .15, y + .15, -.005, .19]],
         "definition": "The cabinet carcass stays within 3 cm of its starting footprint at episode end."}]


def open_drawer_skills():
    site = "cabinet_drawer_pull"
    return [{"kind": "grasp", "seconds": .6},
        # Lower closed fingers into the gap behind the handle bar, then pull along the slide.
        {"kind": "move_to", "target": {"site": site}, "offset": [0, 0, .12], "via_height": .3, "seconds": 3, "tool": "push"},
        {"kind": "move_to", "target": {"site": site}, "seconds": 1.5, "tool": "push"},
        {"kind": "follow_joint", "joint": "cabinet_drawer_slide", "site": site, "to": .12, "steps": 8, "seconds": 4},
        {"kind": "move_to", "target": {"site": site}, "offset": [0, 0, .12], "seconds": 1.2, "tool": "push"}]


# The towel lies on a 2 cm folding board so every robot's fingertips reach its corners above z = 0.025 m.
BOARD_TOP = .02
TOWEL = {"name": "towel", "position": [-.1, 0, BOARD_TOP + .004], "count": [9, 9], "spacing": .02, "mass": .03}
FOLD_CORNER, FOLD_ONTO = 0, 8  # Two corners of the same 16 cm edge (vertex = row * 9 + column).


def fold_towel_scene():
    board = {"name": "folding_board", "position": [-.1, 0, BOARD_TOP / 2], "size": [.13, .13, BOARD_TOP / 2], "color": [.78, .74, .66, 1]}
    return Scene(embodiment="franka-panda-v1", include_cube=False, include_tray=False,
        table_color=[.62, .64, .62, 1], props=[board], cloths=[TOWEL])


def fold_towel_requirements():
    return [
        {"id": "corner_folded", "label": "Corner folded onto its partner", "check": "cloth_points_near", "object_id": "towel",
         "vertex_a": FOLD_CORNER, "vertex_b": FOLD_ONTO, "max_distance": .05, "required": True,
         "definition": "At episode end the towel's lifted corner lies within 5 cm of the opposite corner of the same edge."},
        {"id": "towel_released", "label": "Towel released", "check": "released", "object_id": "towel", "required": True,
         "definition": "At episode end the gripper no longer pinches the towel."},
        {"id": "towel_on_table", "label": "Towel stays on the table", "check": "inside_region", "object_id": "towel", "required": True,
         "bounds": [-.35, .15, -.2, .2, -.01, .1], "definition": "The whole towel lies within the work area, below 10 cm, at episode end."}]


def fold_towel_skills():
    corner = {"cloth": "towel", "vertex": FOLD_CORNER}
    onto = {"cloth": "towel", "vertex": FOLD_ONTO}
    return [
        {"kind": "move_to", "target": corner, "offset": [0, 0, .08], "via_height": .2, "seconds": 2.5, "tool": "pinch"},
        {"kind": "move_to", "target": corner, "offset": [0, 0, .008], "seconds": 1.2, "tool": "pinch"},
        {"kind": "grasp", "seconds": .5},
        {"kind": "move_to", "target": corner, "offset": [0, 0, .1], "seconds": 1.5, "tool": "pinch"},
        # Carry the corner over its partner along an arc, then lay it down before letting go.
        {"kind": "move_to", "target": onto, "offset": [0, -.04, .1], "seconds": 1.8, "tool": "pinch"},
        # Overshoot slightly past the partner corner: the trailing cloth drags the corner back on release.
        # Let go 3 cm above the board: lowering further presses the cloth between fingers and board.
        {"kind": "move_to", "target": onto, "offset": [0, .015, .03], "seconds": 1.5, "tool": "pinch"},
        {"kind": "release", "seconds": .6},
        {"kind": "move_to", "target": onto, "offset": [0, 0, .12], "seconds": 1.2, "tool": "pinch"}]
