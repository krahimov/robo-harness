"""Robot embodiment registry.

Each entry names a simulator factory and the manipulation capabilities it
actually simulates. Skills, tasks, and authors check capabilities here rather
than hardcoding embodiment IDs. Registering a robot adds no hardware validity.
"""
from dataclasses import dataclass
from importlib import import_module
from typing import Literal

CONTACT_FINGERS = "contact_fingers"  # Fingers collide with the scene, so they can push and pinch.
WRIST_ROTATION = "wrist_rotation"    # End-effector orientation targets are executed.
WELD_GRASP = "weld_grasp"            # Idealized proximity attachment, no finger contact.
MOBILE_BASE = "mobile_base"          # The base drives; inverse kinematics moves base and arm together.
DEXTEROUS_HAND = "dexterous_hand"   # Multi-finger hand; the gripper command drives a grasp synergy.
CLOTH_PINCH = "idealized_cloth_pinch"  # Closing near a cloth vertex pins it to the hand; cloth itself is a MuJoCo flex.


@dataclass(frozen=True)
class RobotSpec:
    id: str
    name: str
    simulator: str
    capabilities: frozenset
    description: str
    availability: str | None = None
    # Tool points as offsets (metres) in the fingertip/grasp site frame. Missing tools sit at the site itself,
    # which is right for parallel grippers; hands pinch with their fingertips and push with the closed fist.
    tools: tuple = ()

    def tool_offset(self, name):
        return dict(self.tools).get(name, (0., 0., 0.))

    def factory(self):
        module, name = self.simulator.rsplit(":", 1)
        return getattr(import_module(module), name)

    def available(self):
        if not self.availability:
            return True
        module, name = self.availability.rsplit(":", 1)
        return bool(getattr(import_module(module), name)())


ROBOTS = {spec.id: spec for spec in [
    RobotSpec("cartesian-gripper-v1", "Cartesian gripper", "studio.adapters.simulation:Simulator",
        frozenset({WELD_GRASP, CLOTH_PINCH}),
        "Three-axis gantry with an idealized proximity weld. Its geometry has no contacts, so it cannot push, "
        "open, or close articulated parts. Not a commercial arm."),
    RobotSpec("franka-panda-v1", "Franka Panda", "studio.adapters.franka:FrankaSimulator",
        frozenset({CONTACT_FINGERS, WRIST_ROTATION, CLOTH_PINCH}),
        "Articulated MuJoCo Menagerie Panda with contact fingers and full wrist rotation. Controller and "
        "camera settings are research assumptions.", "studio.adapters.franka:available"),
    RobotSpec("ur5e-2f85-v1", "UR5e + Robotiq 2F-85", "studio.adapters.arms:UR5eSimulator",
        frozenset({CONTACT_FINGERS, WRIST_ROTATION, CLOTH_PINCH}),
        "MuJoCo Menagerie Universal Robots UR5e with a Robotiq 2F-85 adaptive gripper, attached at the flange. "
        "Contact fingers and full wrist rotation; controller gains and mounting are research assumptions.",
        "studio.adapters.arms:ur5e_available"),
    RobotSpec("ur5e-allegro-v1", "UR5e + Allegro hand", "studio.adapters.arms:UR5eAllegroSimulator",
        frozenset({CONTACT_FINGERS, WRIST_ROTATION, CLOTH_PINCH, DEXTEROUS_HAND}),
        "MuJoCo Menagerie UR5e with a 16-joint Wonik Allegro right hand. The gripper command drives a whole-hand "
        "power-grasp synergy; grasps are thumb-against-finger contact. Hand gains and mounting are research assumptions.",
        "studio.adapters.arms:allegro_available", tools=(("pinch", (0., 0., .033)), ("push", (.035, 0., .045)))),
    RobotSpec("tidybot-v1", "TidyBot mobile manipulator", "studio.adapters.arms:TidyBotSimulator",
        frozenset({CONTACT_FINGERS, WRIST_ROTATION, CLOTH_PINCH, MOBILE_BASE}),
        "MuJoCo Menagerie Stanford TidyBot: holonomic base, Kinova Gen3 arm, Robotiq 2F-85 gripper, standing on the floor "
        "beside the table. Whole-body inverse kinematics moves the base and arm together. Gains are research assumptions.",
        "studio.adapters.arms:tidybot_available"),
]}
# Pydantic schemas accept exactly the registered embodiments.
EMBODIMENTS = tuple(ROBOTS)
Embodiment = Literal[EMBODIMENTS]


def robot(id: str) -> RobotSpec:
    try:
        return ROBOTS[id]
    except KeyError:
        raise ValueError(f"Unknown robot embodiment {id}") from None


def require(id: str, capability: str, purpose: str):
    spec = robot(id)
    if capability not in spec.capabilities:
        raise ValueError(f"{spec.name} cannot {purpose}. Select a robot with {capability.replace('_', ' ')}.")
    return spec


def catalog():
    return [{"id": spec.id, "name": spec.name, "capabilities": sorted(spec.capabilities),
        "description": spec.description, "available": spec.available()} for spec in ROBOTS.values()]
