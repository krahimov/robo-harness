"""Declarative Blender operations, shared by both Astra author connections."""
from typing import Annotated, Literal

from pydantic import Field, model_validator

from studio.authoring import SceneProgram
from studio.schemas import StrictModel, Color


class BlenderPart(StrictModel):
    shape: Literal["box", "cylinder"] = "box"
    position: list[Annotated[float, Field(ge=-.9, le=.9)]] = Field(min_length=3, max_length=3)
    size: list[Annotated[float, Field(ge=.005, le=.35)]] = Field(min_length=3, max_length=3)
    euler_xyz: list[Annotated[float, Field(ge=-3.142, le=3.142)]] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    bevel: float = Field(default=.002, ge=0, le=.01)
    collision: bool = True


class BlenderAssembly(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,27}$")
    color: Color = Field(default_factory=lambda: [.08, .25, .45, 1])
    metallic: float = Field(default=.3, ge=0, le=1)
    roughness: float = Field(default=.4, ge=.1, le=1)
    parts: list[BlenderPart] = Field(min_length=1, max_length=16)


class BlenderProgram(SceneProgram):
    assemblies: list[BlenderAssembly] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def scene_budget(self):
        names = [a.name for a in self.assemblies]
        if len(names) != len(set(names)):
            raise ValueError("Blender assembly names must be unique")
        collisions = sum(p.collision for a in self.assemblies for p in a.parts)
        if len(self.scene.props) + len(self.assemblies) + collisions > 24:
            raise ValueError("Use at most 24 combined visual assemblies, collision parts, and existing props")
        if len(self.assemblies) + sum(p.shape == "mesh" for p in self.scene.props) > 4:
            raise ValueError("Use at most four generated and imported visual meshes combined")
        if any(p.position[2] < 0 for a in self.assemblies for p in a.parts):
            raise ValueError("Blender parts must be at or above the tabletop")
        return self


BLENDER_RULES = """
You are authoring geometry through Blender's Python runtime. Return a Blender
program with assemblies as well as the existing scene and optional waypoints.
The host implements your declared operations in bpy. Do not return Python code.
Each assembly joins up to 16 beveled boxes or cylinders into one visual mesh.
Use separate assemblies for different materials. Positions are WORLD metres,
Z up. Size is XYZ HALF extents; for cylinders use X for radius and Z for half
length. euler_xyz is Blender's XYZ rotation in radians. Each collidable part
exports a separate conservative axis-aligned BOX collider. Keep each assembly
within .8 metres on every axis and within the tabletop environment. Maximum
24 total scene.props, visual assemblies, and collidable parts, four meshes total.
Use collision=false only for cosmetic detail outside the manipulation path.
Open bins need separate bottom and wall parts so their openings remain empty.
The existing scene.conveyor creates the physical belt; generated rails/rollers
must not obstruct or duplicate its top. Put cosmetic rollers below its surface.
The host adds the selected real robot asset, table, tray and movable boxes to
Blender's layout preview from the actual MuJoCo initial state. Do not model a
substitute robot, table, task box, or destination tray in assemblies.
Blender creates static machinery assets. MuJoCo executes the robot and all task
checks. A Blender preview is a static layout render, not evidence of grasping.
Prefer the supplied task teacher with waypoints=null for a contact-based pickup.
After a build, inspect both the Blender layout image and the MuJoCo rollout
evidence. Preserve human requirements and declared robot. Explain scale and
collision approximations. The photo cannot establish mass, friction, or hidden
geometry. No gradient updates or robot learning occur during authoring.
"""
