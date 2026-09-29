"""Render recorded MuJoCo states in a separate process for macOS GL affinity."""
import base64
import io
import json
import sys
import mujoco
from PIL import Image


def render(request):
    model = mujoco.MjModel.from_xml_string(request["xml"])
    data = mujoco.MjData(model)
    results = []
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for frame in request["frames"]:
            data.qpos[:] = frame["qpos"]
            gap = 0.028 if frame["closed"] else 0.042
            for name, offset in (("finger_left", -gap), ("finger_right", gap)):
                id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if id >= 0:
                    model.geom_pos[id, 0] = offset
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=request.get("camera", "overview"))
            buffer = io.BytesIO()
            Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=80)
            results.append({"t": frame["t"], "image": base64.b64encode(buffer.getvalue()).decode()})
    return results


if __name__ == "__main__":
    print(json.dumps(render(json.load(sys.stdin))))
