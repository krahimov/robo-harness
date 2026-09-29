"""Persistent main-thread MuJoCo camera renderer for the ICRT adapter."""
import base64
import io
import json
import sys

import mujoco
from PIL import Image


class Cameras:
    def __init__(self):
        self.source = None
        self.renderer = None

    def render(self, request):
        if request["xml"] != self.source:
            self.close()
            self.source = request["xml"]
            self.model = mujoco.MjModel.from_xml_string(self.source)
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model, height=180, width=320)
        results = []
        for frame in request["frames"]:
            self.data.qpos[:] = frame["qpos"]
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_left") >= 0:
                gap = 0.028 if frame["closed"] else 0.042
                self.model.geom("finger_left").pos[0] = -gap
                self.model.geom("finger_right").pos[0] = gap
            mujoco.mj_forward(self.model, self.data)
            cameras = {}
            for key, camera in [("side", "overview"), ("wrist", "wrist")]:
                self.renderer.update_scene(self.data, camera=camera)
                buffer = io.BytesIO()
                Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=82)
                cameras[key] = base64.b64encode(buffer.getvalue()).decode()
            results.append(cameras)
        return results

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    cameras = Cameras()
    try:
        for line in sys.stdin:
            try:
                print(json.dumps({"frames": cameras.render(json.loads(line))}), flush=True)
            except Exception:
                print(json.dumps({"error": "ICRT camera rendering failed"}), flush=True)
    finally:
        cameras.close()
