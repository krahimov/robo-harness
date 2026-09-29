"""Verify NVIDIA inference and actual headless side/wrist rendering on Linux."""
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, str(ROOT/'backend'))


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run this preflight on the Linux GPU Pod.')
    subprocess.run([str(ROOT/'integrations/icrt/.venv/bin/python'), '-c',
        'import torch; assert torch.cuda.is_available(); x=torch.ones(16,device="cuda"); print("CUDA verified",float(x.sum()),torch.cuda.get_device_name())'], check=True)
    from studio.adapters.simulation import make_simulator
    from studio.adapters.icrt_render import Cameras
    from studio.schemas import Scene
    from PIL import Image
    sim = make_simulator(Scene(embodiment='franka-panda-v1'))
    cameras = Cameras()
    try:
        result = cameras.render({'xml': sim.source, 'frames': [sim.snapshot()]})[0]
        for name in ('side', 'wrist'):
            image = Image.open(io.BytesIO(base64.b64decode(result[name])))
            if image.size != (320, 180):
                raise ValueError('Camera dimensions differ from the ICRT contract.')
        print(json.dumps({'renderer': 'egl', 'cameras': ['side', 'wrist'], 'dimensions': [320, 180], 'embodiment': 'franka-panda-v1'}))
    finally:
        cameras.close()


if __name__ == '__main__':
    main()
