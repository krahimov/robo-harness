import gzip
import json
import struct

from fastapi import FastAPI
from fastapi.testclient import TestClient
import mujoco
import numpy as np
import pytest

from studio import robot_visuals
from studio.adapters.franka import ASSET_COMMIT, FrankaSimulator, asset_folder, available
from studio.schemas import Scene


def unpack_glb(payload):
    magic, version, length = struct.unpack_from("<4sII", payload)
    assert (magic, version, length) == (b"glTF", 2, len(payload))
    size, kind = struct.unpack_from("<I4s", payload, 12)
    assert kind == b"JSON"
    document = json.loads(payload[20:20 + size])
    return document, payload[28 + size:]


@pytest.mark.skipif(not available(), reason="Install the licensed Menagerie Panda assets")
def test_original_cad_follows_recorded_collision_frames_with_different_scene_offsets():
    canonical = mujoco.MjModel.from_xml_path(str(asset_folder() / "panda.xml"))
    license_text = (asset_folder() / "LICENSE").read_text()
    document, binary = unpack_glb(robot_visuals.visual_glb(canonical, license_text))
    metadata = document["scenes"][0]["extras"]
    assert metadata["robot_asset_commit"] == ASSET_COMMIT
    assert metadata["license_text"] == license_text
    assert len(document["nodes"]) == 58
    for extra_props in [[], [{"name": "offset_test", "position": [.4, .3, .1], "size": [.02, .02, .1]}]]:
        sim = FrankaSimulator(Scene(embodiment="franka-panda-v1", props=extra_props))
        for angle in [0, .13]:
            sim.data.qpos[sim.arm_qpos[0]] += angle
            sim.data.joint("finger_joint1").qpos[0] = .01 + angle / 10
            mujoco.mj_forward(sim.model, sim.data)
            frame = sim.snapshot()
            anchors = sorted((g for g in frame["geoms"] if g["name"].startswith("panda_geom_")), key=lambda g: int(g["name"][11:]))
            first = int(anchors[0]["name"][11:])
            assert [int(g["name"][11:]) - first for g in anchors] == metadata["collision_offsets"]
            shift = first - next(i for i in range(canonical.ngeom) if canonical.geom_group[i] != 2)
            for node in document["nodes"]:
                anchor = next(g for g in anchors if g["name"] == f"panda_geom_{first + node['extras']['anchor_offset']}")
                recorded = np.eye(4)
                recorded[:3, :3] = np.array(anchor["matrix"]).reshape(3, 3)
                recorded[:3, 3] = anchor["position"]
                relative = np.array(node["matrix"]).reshape(4, 4).T
                index = node["extras"]["visual_geom_index"]
                visual = sim.data.geom(f"panda_geom_{shift + index}")
                expected = np.eye(4)
                expected[:3, :3] = visual.xmat.reshape(3, 3)
                expected[:3, 3] = visual.xpos
                assert recorded @ relative == pytest.approx(expected, abs=7e-6)
                accessor = document["accessors"][document["meshes"][node["mesh"]]["primitives"][0]["attributes"]["POSITION"]]
                view = document["bufferViews"][accessor["bufferView"]]
                vertex = np.frombuffer(binary, dtype="<f4", count=3, offset=view["byteOffset"])
                mid = canonical.geom_dataid[index]
                face = canonical.mesh_face[canonical.mesh_faceadr[mid]][0]
                assert vertex == pytest.approx(canonical.mesh_vert[canonical.mesh_vertadr[mid] + face])
            assert all("mesh_vertices" not in g for g in anchors)


def test_binary_visual_endpoint_caches_and_honors_etag(tmp_path, monkeypatch):
    (tmp_path / "panda.xml").write_text("unused")
    (tmp_path / "LICENSE").write_text("test license")
    monkeypatch.setattr(robot_visuals, "asset_folder", lambda: tmp_path)
    payload = b"binary-glb-test"
    monkeypatch.setattr(robot_visuals, "visual_bundle", lambda *args: (payload, gzip.compress(payload), "test-digest"))
    app = FastAPI()
    app.include_router(robot_visuals.router)
    with TestClient(app) as client:
        response = client.get("/api/robot-visuals/franka-panda-v1.glb")
        assert response.status_code == 200 and response.content == payload
        assert response.headers["content-type"] == "model/gltf-binary"
        assert response.headers["content-encoding"] == "gzip"
        assert response.headers["cache-control"] == "private, max-age=3600"
        assert client.get("/api/robot-visuals/franka-panda-v1.glb", headers={"If-None-Match": response.headers["etag"]}).status_code == 304


def test_missing_optional_visuals_returns_safe_unavailable_status(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_visuals, "asset_folder", lambda: tmp_path)
    app = FastAPI()
    app.include_router(robot_visuals.router)
    with TestClient(app) as client:
        response = client.get("/api/robot-visuals/franka-panda-v1.glb")
        assert response.status_code == 503
        assert str(tmp_path) not in response.text
        assert "Recorded geometry remains accessible" in response.text
