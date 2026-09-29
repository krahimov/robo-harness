import os
import httpx
import numpy as np
from studio.schemas import Waypoint


class ReplayModel:
    """Nearest-demo trajectory replay. Deliberately not an ICL model.

    Selects one complete demonstration by initial object position, breaking ties
    by context order, then replays world-space actions without adaptation.
    """
    supports_zero_demo = False
    def __init__(self, demonstrations, observation, **kwargs):
        self.selected = min(demonstrations, key=lambda demo: np.linalg.norm(
            np.array(demo["artifact"]["frames"][0]["cube"][:2]) - np.array(observation["cube"][:2])))
        self.actions = self.selected["artifact"]["actions"]
        self.index = 0

    def act(self, observation):
        action = self.actions[min(self.index, len(self.actions)-1)]
        self.index += 1
        return action

    def close(self):
        pass


class HTTPModel:
    """Explicit local adapter protocol; does not impersonate a vendor API."""
    supports_zero_demo = True
    def __init__(self, demonstrations, observation, task, instructions, **kwargs):
        endpoint = os.environ.get("STUDIO_ROBOT_ENDPOINT", "").rstrip("/")
        if not endpoint:
            raise ValueError("Set STUDIO_ROBOT_ENDPOINT to use the HTTP robotics adapter")
        token = os.environ.get("STUDIO_ROBOT_TOKEN")
        self.client = httpx.Client(base_url=endpoint, timeout=45, headers={"Authorization": f"Bearer {token}"} if token else {})
        self.session_id = None
        try:
            response = self.client.post("/sessions", json={"protocol": "robot-harness-v1", "task": task, "instructions": instructions,
                "demonstrations": [{"id": d["id"], "frames": d["artifact"]["frames"], "actions": d["artifact"]["actions"], "frequency": d["artifact"]["frequency"]} for d in demonstrations], "observation": observation})
            response.raise_for_status()
            self.session_id = response.json()["session_id"]
            if not isinstance(self.session_id, str) or not self.session_id.isalnum() or len(self.session_id) > 128:
                raise ValueError("Invalid model session ID")
        except Exception:
            self.client.close()

            raise

    def act(self, observation):
        response = self.client.post(f"/sessions/{self.session_id}/act", json={"observation": observation})
        response.raise_for_status()
        action = response.json()["action"]
        # Reject malformed or unsafe values at the boundary before simulation.
        validated = Waypoint(**action).model_dump()
        return {key: validated[key] for key in ["x", "y", "z", "closed"]}

    def close(self):
        try:
            if self.session_id:
                self.client.delete(f"/sessions/{self.session_id}")
        except httpx.HTTPError:
            # Remote cleanup failure must not discard a completed local rollout.
            pass
        finally:
            self.client.close()


def build_model(adapter, demonstrations, observation, task, instructions, simulator, horizon_steps=None):
    if adapter == "replay":
        implementation = ReplayModel
    elif adapter == "http":
        implementation = HTTPModel
    elif adapter == "icrt":
        from studio.adapters.icrt import ICRTModel
        implementation = ICRTModel
    else:
        raise ValueError("Unknown robotics model adapter")
    return implementation(demonstrations=demonstrations, observation=observation, task=task,
        instructions=instructions, simulator=simulator, horizon_steps=horizon_steps)


def adapter_statuses():
    from studio.adapters.icrt import status
    from studio.author_providers import codex_status
    return [{"id": "astra", "name": "GPT-6 Astra · motion programs", "type": "model", "ready": codex_status()["configured"],
             "supports_zero_demo": True, "supports_comparisons": False, "setup": "Sign in with codex login on the backend machine",
             "description": "State-assisted Franka motion programs with bounded development revisions and frozen holdouts. Uses initial RGB and sampled demonstration states/actions through your Codex subscription."},{"id": "replay", "name": "Nearest-demonstration replay", "type": "model", "ready": True,
             "supports_zero_demo": False, "description": "Replays a selected trajectory without adaptation; not an ICL model."},
        {"id": "http", "name": "Robotics model endpoint", "type": "model", "ready": bool(os.environ.get("STUDIO_ROBOT_ENDPOINT")),
             "supports_zero_demo": True, "description": "External robot-harness-v1 state/action endpoint; vendor adapter required."}, status()]
