"""Local HTTP worker for pretrained ICRT. No training or replay fallback."""
import argparse
import base64
import contextlib
import hmac
import io
import math
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime import ICRTRuntime


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Observation(Strict):
    side: str = Field(min_length=4, max_length=120000)
    wrist: str = Field(min_length=4, max_length=120000)
    proprio: list[float] = Field(min_length=7, max_length=7)

    @field_validator("proprio")
    @classmethod
    def bounded_pose(cls, value):
        if any(abs(x) > 10 for x in value[:3]) or any(abs(x) > 2 * math.pi for x in value[3:6]) or not 0 <= value[6] <= 1:
            raise ValueError("Invalid position, Euler rotation, or gripper value")
        return value


class Demo(Strict):
    id: str = Field(max_length=128)
    frames: list[Observation] = Field(min_length=1, max_length=256)
    actions: list[list[float]] = Field(min_length=1, max_length=256)

    @field_validator("actions")
    @classmethod
    def bounded_actions(cls, values):
        for row in values:
            if len(row) != 7 or not all(math.isfinite(x) for x in row):
                raise ValueError("Actions need seven finite values")
            Observation.bounded_pose(row)
        return values


class SessionRequest(Strict):
    protocol: str
    demonstrations: list[Demo] = Field(max_length=4)


class ActionRequest(Strict):
    observation: Observation
    previous_action: list[float] | None = Field(default=None, min_length=7, max_length=7)

    @field_validator("previous_action")
    @classmethod
    def bounded_previous_action(cls, value):
        return Observation.bounded_pose(value) if value is not None else None


def decode_observation(frame):
    if isinstance(frame, Observation):
        frame = frame.model_dump()
    result = {"proprio": frame["proprio"]}
    for camera in ("side", "wrist"):
        try:
            raw = base64.b64decode(frame[camera], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width > 1024 or image.height > 1024 or image.format not in {"JPEG", "PNG"}:
                    raise ValueError()
                result[camera] = image.convert("RGB")
        except Exception:
            raise ValueError("Provide valid bounded JPEG or PNG camera frames") from None
    return result


def create_app(runtime_factory=ICRTRuntime):
    guard = threading.Lock()
    session = {"id": None, "accessed": 0.0}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime_factory()
        yield

    app = FastAPI(title="Robot Harness ICRT worker", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request, call_next):
        token = os.environ.get("STUDIO_ICRT_TOKEN", "")
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            return JSONResponse({"detail": "ICRT worker authentication required"}, status_code=401)
        if int(request.headers.get("content-length", "0") or "0") > 32 * 1024 * 1024:
            return JSONResponse({"detail": "ICRT request too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"detail": "Invalid ICRT observation or demonstration payload"}, status_code=422)

    @app.get("/health")
    def health():
        return {"ready": True, "protocol": "robot-harness-icrt-v1", "supports_zero_demo": True,
            "cameras": ["side", "wrist"], "pose": "absolute xyz, intrinsic Euler XYZ radians, gripper 0=open 1=closed",
            **app.state.runtime.metadata()}

    @app.post("/sessions")
    def start(body: SessionRequest):
        if body.protocol != "robot-harness-icrt-v1":
            raise HTTPException(422, "Unsupported ICRT protocol")
        with guard:
            if session["id"] and time.monotonic() - session["accessed"] < 600:
                raise HTTPException(409, "ICRT has an active session; close it before starting another")
            try:
                frames_count = sum(len(demo.frames) for demo in body.demonstrations)
                if frames_count > 256:
                    raise ValueError("Too many context frames")
                demos = [{"frames": [decode_observation(frame) for frame in demo.frames], "actions": demo.actions}
                    for demo in body.demonstrations]
                app.state.runtime.reset(demos)
            except ValueError:
                raise HTTPException(422, "The ICRT context is invalid or exceeds 256 frames") from None
            session.update(id=uuid.uuid4().hex, accessed=time.monotonic())
            return {"session_id": session["id"], "metadata": app.state.runtime.metadata()}

    @app.post("/sessions/{session_id}/act")
    def act(session_id: str, body: ActionRequest):
        with guard:
            if session_id != session["id"]:
                raise HTTPException(404, "ICRT session not found")
            try:
                observation = decode_observation(body.observation)
                if body.previous_action is None:
                    action = app.state.runtime.act(observation)
                else:
                    if not app.state.runtime.metadata().get("supports_executed_action_feedback", False):
                        raise ValueError("This runtime does not accept executed-action feedback")
                    action = app.state.runtime.act(observation, previous_action=body.previous_action)
            except ValueError:
                raise HTTPException(422, "ICRT could not infer a valid action or reached its context capacity") from None
            session["accessed"] = time.monotonic()
            return {"action": action}

    @app.delete("/sessions/{session_id}")
    def close(session_id: str):
        with guard:
            if session_id != session["id"]:
                raise HTTPException(404, "ICRT session not found")
            app.state.runtime.reset([])
            session["id"] = None
        return {"closed": True}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("STUDIO_ICRT_TOKEN"):
        parser.error("Set STUDIO_ICRT_TOKEN before exposing this worker beyond localhost")
    import uvicorn
    uvicorn.run(create_app(lambda: ICRTRuntime(device=args.device)), host=args.host, port=args.port)
