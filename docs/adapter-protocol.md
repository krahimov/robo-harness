# Adapter contracts

## Robotics model

`STUDIO_ROBOT_ENDPOINT` points to a server that implements the following application-specific protocol. It is **not** a claimed GEN-1.5, S1, GR00T, or OpenPI API. A vendor-specific bridge must implement this contract and reject incompatible demonstrations.

### Create an episode session

```http
POST /sessions
Content-Type: application/json
Authorization: Bearer <STUDIO_ROBOT_TOKEN, if set>
```

```json
{
  "protocol": "robot-harness-v1",
  "task": {"id": "block-to-tray", "version": 1},
  "instructions": "Place the block into the tray.",
  "demonstrations": [{"id": "record-id", "frames": [], "actions": [], "frequency": 20}],
  "observation": {"t": 0, "ee": [-0.22, -0.08, 0.28], "cube": [-0.22, -0.08, 0.025]}
}
```

The actual task and frames include all fields from the saved artifacts. Each frame contains joint positions, geometry transforms, gripper state, object speed, and bounding-box extents. The model session is specific to one episode. The configured demonstrations must remain available to inference for its lifetime.

Return an alphanumeric session identifier, at most 128 characters:

```json
{"session_id": "episode123"}
```

### Act

```http
POST /sessions/episode123/act
```

Request:

```json
{"observation": {"t": 0.05, "ee": [-0.22, -0.08, 0.28], "cube": [-0.22, -0.08, 0.025]}}
```

Response:

```json
{"action": {"x": -0.22, "y": -0.08, "z": 0.18, "closed": false}}
```

Actions are absolute Cartesian position targets in meters in the MuJoCo world frame. X bounds are [-0.5, 0.5], Y [-0.35, 0.35], and Z [0.025, 0.45]. They are applied for 0.05 simulated seconds. Wall-clock inference time does not advance physics. NaN, infinity, out-of-bounds coordinates, and malformed actions are rejected. This is synchronous simulation, not a real-time robot controller.

An episode lasts as many actions as the longest selected demonstration, capped at 1,200 actions. There is no model-initiated termination in V1.

### Close

```http
DELETE /sessions/episode123
```

The runner attempts to close the session after success, error, or cancellation.

## Demonstration import

The library accepts JSON exported by `/api/demos/{id}/export`:

```json
{
  "name": "My demonstration",
  "scene": {"seed": 42, "variation": 0, "friction": 0.8, "cube_x": -0.22, "cube_y": -0.08},
  "actions": [{"x": -0.22, "y": -0.08, "z": 0.2, "closed": false}]
}
```

Each action is one 20 Hz control tick. Import re-simulates these actions, rather than trusting user-supplied outcome labels or fabricated state traces. Video-only demonstrations and other embodiments need separate adapters.

## Judge provider

`STUDIO_JUDGE_ENDPOINT` is the full URL accepting a chat-completions-style POST. It must support `model`, `messages`, and function `tools`; `image_url` content is required for the replay tool. The interface lets users configure model identifiers, rubric instructions, permitted tools, and human review rules. The endpoint and credential are configured on the server.

Evidence tools:

- `inspect_state(start, end)` samples up to 40 recorded states across the selected time range. It returns timestamps, positions, bounds, speed, and attachment state.
- `replay_frames(start, end)` renders up to four MuJoCo camera frames from the selected time range. The provider receives these as JPEG image messages with timestamps.

The rubric is passed as agent instructions. The task requirements are separate data. Evidence tools cannot edit either, issue robot actions, or change stored demonstrations.

The final response must be JSON:

```json
{
  "outcomes": [
    {"id": "inside_tray", "status": "pass", "explanation": "The block fits inside the tray.", "timestamps": [9.0]},
    {"id": "released", "status": "pass", "explanation": "The grasp is inactive.", "timestamps": [9.0]},
    {"id": "settled", "status": "unknown", "explanation": "More evidence is needed for the final half-second.", "timestamps": []},
    {"id": "no_drops", "status": "pass", "explanation": "The inspected trace contains no off-table drop.", "timestamps": [0.0, 9.0]}
  ],
  "summary": "Placement appears complete; settling requires review.",
  "needs_review": true
}
```

Overall success is computed from required outcomes, not accepted from a model-provided headline. Evidence can be incomplete or sampled; the VLM judge is not ground truth. Provider errors preserve the episode and record an error evaluation. Human review can correct an evaluation without rewriting it.

## Extending toward in-context robotics models

The initial bridge is state-based. Before connecting a vision-based ICL model, add its camera observations and supported sensorimotor demonstration format, context limits, and action representation. Keep model-specific preprocessing inside its adapter. Do not silently replace in-context conditioning with fine-tuning or an LLM-generated script.
