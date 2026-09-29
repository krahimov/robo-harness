# Run the frozen ICRT comparison on Runpod

This first cloud milestone runs the existing model, simulator and numerical judge on one GPU Pod. It reproduces the local experiment settings, preserves the original demonstration trajectories, and downloads the resulting episodes and model traces. It does not assume the task will succeed or that CUDA will produce bitwise-identical CPU results.

The local authoring studio remains at port 5178. The isolated cloud studio is available through an SSH tunnel at **http://127.0.0.1:8789**. It has a separate database and no Codex login or OpenAI key. Author scenes locally; this first cloud path is for running and inspecting the frozen comparison. A one-click remote job launcher inside the local UI is not implemented yet.

## Pod configuration

- One on-demand RTX 4090, 24 GB VRAM, as the initial configuration to validate.
- Official Runpod PyTorch template on Linux. Bootstrap installs our locked Python environments separately.
- A 50 GB volume disk mounted at `/workspace` and at least 30 GB container disk for a disposable trial. Download evidence before terminating the Pod; its volume disk is deleted with it. Use a separate network volume only when storage must survive Pod termination. Confirm free space after installation.
- Set `NVIDIA_DRIVER_CAPABILITIES=all` before starting the Pod, so NVIDIA graphics/EGL libraries are available as well as CUDA compute.
- Enable the template's normal SSH service and a public-IP TCP mapping for port 22. Use Runpod's **SSH over exposed TCP** connection, which supports SCP and forwarding. Keep studio ports 8788 and 8790 unexposed; both bind only to loopback.

Runpod already manages the container. The initial path uses two supervised Python processes inside its official template; Docker Compose or nested Docker is not required. Before launch, review the actual GPU rate and recurring storage charge in the console. No provisioning, funding, API-key creation, or billable resources are performed by these scripts.

References: [Pods](https://docs.runpod.io/pods/overview), [SSH](https://docs.runpod.io/pods/connect-to-a-pod), [storage](https://docs.runpod.io/pods/storage/types).

## Prepare the transfer locally

From the repository root, build the frontend with `npm --prefix web run build`, then:

```sh
backend/.venv/bin/python integrations/runpod/bundle.py \
  --comparison c483d132b2a4478fafb04d3a6e345e30
```

The comparison ID above is this workspace's corrected nine-episode reference, with zero, one and two demonstrations, seeds 2000–2002, and 150 control steps per episode. Other installations must select their own completed ICRT comparison.

Output goes to `.data/runpod/upload/`:

- `studio-source.tar.gz`: application code, lockfiles and built frontend.
- `trial.json`: the frozen evaluation specification, state judge, task, selected demonstration states/actions, required mesh geometry and model provenance.
- `manifest.json`: file and fixture hashes.

The export excludes source photos, author conversations, credentials, local databases, model weights and unrelated records. It uses each experiment's frozen context rather than the current demonstration record. Original task requirements and demo order must match when restored.

## Transfer and install

Use the host, SSH port and identity from the Pod's actual connection panel. Configure a local SSH alias named `robot-sandbox` for that host if desired, then upload the three generated files with SCP. Keep SSH host-key verification enabled.

```sh
ssh robot-sandbox 'mkdir -p /workspace/robot-harness-studio'
scp .data/runpod/upload/studio-source.tar.gz .data/runpod/upload/trial.json .data/runpod/upload/manifest.json robot-sandbox:/workspace/robot-harness-studio/
```

On the Pod, extract this locally generated source archive and bootstrap:

```sh
cd /workspace/robot-harness-studio
tar --no-same-owner -xzf studio-source.tar.gz
python3 integrations/runpod/verify_bundle.py
# Optional on a fresh disposable Pod: install model packages on local disk.
# This avoids slow copies of thousands of files onto /workspace.
uv venv --python 3.12 /opt/studio-icrt-venv
ln -s /opt/studio-icrt-venv integrations/icrt/.venv
bash integrations/runpod/bootstrap.sh
```

The local-disk environment is discarded when the Pod stops; rebuild it before restarting inference. Omit those two optional commands to keep the environment under `/workspace`. Only create the symlink before the first install, when `integrations/icrt/.venv` does not exist. `--no-same-owner` prevents extraction from trying to restore the Mac's user IDs on Runpod storage.

Bootstrap fetches pinned ICRT weights/source and Panda assets with their existing integrity and revision checks. It also verifies CUDA tensor execution and actual EGL side/wrist camera rendering. It must pass before starting inference. This is not a CPU fallback path. Use only a trusted archive produced by the bundle script.

Restore the trial into a **new** directory. Existing data directories are rejected, and a failed restore cleans up only the new directory it created:

```sh
backend/.venv/bin/python integrations/runpod/seed_trial.py trial.json \
  --data-dir /workspace/robot-harness-studio/.data/cloud-trial-1
nohup backend/.venv/bin/python integrations/runpod/serve.py \
  --data-dir /workspace/robot-harness-studio/.data/cloud-trial-1 \
  > /workspace/robot-harness-studio/.data/cloud-trial-1/supervisor.log 2>&1 < /dev/null &
```

The supervisor requires CUDA and stops both services if one exits. Its logs and the trial database stay under `/workspace`. It reconstructs XML against installed robot assets while keeping the demonstration's recorded states and actions unchanged. A geometry layout, robot revision or task mismatch aborts restoration.

## Open the sandbox and run

From the Mac, keep this SSH tunnel open:

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8789:127.0.0.1:8788 robot-sandbox
```

Open http://127.0.0.1:8789 to inspect the cloud studio. Start the trial from the local repository:

```sh
backend/.venv/bin/python integrations/runpod/run_trial.py \
  --fixture .data/runpod/upload/trial.json \
  --output .data/runpod/gpu-trial-1
```

The runner verifies the target's fixture hash and model provenance before submitting. It saves the comparison ID immediately, displays progress, then downloads the comparison and every episode artifact. The final audit checks seeds, unchanged task/judge, demonstration order, model hash, adapter versions, CUDA execution, fixed weights, and exactly 150 predictions/actions per episode. Pass/fail outcomes remain separate from the infrastructure audit.

Use the same command with `--resume` to reconnect without submitting another experiment. After an uncertain submission with no `run.json`, inspect the cloud studio's comparisons before creating any new job. Ctrl+C or a deadline requests cancellation; the Pod retains completed episodes. The explicit `--allow-cpu` option is for local transport testing only and does not constitute a GPU validation result.

## Finish the session

Confirm `comparison.json`, `audit.json` and `artifacts/` downloaded locally. Then stop or terminate the GPU Pod in Runpod. **Stopping the Python processes or SSH tunnel does not stop GPU billing.** The scripts do not automatically terminate a Pod and receive no Runpod management credential. A retained network volume continues to incur storage charges independently; choose whether to keep it for the next session.

## Validation limits

Eleven code and fixture-restoration tests pass locally, including preserving trajectories, rollback, export exclusions, origin restrictions, and rejection of CPU fallback or missing predictions. The native factory fixture completed both an isolated CPU comparison and a real Runpod RTX 4090 comparison: nine episodes and 1,350 predictions in each, no provider errors, and passing contract audits. Each context condition scored zero passes out of three on both hosts. The GPU trial also verified Linux installation, CUDA model execution, actual EGL side/wrist rendering, SSH transfer, evidence download, and Pod termination. See [validation](../../docs/validation.md) for the exact configuration, deployment fixes, and limits. Cloud execution is working; learned task success remains unresolved.
