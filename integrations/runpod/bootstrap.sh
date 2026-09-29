#!/usr/bin/env bash
# Run inside an official Runpod PyTorch Pod, with this repository under /workspace.
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$(uname -s)" = Linux || { echo 'This bootstrap requires Linux.' >&2; exit 1; }
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
apt-get update
apt-get install -y --no-install-recommends git libegl1 libgl1 libglib2.0-0 libopengl0 libgles2 libglvnd0
command -v uv >/dev/null || python3 -m pip install 'uv==0.8.3'
uv sync --project backend --frozen --python 3.12
uv sync --project integrations/icrt --frozen --python 3.12
integrations/icrt/.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check Pod drivers and PyTorch"; print(torch.cuda.get_device_name(), torch.version.cuda)'
integrations/icrt/.venv/bin/python integrations/icrt/prepare.py
integrations/icrt/.venv/bin/python integrations/icrt/prepare_franka.py
backend/.venv/bin/python integrations/runpod/preflight.py
echo 'Dependencies and pinned assets ready. Next: seed the trial, then start serve.py.'
