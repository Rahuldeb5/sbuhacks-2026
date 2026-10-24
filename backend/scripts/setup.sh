#!/usr/bin/env bash
# Rebuild the backend env. Reuses the existing DA checkpoint; never downloads it if present.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$HOME/checkpoints/depth_anything_v2}
DA_COMMIT=a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
F=depth_anything_v2_metric_hypersim_vitl.pth
uv venv .venv --python /usr/bin/python3.14
# torch FIRST and alone from cu132 (sm_120); everything else constrained so nothing swaps torch.
uv pip install --python .venv/bin/python torch==2.14.1+cu132 torchvision==0.29.1+cu132 \
  --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv/bin/python -c constraints.txt -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
./scripts/fix_open3d_libusb.sh
[ -d third-party/Depth-Anything-V2 ] || git clone https://github.com/DepthAnything/Depth-Anything-V2 third-party/Depth-Anything-V2
git -C third-party/Depth-Anything-V2 checkout -q $DA_COMMIT
[ -f "$CHECKPOINT_DIR/$F" ] || { echo "missing $CHECKPOINT_DIR/$F"; exit 1; }
mkdir -p checkpoints && ln -sfn "$CHECKPOINT_DIR/$F" checkpoints/$F
[ -f ~/.rembg/models/u2net/u2net.onnx ] || .venv/bin/python -c "import rembg; rembg.new_session('u2net')"
.venv/bin/python -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_capability())"
