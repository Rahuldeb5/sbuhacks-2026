from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("pipeline.depth")

BACKEND = Path(__file__).resolve().parent.parent
DA_METRIC = BACKEND / "third-party" / "Depth-Anything-V2" / "metric_depth"
CKPT_NAME = "depth_anything_v2_metric_hypersim_vitl.pth"
INPUT_SIZE = int(os.environ.get("DA_INPUT_SIZE", "518"))
PRECISION = os.environ.get("DA_PRECISION", "fp16")

_model = None
_lock = threading.Lock()


def _ckpt_path() -> Path:
    env = os.environ.get("CHECKPOINT_DIR")
    if env and (Path(env) / CKPT_NAME).exists():
        return Path(env) / CKPT_NAME
    return BACKEND / "checkpoints" / CKPT_NAME


def get_model():
    global _model
    with _lock:
        if _model is None:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - refusing to run depth on CPU")
            if str(DA_METRIC) not in sys.path:
                sys.path.insert(0, str(DA_METRIC))
            from depth_anything_v2.dpt import DepthAnythingV2

            t = time.perf_counter()
            m = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20.0)
            m.load_state_dict(torch.load(_ckpt_path(), map_location="cpu", weights_only=True))
            m = m.cuda().eval()
            _model = m.half() if PRECISION == "fp16" else m
            log.info("depth model loaded in %.2fs on %s", time.perf_counter() - t, torch.cuda.get_device_name(0))
    return _model


def infer(rgb: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """RGB uint8 (H,W,3) -> planar z-depth Z_f (H,W) float32 in **cm**."""
    import torch
    m = get_model()
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with _lock:
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=PRECISION == "fp16"):
            depth_m = m.infer_image(bgr, input_size)
        depth_m = depth_m.astype(np.float32)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    log.info("depth inference %.0f ms (%dx%d, input_size=%d, %s)", dt * 1000, rgb.shape[1], rgb.shape[0], input_size, PRECISION)
    infer.last_ms = dt * 1000
    return (depth_m * 100.0).astype(np.float32)


infer.last_ms = 0.0
