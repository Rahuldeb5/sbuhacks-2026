from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import depth, segment
from .intrinsics import MAX_LONG_SIDE, Camera, load_image

log = logging.getLogger("pipeline.perceive")

CACHE_DIR = depth.BACKEND / "cache"
ERODE_PX = 2


@dataclass
class Perception:
    rgb: np.ndarray       
    cam: Camera
    Z_raw: np.ndarray     
    mask: np.ndarray      
    mask_full: np.ndarray
    key: str
    cached: bool
    timings: dict = field(default_factory=dict)


def cache_key(data: bytes, erode_px: int = ERODE_PX) -> str:
    tag = f"{MAX_LONG_SIDE}|{depth.PRECISION}|{depth.INPUT_SIZE}|{segment.REMBG_MODEL}|{erode_px}"
    return hashlib.sha1(data + tag.encode()).hexdigest()[:12]


def _camera_json(cam: Camera) -> dict:
    return {"K": cam.K.tolist(), "width": cam.width, "height": cam.height, "hfov_deg": cam.hfov_deg,
            "source": cam.source, "f35": cam.f35}


def _load(d: Path, key: str) -> Perception:
    c = json.loads((d / "camera.json").read_text())
    cam = Camera(np.array(c["K"]), c["width"], c["height"], c["hfov_deg"], c["source"], c["f35"])
    rgb = cv2.cvtColor(cv2.imread(str(d / "photo.png")), cv2.COLOR_BGR2RGB)
    m = np.load(d / "masks.npz")
    return Perception(rgb, cam, np.load(d / "depth_raw.npy"), m["mask"], m["mask_full"], key, True)


def _save(d: Path, p: Perception):
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "photo.png"), cv2.cvtColor(p.rgb, cv2.COLOR_RGB2BGR))
    np.save(d / "depth_raw.npy", p.Z_raw)
    np.savez_compressed(d / "masks.npz", mask=p.mask, mask_full=p.mask_full)
    (d / "camera.json").write_text(json.dumps(_camera_json(p.cam), indent=1))


def load_cached(data: bytes, cache_dir: Path = CACHE_DIR, erode_px: int = ERODE_PX) -> Perception | None:
    """The cached perception for these image bytes, or None (for callers that run the stages themselves)."""
    key = cache_key(data, erode_px)
    d = Path(cache_dir) / key
    return _load(d, key) if (d / "camera.json").exists() else None


def store(p: Perception, cache_dir: Path = CACHE_DIR):
    _save(Path(cache_dir) / p.key, p)


def perceive(image: str | Path | bytes, cache_dir: Path | None = CACHE_DIR, erode_px: int = ERODE_PX) -> Perception:
    """Photo -> rgb, camera, raw DA depth, masks. cache_dir=None disables the cache."""
    t0 = time.perf_counter()
    data = image if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    key = cache_key(data, erode_px)
    d = Path(cache_dir) / key if cache_dir is not None else None
    if d is not None and (d / "camera.json").exists():
        p = _load(d, key)
        p.timings["cache_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return p

    tm: dict = {}
    t = time.perf_counter()
    rgb, cam = load_image(data)
    tm["load_ms"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter()
    depth.get_model()
    tm["depth_model_load_ms"] = round((time.perf_counter() - t) * 1000, 1)

    def timed(fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        return r, round((time.perf_counter() - t) * 1000, 1)

    with ThreadPoolExecutor(2) as pool:
        fd = pool.submit(timed, depth.infer, rgb)
        fs = pool.submit(timed, segment.segment, rgb, erode_px)
        Z_raw, tm["depth_ms"] = fd.result()
        (mask, mask_full), tm["segment_ms"] = fs.result()
    tm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    p = Perception(rgb, cam, Z_raw, mask, mask_full, key, False, tm)
    if d is not None:
        _save(d, p)
    log.info("perceive %s: %s", key, tm)
    return p
