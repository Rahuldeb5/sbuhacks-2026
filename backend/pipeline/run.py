"""Orchestration shared by the CLI (and later the API): perception -> shape -> cleanup/world frame."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from .perceive import Perception
from .shapes import METHODS
from .shapes.common import VOXEL_CM, ShapeResult
from .voxels import VoxelModel, finalize

MAX_DIM_VOXELS = 128
LEVEL_R = np.diag([1.0, -1.0, -1.0])


@contextmanager
def timed(timings: dict, key: str):
    t = time.perf_counter()
    yield
    timings[key] = round((time.perf_counter() - t) * 1000, 1)


def auto_voxel(Z_f, mask, K, voxel_cm: float = VOXEL_CM, thickness: float = 1.0) -> float:
    """Safety cap: if the object comes out huge (usually DA's metric scale, 3-11x on real photos), coarsen the voxel
    so the longest side stays <= MAX_DIM_VOXELS. A 2 mm voxel on a 60 cm object = 17M voxels"""
    vs, us = np.nonzero(mask)
    ext_px = max(np.ptp(us), np.ptp(vs)) + 1
    ext_cm = ext_px * float(np.median(Z_f[mask])) / K[0, 0] * max(1.0, thickness)
    return max(voxel_cm, ext_cm * 1.15 / MAX_DIM_VOXELS)


def run_shape(Z_f, mask, mask_full, K, rgb, method: str, thickness: float = 1.0, voxel_cm: float = VOXEL_CM,
              timings: dict | None = None, cap: bool = True, frame_R=LEVEL_R,
              extra: dict | None = None) -> tuple[ShapeResult, VoxelModel]:
    tm = timings if timings is not None else {}
    v = auto_voxel(Z_f, mask, K, voxel_cm, thickness) if cap else voxel_cm
    params = {"thickness": thickness, "mask_full": mask_full, "voxel_cm": v, **(extra or {})}
    if frame_R is not None:
        params["frame_R"] = frame_R
    with timed(tm, f"shape_{method}_ms"):
        res = METHODS[method](Z_f, mask, K, rgb, params)
    with timed(tm, "voxel_cleanup_ms"):
        model = finalize(res)
    return res, model


def shape_from_perception(p: Perception, method: str, thickness: float = 1.0, timings=None,
                          voxel_cm: float = VOXEL_CM, extra=None) -> tuple[ShapeResult, VoxelModel]:
    """Estimate scale for now (raw DA depth); step 5 adds known_dim / scale info here."""
    return run_shape(p.Z_raw, p.mask, p.mask_full, p.cam.K, p.rgb, method, thickness, voxel_cm, timings, extra=extra)
