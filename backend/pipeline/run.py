"""Orchestration shared by the CLI and the API: photo -> Scan (perception + scale + auto_select) -> shape model."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .intrinsics import Camera
from .perceive import CACHE_DIR, Perception, perceive
from .scale import MARKER_CM, ScaleInfo, apply_marker, known_dim_factor, world_from_cam
from .shapes import METHODS
from .shapes.auto_select import auto_select
from .shapes.common import VOXEL_CM, ShapeResult
from .voxels import VoxelModel, finalize

MAX_DIM_VOXELS = 128


@contextmanager
def timed(timings: dict, key: str):
    t = time.perf_counter()
    yield
    timings[key] = round((time.perf_counter() - t) * 1000, 1)


@dataclass
class Scan:
    rgb: np.ndarray
    cam: Camera
    Z_f: np.ndarray            # depth after scale correction (cm); == Z_raw unless marker
    mask: np.ndarray
    mask_full: np.ndarray
    auto: dict
    timings: dict = field(default_factory=dict)
    Z_raw: np.ndarray | None = None
    scale: ScaleInfo = field(default_factory=ScaleInfo)
    known: dict | None = None  # e.g. {"h": 6.0} for known_dim mode
    prior_scale: float = 1.0   # mirror plane prior multiplier from the VLM shape_category (plane_fit.CATEGORY_PRIOR)
    key: str = ""              # perception cache key

    @property
    def K(self):
        return self.cam.K

    @property
    def frame_R(self):
        return world_from_cam(self.scale.up_cam)


def apply_scale(rgb, cam: Camera, Z_raw, mask_full, mode="auto", marker_cm=MARKER_CM, known=None):
    """mode: estimate | marker | known_dim | auto (marker if found, else known_dim if given, else estimate).
    known_dim is finished later in run_shape (needs a measurement); here it only records the request."""
    if mode in ("marker", "auto"):
        r = apply_marker(rgb, Z_raw, cam.K, mask_full, marker_cm, cam.source)
        if r is not None:
            Zc, K, info = r
            return Zc, Camera(K, cam.width, cam.height, cam.hfov_deg, cam.source, cam.f35), info
        if mode == "marker":
            return Z_raw, cam, ScaleInfo("estimate", info={"marker_not_found": True})
    if mode in ("known_dim", "auto") and known:
        return Z_raw, cam, ScaleInfo("known_dim", info={"known_cm": known})
    return Z_raw, cam, ScaleInfo("estimate")


def scan_from_perception(p: Perception, scale_mode: str = "auto", marker_cm: float = MARKER_CM,
                         known: dict | None = None, timings: dict | None = None) -> Scan:
    tm = {**p.timings, **(timings or {})}
    with timed(tm, "scale_ms"):
        Z_f, cam, sc = apply_scale(p.rgb, p.cam, p.Z_raw, p.mask_full, scale_mode, marker_cm, known)
    with timed(tm, "auto_select_ms"):
        auto = auto_select(Z_f, p.mask, cam.K, p.mask_full, None, sc.up_cam if sc.source == "marker" else None)
    return Scan(p.rgb, cam, Z_f, p.mask, p.mask_full, auto, tm, p.Z_raw, sc,
                known if sc.source == "known_dim" else None, key=p.key)


def prepare(image, scale_mode: str = "auto", marker_cm: float = MARKER_CM, known: dict | None = None,
            cache_dir=CACHE_DIR) -> Scan:
    """Photo (path or bytes) -> Scan. Depth + mask come from the perception cache when possible."""
    tm: dict = {}
    with timed(tm, "perceive_ms"):
        p = perceive(image, cache_dir)
    return scan_from_perception(p, scale_mode, marker_cm, known, tm)


def clip_below_table(res: ShapeResult, n, d):
    """Marker mode: the object rests ON the table plane n.X + d = 0 (n = up), so nothing can be below it."""
    C = res.grid.centers()
    below = (C @ n + d < 0).reshape(res.grid.dims)
    res.debug["table_clipped_voxels"] = int((res.occ & below).sum())
    res.occ &= ~below
    res.observed &= ~below


def auto_voxel(Z_f, mask, K, voxel_cm: float = VOXEL_CM, thickness: float = 1.0, plane_offset_cm: float = 0.0) -> float:
    """Safety cap: if the object comes out huge (usually DA's metric scale, 3-11x on real photos), coarsen the voxel
    so the longest side stays <= MAX_DIM_VOXELS. A 2 mm voxel on a 60 cm object = 17M voxels.
    A deeper mirror plane grows the object too: diverging rays widen it by ~off/z, the back moves by 2*off
    (real duck at +11 cm was 145 voxels wide and 2.6 s without this)."""
    vs, us = np.nonzero(mask)
    ext_px = max(np.ptp(us), np.ptp(vs)) + 1
    z = float(np.median(Z_f[mask]))
    ext_cm = ext_px * z / K[0, 0]
    off = max(0.0, plane_offset_cm)
    ext = max(ext_cm * max(1.0, thickness), ext_cm * (1 + off / z), 2 * off + 0.5 * ext_cm)
    return max(voxel_cm, ext * 1.15 / MAX_DIM_VOXELS)


def run_shape(Z_f, mask, mask_full, K, rgb, method: str, thickness: float = 1.0, voxel_cm: float = VOXEL_CM,
              timings: dict | None = None, cap: bool = True, frame_R=None, extra: dict | None = None,
              known: dict | None = None, table=None) -> tuple[ShapeResult, VoxelModel]:
    """known: {"w"|"h"|"d": cm} -> measure at the current scale, rescale depth, rerun (fixed point, <= 3 rescales)."""
    tm = timings if timings is not None else {}
    params = {"thickness": thickness, "mask_full": mask_full, **(extra or {})}
    if frame_R is not None:
        params["frame_R"] = frame_R

    def once(Z, key):
        v = auto_voxel(Z, mask, K, voxel_cm, thickness, params.get("plane_offset_cm", 0.0)) if cap else voxel_cm
        with timed(tm, key):
            r = METHODS[method](Z, mask, K, rgb, {**params, "voxel_cm": v})
            if table is not None:
                clip_below_table(r, *table)
        with timed(tm, "voxel_cleanup_ms"):
            m = finalize(r)
        return r, m

    res, model = once(Z_f, f"shape_{method}_ms")
    if known:
        # shape steps (RANSAC thresholds in cm, voxel cap) are not exactly scale-equivariant, so one rescale can miss
        total = 1.0
        for i in range(3):
            f, dim, val = known_dim_factor(model.measurements_cm, known)
            total *= f
            res, model = once((Z_f * total).astype(np.float32), f"shape_{method}_rescaled{i + 1}_ms")
            if abs(model.measurements_cm[dim] - val) / val < 0.03:
                break
        model.cleanup_info["known_dim"] = {"dim": dim, "value_cm": val, "factor": round(total, 4), "passes": i + 2}
    return res, model


def _marker_extra(scan: Scan, ex: dict) -> dict:
    if scan.scale.source == "marker":
        ex.setdefault("up_cam", scan.scale.up_cam)   # world-vertical revolve / plane search about the true up
        ex.setdefault("revolve_mode", "world")
    return ex


def shape_from_scan(scan: Scan, method: str, thickness: float = 1.0, timings=None, voxel_cm=VOXEL_CM, extra=None):
    res, model = run_shape(scan.Z_f, scan.mask, scan.mask_full, scan.K, scan.rgb, method, thickness, voxel_cm,
                           timings, frame_R=scan.frame_R, known=scan.known, extra=_marker_extra(scan, dict(extra or {})),
                           table=scan.scale.table if scan.scale.source == "marker" else None)
    model.scale_source = scan.scale.source
    return res, model


def mirror_from_scan(scan: Scan, offset_cm: float, cache: dict, timings=None, voxel_cm=VOXEL_CM):
    """Mirror v2 with a per-scan cache of the AUTO plane (and the known_dim factor), so the "symmetry plane depth"
    slider only shifts that same plane: stable semantics, no re-estimation per update. The known_dim factor is resolved
    once at offset 0; moving the plane then makes the object fatter instead of being rescaled back (by design)."""
    key = (scan.scale.source, json.dumps(scan.known), scan.prior_scale)
    c = cache.get(key)
    pl = scan.auto.get("plane")
    if c is None and pl and scan.known is None and abs(scan.prior_scale - 1.0) < 1e-9:
        # auto_select already estimated this exact plane (same depth, mask, up vector, prior): reuse it
        c = cache[key] = {"factor": 1.0, "known_dim": None, "plane": (pl["n"], pl["d"]), "estimator": pl.get("info")}
    if c is None:
        res, model = shape_from_scan(scan, "mirror", 1.0, timings, voxel_cm, extra={"prior_scale": scan.prior_scale})
        kd = model.cleanup_info.get("known_dim")
        c = cache[key] = {"factor": kd["factor"] if kd else 1.0, "known_dim": kd, "plane": res.debug["plane_auto"],
                          "estimator": res.debug.get("plane_estimator")}
        if abs(offset_cm) < 1e-9:
            return res, model
    ex = _marker_extra(scan, {"plane": c["plane"], "plane_offset_cm": float(offset_cm)})
    res, model = run_shape((scan.Z_f * c["factor"]).astype(np.float32), scan.mask, scan.mask_full, scan.K, scan.rgb,
                           "mirror", 1.0, voxel_cm, timings, frame_R=scan.frame_R, extra=ex,
                           table=scan.scale.table if scan.scale.source == "marker" else None)
    res.debug["plane_estimator"] = c["estimator"]
    model.scale_source = scan.scale.source
    if c["known_dim"]:
        model.cleanup_info["known_dim"] = c["known_dim"]
    return res, model


def mirror_offset_min(scan: Scan, cache: dict) -> float:
    """Most negative sensible plane offset (cm, in the scaled frame): the plane must stay behind ~95 % of the visible
    front points, else most rays never reach it and the object collapses (real duck: -1 cm -> a 1 cm fragment).
    Never above 0, so the auto plane is always allowed."""
    from .intrinsics import unproject
    from .shapes.mirror_v2 import as_plane
    key = (scan.scale.source, json.dumps(scan.known), scan.prior_scale)
    c = cache.get(key) or {"factor": 1.0, "plane": (scan.auto["plane"]["n"], scan.auto["plane"]["d"])}
    n, d = as_plane(c["plane"])
    vs, us = np.nonzero(scan.mask)
    P = unproject(scan.K, us, vs, (scan.Z_f[vs, us] * c["factor"]).astype(np.float64))
    s = P @ n + d          # signed distance to the auto plane; with offset o a ray reaches the plane iff s < o
    return float(min(0.0, np.percentile(s, 95)))
