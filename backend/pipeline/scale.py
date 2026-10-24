"""Metric scale + gravity: "marker" (ArUco calibration plate), "known_dim" (one real dimension), "estimate".

Marker convention: cv2.aruco DICT_4X4_50, id 0, side length = full outer black square (default 6.0 cm),
lying flat on the table. Corner order from the detector: TL, TR, BR, BL.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .intrinsics import pixel_rays

log = logging.getLogger("pipeline.scale")

MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_CM = 6.0
LEVEL_UP = np.array([0.0, -1.0, 0.0])   # camera frame: y is down, so "up" for a level camera is -y

MASK_DILATE_PX = 15
RANSAC_ITERS = 300
RANSAC_THRESH_FRAC = 0.03      # of marker depth
MIN_INLIERS = 300
MIN_SPAN_FRAC = 0.15           # inlier plane-depth span (p95-p5) relative to marker depth


@dataclass
class ScaleInfo:
    source: str = "estimate"               # estimate | known_dim | marker
    up_cam: np.ndarray = field(default_factory=lambda: LEVEL_UP.copy())
    table: tuple | None = None             # (n, d): n.X + d = 0 in camera frame, n = up
    info: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"scale_source": self.source, "up_cam": [round(float(x), 4) for x in self.up_cam], **self.info}


def world_from_cam(up_cam: np.ndarray) -> np.ndarray:
    """Rows = world axes in camera coords: y = up, x = camera x projected onto the horizontal plane."""
    y = up_cam / np.linalg.norm(up_cam)
    x = np.array([1.0, 0, 0]) - y[0] * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])


def detect_marker(rgb: np.ndarray, marker_id: int = MARKER_ID):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == marker_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def _obj_points(side_cm: float) -> np.ndarray:
    h = side_cm / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def focal_from_homography(corners: np.ndarray, side_cm: float, cx: float, cy: float) -> float | None:
    """Square pixels, known principal point, zero skew: f from r1.r2 = 0 and |r1| = |r2|."""
    H, _ = cv2.findHomography(_obj_points(side_cm)[:, :2], corners - [cx, cy])
    if H is None:
        return None
    (h11, h12, _), (h21, h22, _), (h31, h32, _) = H
    est = []
    den1 = h31 * h32
    if abs(den1) > 1e-12:
        f2 = -(h11 * h12 + h21 * h22) / den1
        if f2 > 0:
            est.append((abs(den1), f2))
    den2 = h31 ** 2 - h32 ** 2
    if abs(den2) > 1e-12:
        f2 = (h12 ** 2 + h22 ** 2 - h11 ** 2 - h21 ** 2) / den2
        if f2 > 0:
            est.append((abs(den2), f2))
    if not est:
        return None   # (near) fronto-parallel marker: no perspective, focal unobservable
    w = np.array([e[0] for e in est])
    return float(math.sqrt(np.sum(w * np.array([e[1] for e in est])) / w.sum()))


def marker_pose(corners: np.ndarray, K: np.ndarray, side_cm: float):
    ok, rvec, tvec = cv2.solvePnP(_obj_points(side_cm), corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def pitch_roll_deg(up_cam: np.ndarray) -> tuple[float, float]:
    """pitch = how far the optical axis points below horizontal; roll = image-up vs world-up angle."""
    pitch = math.degrees(math.asin(float(np.clip(-up_cam[2], -1, 1))))
    roll = math.degrees(math.atan2(up_cam[0], -up_cam[1]))
    return pitch, roll


def plane_depth(K, n, d, shape) -> np.ndarray:
    H, W = shape
    vs, us = np.mgrid[0:H, 0:W]
    rays = pixel_rays(K, us.ravel().astype(float), vs.ravel().astype(float))
    den = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(np.abs(den) > 1e-6, -d / den, np.nan)
    Z[Z <= 0] = np.nan
    return Z.reshape(H, W)


def apply_marker(rgb, Z_da, K, mask_full, side_cm: float = MARKER_CM, cam_source: str = "default"):
    """Returns (Z_corrected, K_used, ScaleInfo) or None if no marker is found."""
    corners = detect_marker(rgb)
    if corners is None:
        return None
    H, W = Z_da.shape
    info: dict = {"marker_corners_px": corners.round(2).tolist(), "marker_cm": side_cm}
    K = K.copy()
    f_h = focal_from_homography(corners, side_cm, K[0, 2], K[1, 2])
    info["focal_from_marker_px"] = f_h
    if cam_source != "exif" and f_h is not None and 0.3 * W < f_h < 5 * W:
        K[0, 0] = K[1, 1] = f_h
        info["focal_source"] = "marker_homography"
    else:
        info["focal_source"] = "exif" if cam_source == "exif" else "default_hfov"
    pose = marker_pose(corners, K, side_cm)
    if pose is None:
        return None
    Rm, t = pose
    n = Rm[:, 2]
    if n @ (-t) < 0:          # normal must point toward the camera's side of the table = up
        n = -n
    d = -float(n @ t)
    pitch, roll = pitch_roll_deg(n)
    info.update({"pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2), "marker_dist_cm": round(float(np.linalg.norm(t)), 2)})

    Zp = plane_depth(K, n, d, (H, W))
    quad = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(quad, np.round(corners).astype(np.int32), 1)
    quad = quad.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_DILATE_PX + 1,) * 2)
    obj = cv2.dilate(mask_full.astype(np.uint8), k).astype(bool)
    zm = float(t[2])
    cand = ~obj & np.isfinite(Zp) & (Zp < 4 * zm) & np.isfinite(Z_da)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(cand)
    if len(idx) > 20000:
        idx = rng.choice(idx, 20000, replace=False)
    idx = np.union1d(idx, np.flatnonzero(quad & np.isfinite(Zp)))
    x, y = Z_da.ravel()[idx].astype(np.float64), Zp.ravel()[idx]
    thr = RANSAC_THRESH_FRAC * zm
    best, best_n = None, -1
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, len(x), 2)
        if abs(x[i] - x[j]) < 1e-6:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0.02:
            continue
        b = y[i] - a * x[i]
        cnt = int((np.abs(y - (a * x + b)) < thr).sum())
        if cnt > best_n:
            best_n, best = cnt, (a, b)
    mode = "scale_only"
    if best is not None:
        inl = np.abs(y - (best[0] * x + best[1])) < thr
        a, b = np.polyfit(x[inl], y[inl], 1)
        inl = np.abs(y - (a * x + b)) < thr
        span = np.percentile(y[inl], 95) - np.percentile(y[inl], 5) if inl.sum() else 0
        info.update({"affine_inliers": int(inl.sum()), "affine_span_cm": round(float(span), 2)})
        if inl.sum() >= MIN_INLIERS and span >= MIN_SPAN_FRAC * zm and a > 0.02:
            mode = "affine"
    if mode == "affine":
        Zc = (a * Z_da + b).astype(np.float32)
        info.update({"depth_correction": "affine", "affine_a": round(float(a), 4), "affine_b_cm": round(float(b), 2)})
    else:
        mq = quad & np.isfinite(Zp)
        s = float(np.median(Zp[mq] / Z_da[mq]))
        Zc = (s * Z_da).astype(np.float32)
        info.update({"depth_correction": "scale_only", "scale": round(s, 4)})
    Zc = np.maximum(Zc, 1.0)
    info["da_vs_plane_at_marker"] = round(float(np.median(Z_da[quad] / np.where(np.isfinite(Zp[quad]), Zp[quad], np.nan))), 3)
    log.info("marker: %s", {k: v for k, v in info.items() if k != "marker_corners_px"})
    return Zc, K, ScaleInfo("marker", n, (n, d), info)


def known_dim_factor(measured_cm: dict, known: dict) -> tuple[float, str, float]:
    """known e.g. {"h": 6.0}. Returns (factor, dim, value)."""
    (dim, val), = known.items()
    return float(val) / float(measured_cm[dim]), dim, float(val)


def parse_known(s: str | None) -> dict | None:
    """'h=6' / 'height_cm=6' / 'w=8' -> {'h': 6.0}."""
    if not s:
        return None
    k, v = s.split("=")
    k = k.strip().lower().replace("_cm", "")
    k = {"height": "h", "width": "w", "length": "w", "depth": "d"}.get(k, k)
    if k not in ("w", "h", "d"):
        raise ValueError(f"known dimension must be w/h/d (got {k})")
    return {k: float(v)}
