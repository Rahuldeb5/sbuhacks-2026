"""Midplane variants for mirror. All return a common.Midplane; all share the same front surface.

M0  rim RANSAC plane (orientation + offset from the contour band)             [original]
M1  rim orientation, offset so the plane passes (LSQ) through Z_f + h_inflate  [inflation prior]
M2  RANSAC on the OUTER rim of the uneroded mask, rejecting background-bled rim depths
M3  rim orientation, offset = P10(front depth) + half the median medial-axis silhouette width
M4  (marker) rim normal projected horizontal (plane ⊥ table), offset by M1's rule
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..intrinsics import pixel_rays, unproject
from .common import Midplane, fit_midplane, mask_pixels, ransac_plane

VARIANTS = ("M0", "M1", "M2", "M3", "M4")


def _dt(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def _with_offset(mid: Midplane, n: np.ndarray, c: float) -> Midplane:
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n, c = -n, -c
    return Midplane(n, float(c), mid.band_uv, mid.residuals, mid.inliers, mid.fallback)


def inflate_h(Z_f, mask, mask_full, K):
    """Inflate half-thickness (cm) at mask pixels; px->cm at the front depth."""
    us, vs = mask_pixels(mask)
    dt_cm = _dt(mask_full)[vs, us] * Z_f[vs, us] / K[0, 0]
    return us, vs, np.sqrt(dt_cm * dt_cm.max())


def offset_inflation(n, Z_f, mask, mask_full, K) -> float:
    us, vs, h = inflate_h(Z_f, mask, mask_full, K)
    P = unproject(K, us, vs, Z_f[vs, us].astype(np.float64) + h)
    return float(-np.mean(P @ n))


def m2_rim(Z_f, mask_full, K, thresh_cm=0.5):
    dt_in = _dt(mask_full)
    dt_out = _dt(~mask_full)
    rim = mask_full & (dt_in <= 1.5)
    bg = (~mask_full & (dt_out >= 4) & (dt_out <= 10)).astype(np.float32)
    it = (mask_full & (dt_in >= 6) & (dt_in <= 12)).astype(np.float32)
    Zf = Z_f.astype(np.float32)
    box = (25, 25)
    zbg = cv2.boxFilter(Zf * bg, -1, box, normalize=False) / np.maximum(cv2.boxFilter(bg, -1, box, normalize=False), 1e-6)
    zin = cv2.boxFilter(Zf * it, -1, box, normalize=False) / np.maximum(cv2.boxFilter(it, -1, box, normalize=False), 1e-6)
    keep = rim & (np.abs(Zf - zin) <= np.abs(Zf - zbg))
    us, vs = mask_pixels(keep)
    frac = float(keep.sum() / max(1, rim.sum()))
    if len(us) < 10:
        return None, frac
    P = unproject(K, us, vs, Z_f[vs, us].astype(np.float64))
    n, d, inl, res = ransac_plane(P, thresh_cm)
    if abs(n[2]) < 0.3:
        return None, frac
    return Midplane(n, d, np.stack([us, vs], 1), res, inl, False), frac


def offset_m3(n, Z_f, mask, mask_full, K) -> tuple[float, float]:
    z10 = float(np.percentile(Z_f[mask], 10))
    sk = skeletonize(mask_full) & mask
    dt = _dt(mask_full)
    if sk.sum() < 3:
        sk = mask
    width_cm = np.median(2 * dt[sk] * Z_f[sk] / K[0, 0])
    z_mid = z10 + 0.5 * width_cm
    vs, us = np.nonzero(mask)
    ray = pixel_rays(K, np.array([us.mean()]), np.array([vs.mean()]))[0]
    return float(-(n @ (ray * z_mid))), float(width_cm)


def midplane_variant(name: str, Z_f, mask, mask_full, K, up_cam=None) -> Midplane:
    base = fit_midplane(Z_f, mask, K)        # M0; also supplies the rim orientation for M1/M3/M4
    if name == "M0":
        return base
    if name == "M1":
        return _with_offset(base, base.n, offset_inflation(base.n, Z_f, mask, mask_full, K))
    if name == "M2":
        mid, frac = m2_rim(Z_f, mask_full, K)
        out = mid if mid is not None else base
        out.inlier_frac_kept = frac  # type: ignore[attr-defined]
        return out
    if name == "M3":
        c, _ = offset_m3(base.n, Z_f, mask, mask_full, K)
        return _with_offset(base, base.n, c)
    if name == "M4":
        if up_cam is None:
            raise ValueError("M4 needs the table up vector (marker mode)")
        n = base.n - (base.n @ up_cam) * up_cam
        n /= np.linalg.norm(n)
        return _with_offset(base, n, offset_inflation(n, Z_f, mask, mask_full, K))
    raise ValueError(name)
