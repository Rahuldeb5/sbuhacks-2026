from __future__ import annotations

import cv2
import numpy as np

from ..intrinsics import unproject
from .common import VOXEL_CM, ray_pixels, ShapeResult, compute_observed, fill_rays, fit_midplane, mask_pixels, ray_plane_z


def inflate(Z_f, mask, K, rgb, params=None) -> ShapeResult:
    p = params or {}
    t = float(p.get("thickness", 1.0))
    vox = float(p.get("voxel_cm", VOXEL_CM))
    mid = p.get("midplane") or fit_midplane(Z_f, mask, K, p.get("band_px", 3))
    full = p.get("mask_full", mask)
    dt_px = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, 5)
    us, vs = ray_pixels(mask, Z_f, K, vox)
    zf = Z_f[vs, us].astype(np.float64)
    zc = ray_plane_z(K, us, vs, mid.n, mid.d)
    dt_cm = dt_px[vs, us] * zc / K[0, 0]
    h = np.sqrt(dt_cm * dt_cm.max())
    zb = np.maximum(zf + t * (zc + h - zf), zf + p.get("min_thick_cm", 2 * vox))
    occ, colors, grid = fill_rays(K, us, vs, zf, zb, rgb, vox, R=p.get("frame_R"))
    front = unproject(K, us, vs, zf)
    return ShapeResult(occ, compute_observed(occ, grid, front), colors, grid, front, "inflate", t,
                       {"midplane": mid, "h_max_cm": float(h.max())})
