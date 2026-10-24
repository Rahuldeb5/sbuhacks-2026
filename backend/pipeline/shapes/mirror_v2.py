from __future__ import annotations

import numpy as np

from ..intrinsics import pixel_rays, unproject
from .common import VOXEL_CM, Grid, Midplane, ShapeResult, compute_observed, fill_rays, ray_pixels
from .symmetry import reflect_points, reflect_union


def as_plane(pl) -> tuple[np.ndarray, float]:
    if isinstance(pl, Midplane):
        n, d = np.asarray(pl.n, float), float(pl.d)
    else:
        n, d = np.asarray(pl[0], float), float(pl[1])
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    if n[2] < 0:
        n, d = -n, -d
    return n, d


def half_solid_rays(Z_f, mask, K, n, d, vox):
    """(us, vs, z_front, z_plane) for pixels whose ray reaches the plane behind the front surface."""
    us, vs = ray_pixels(mask, Z_f, K, vox)
    zf = Z_f[vs, us].astype(np.float64)
    rays = pixel_rays(K, us.astype(np.float64), vs.astype(np.float64))
    den = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = np.where(den > 1e-6, -d / den, -np.inf)       # n points away from the camera: den > 0 for a hit ahead
    ok = zc > zf
    return us, vs, zf, zc, ok


def mirror_v2(Z_f, mask, K, rgb, params=None) -> ShapeResult:
    from .plane_fit import estimate_plane   # local import: plane_fit imports this module's helpers
    p = params or {}
    vox = float(p.get("voxel_cm", VOXEL_CM))
    R = p.get("frame_R")
    if p.get("plane") is not None:
        n, d = as_plane(p["plane"])
        est_info = {"estimator": "given"}
    else:
        (n, d), est_info = estimate_plane(Z_f, mask, p.get("mask_full", mask), K,
                                          p.get("plane_estimator"), p.get("up_cam"), p.get("prior_scale", 1.0),
                                          p.get("depth_ratio"))
    off = float(p.get("plane_offset_cm", 0.0))
    d_used = d - off                                   # move the plane by `off` along +n

    us, vs, zf, zc, ok = half_solid_rays(Z_f, mask, K, n, d_used, vox)
    if ok.sum() < 10:
        # degenerate (plane in front of almost everything): keep a thin shell so the pipeline still returns
        zc = np.where(ok, zc, zf + 2 * vox)
        ok = np.ones_like(ok)
    us, vs, zf, zc = us[ok], vs[ok], zf[ok], zc[ok]
    rays = pixel_rays(K, us.astype(np.float64), vs.astype(np.float64))
    ends = np.concatenate([rays * zf[:, None], rays * zc[:, None]])
    grid = Grid.around_points(np.concatenate([ends, reflect_points(ends, n, d_used)]), vox, R=R)
    occ_h, col_h, grid = fill_rays(K, us, vs, zf, zc, rgb, vox, grid=grid)

    occ, colors = reflect_union(occ_h, col_h, grid, n, d_used)   # full = H ∪ reflect(H)

    front = unproject(K, us, vs, zf)
    return ShapeResult(occ, compute_observed(occ, grid, front), colors, grid, front, "mirror", 1.0,
                       {"midplane": Midplane(n, d_used, np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool), False),
                        "plane_auto": (n.tolist(), d), "plane_offset_cm": off, "plane_estimator": est_info,
                        "half_rays": int(ok.sum()), "symmetric_plane": (n, d_used)})
