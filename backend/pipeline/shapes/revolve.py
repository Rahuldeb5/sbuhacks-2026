"""Revolve: solid of revolution about a (roughly vertical) axis fitted through the silhouette row centers."""
from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes

from ..intrinsics import pixel_rays, project, unproject
from .common import VOXEL_CM, Grid, ShapeResult, compute_observed, mask_pixels, nearest_mask_pixel


LEVEL_UP = np.array([0.0, -1.0, 0.0])
# step 4: with a marker the caller passes revolve_mode="world" + up_cam; without one, "axisup" (up from the v1 axis
# direction; the specified level-camera assumption was much worse under tilt). "v1" = original.
DEFAULT_MODE = "axisup"


def row_profile(mask: np.ndarray, min_halfwidth_px: float = 2.0):
    """For each mask row: v, left, right, center, half-width (px)."""
    rows = np.nonzero(mask.any(1))[0]
    sub = mask[rows]
    left = np.argmax(sub, axis=1)
    right = mask.shape[1] - 1 - np.argmax(sub[:, ::-1], axis=1)
    hw = (right - left + 1) / 2.0
    keep = hw >= min_halfwidth_px
    v = rows[keep].astype(np.float64)
    return v, left[keep].astype(np.float64), right[keep].astype(np.float64), (left[keep] + right[keep]) / 2.0, hw[keep]


def robust_line(x: np.ndarray, y: np.ndarray, max_pts: int = 300):
    """Theil-Sen y = a*x + b (subsampled for speed). Robust to ~29% outlier rows."""
    if len(x) > max_pts:
        s = np.linspace(0, len(x) - 1, max_pts).astype(int)
        x, y = x[s], y[s]
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0, float(np.median(y))
    a, b, _, _ = theilslopes(y, x)
    return float(a), float(b)


def fit_axis(Z_f, mask, K, mask_full=None):
    """Returns dict describing the 3D axis and per-row radii (cm)."""
    full = mask if mask_full is None else mask_full
    v, l, r, c, hw = row_profile(full)
    a, b = robust_line(v, c)
    c_fit = a * v + b
    cos_tilt = 1.0 / np.sqrt(1.0 + a * a)
    w_perp = hw * cos_tilt                                  # row cuts a tilted axis obliquely
    fx = K[0, 0]
    # median front depth near the axis, per row (eroded mask = trustworthy depth)
    zfront = np.full(len(v), np.nan)
    for i, (vv, cc, w) in enumerate(zip(v.astype(int), c_fit, hw)):
        half = max(2.0, 0.15 * w)
        u0, u1 = int(max(0, np.floor(cc - half))), int(min(mask.shape[1], np.ceil(cc + half) + 1))
        m = mask[vv, u0:u1]
        if m.any():
            zfront[i] = np.median(Z_f[vv, u0:u1][m])
    ok = np.isfinite(zfront)
    if ok.sum() < 2:
        raise ValueError("revolve: no depth near the axis")
    r_cm = w_perp[ok] * zfront[ok] / (fx - w_perp[ok])     # solves r = w*(Zfront + r)/fx
    za, zb = robust_line(v[ok], zfront[ok] + r_cm)          # axis depth is linear-ish in v (straight 3D axis)
    z_axis = za * v + zb
    A = unproject(K, c_fit, v, z_axis)                      # axis point per row
    radius = w_perp * z_axis / fx
    d = A[-1] - A[0]
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0, 1.0, 0])
    s = (A - A[0]) @ d
    o = np.argsort(s)
    return {"v": v, "left": l, "right": r, "center": c, "center_fit": c_fit, "hw": hw, "slope": a,
            "A0": A[0], "d": d, "s": s[o], "radius": radius[o], "A": A, "z_axis": z_axis}


def fit_axis_world(Z_f, mask, K, mask_full, up_cam, init=None, max_pts=4000):
    """Step 4: axis VERTICAL in the world (up_cam from the marker, else a level camera). Unknown = the axis's
    horizontal position (a, b). Each silhouette edge ray is tangent to the surface: its horizontal distance to
    the axis is the radius at the tangent height (the ray's closest-approach height). Front-surface points must
    lie at that radius from the axis at their height. Solve (a, b) by robust least squares."""
    from scipy.optimize import least_squares
    from ..scale import world_from_cam

    R = world_from_cam(up_cam)
    v, l, r, c, hw = row_profile(mask_full)
    dl = pixel_rays(K, l - 0.5, v) @ R.T          # silhouette edges sit on pixel boundaries
    dr = pixel_rays(K, r + 0.5, v) @ R.T
    us, vs = mask_pixels(mask)
    if len(us) > max_pts:
        sel = np.linspace(0, len(us) - 1, max_pts).astype(int)
        us, vs = us[sel], vs[sel]
    Pw = unproject(K, us, vs, Z_f[vs, us].astype(np.float64)) @ R.T

    def tangent(dw, a, b):
        h2 = dw[:, 0] ** 2 + dw[:, 2] ** 2
        rad = np.abs(-a * dw[:, 2] + b * dw[:, 0]) / np.sqrt(h2)
        tt = (a * dw[:, 0] + b * dw[:, 2]) / h2
        return rad, tt * dw[:, 1]

    def profile(ab):
        a, b = ab
        rl, yl = tangent(dl, a, b)
        rr, yr = tangent(dr, a, b)
        y, rad = (yl + yr) / 2, (rl + rr) / 2
        o = np.argsort(y)
        return y[o], rad[o]

    def resid(ab):
        y, rad = profile(ab)
        ok = (Pw[:, 1] >= y[0]) & (Pw[:, 1] <= y[-1])
        rho = np.hypot(Pw[:, 0] - ab[0], Pw[:, 2] - ab[1])
        return np.where(ok, rho - np.interp(Pw[:, 1], y, rad), 0.0)   # fixed length for least_squares

    if init is not None:
        mid_c = init["A0"] + init["d"] * (init["s"][0] + init["s"][-1]) / 2
        x0 = (R @ mid_c)[[0, 2]]
    else:
        x0 = np.array([np.median(Pw[:, 0]), np.median(Pw[:, 2]) + 3.0])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.2)
    a, b = sol.x
    y, rad = profile(sol.x)
    # keep the profile within the object's visible height range and drop runaway radii (rays nearly parallel to
    # the axis on non-revolve shapes give km-scale tangent heights and a giant voxel grid)
    span = np.ptp(Pw[:, 1])
    keep = (y >= Pw[:, 1].min() - 0.1 * span) & (y <= Pw[:, 1].max() + 0.1 * span) & (rad <= 3 * np.median(rad))
    if keep.sum() >= 2:
        y, rad = y[keep], rad[keep]
    A0w = np.array([a, y[0], b])
    up = up_cam / np.linalg.norm(up_cam)
    return {"v": v, "left": l, "right": r, "center": c, "center_fit": c, "hw": hw, "slope": 0.0,
            "A0": R.T @ A0w, "d": up, "s": y - y[0], "radius": rad, "mode": "world",
            "fit_cost": float(sol.cost), "fit_init_xz": x0.tolist(), "fit_xz": [float(a), float(b)]}


def revolve(Z_f, mask, K, rgb, params=None) -> ShapeResult:
    p = params or {}
    t = float(p.get("thickness", 1.0))
    vox = float(p.get("voxel_cm", VOXEL_CM))
    full = p.get("mask_full", mask)
    mode = p.get("revolve_mode", DEFAULT_MODE)
    ax = p.get("axis")
    if ax is None:
        ax = fit_axis(Z_f, mask, K, full)
        if mode == "axisup":
            # no gravity known: assume the object stands upright, take "up" from the v1 axis direction, and
            # voxelize in that frame so the object comes out standing (step 4)
            up = ax["d"] if ax["d"] @ LEVEL_UP > 0 else -ax["d"]
            ax = fit_axis_world(Z_f, mask, K, full, up, init=ax)
            ax["mode"] = "axisup"
            from ..scale import world_from_cam
            p = {**p, "frame_R": world_from_cam(up)} if p.get("frame_R") is not None else p
        elif mode == "world":
            ax = fit_axis_world(Z_f, mask, K, full, np.asarray(p.get("up_cam", LEVEL_UP), float), init=ax)
    A0, d, s_rows, r_rows = ax["A0"], ax["d"], ax["s"], ax["radius"]

    mid = A0 + d * (s_rows[0] + s_rows[-1]) / 2
    view = mid / np.linalg.norm(mid)
    e_dep = view - (view @ d) * d
    e_dep /= np.linalg.norm(e_dep)
    e_lat = np.cross(d, e_dep)

    rmax = r_rows.max() * max(1.0, 2 * t - 1)
    # bounding cylinder (radius rmax around the axis, between the profile ends), boxed in the grid frame
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = rmax * (np.cos(ang)[:, None] * e_lat + np.sin(ang)[:, None] * e_dep)
    box = np.concatenate([A0 + d * s_rows[0] + ring, A0 + d * s_rows[-1] + ring])
    grid = Grid.around_points(box, vox, R=p.get("frame_R"))
    C = grid.centers()
    q = C - A0
    s = q @ d
    rad = q - s[:, None] * d
    lat, dep = rad @ e_lat, rad @ e_dep
    r = np.interp(s, s_rows, r_rows, left=0.0, right=0.0)
    # t scales the depth semi-axis; the ellipse center moves back by (t-1)*r so the FRONT never moves
    inside = (r > 0) & ((lat / np.maximum(r, 1e-9)) ** 2 + ((dep - (t - 1) * r) / np.maximum(t * r, 1e-9)) ** 2 <= 1.0)
    occ = inside.reshape(grid.dims)

    # color: each voxel takes the pixel of the front-surface point at the same height + lateral offset
    colors = np.zeros(grid.dims + (3,), np.uint8)
    ii = np.nonzero(inside)[0]
    ri = r[ii]
    li = np.clip(lat[ii], -ri, ri)
    Pf = A0 + s[ii, None] * d + li[:, None] * e_lat - np.sqrt(ri ** 2 - li ** 2)[:, None] * e_dep
    u, v = project(K, Pf)
    u = np.clip(np.round(u), 0, rgb.shape[1] - 1).astype(int)
    v = np.clip(np.round(v), 0, rgb.shape[0] - 1).astype(int)
    iy, ix = nearest_mask_pixel(full)
    colors.reshape(-1, 3)[ii] = rgb[iy[v, u], ix[v, u]]

    us, vs = mask_pixels(mask)
    front = unproject(K, us, vs, Z_f[vs, us].astype(np.float64))
    return ShapeResult(occ, compute_observed(occ, grid, front), colors, grid, front, "revolve", t,
                       {"axis": ax, "revolve_mode": ax.get("mode", "v1")})
