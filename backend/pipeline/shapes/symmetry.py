"""Exact plane reflection utilities (camera frame, cm). Plane: n.X + d = 0, |n| = 1."""
from __future__ import annotations

import numpy as np

from .common import Grid


def reflect_points(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    return P - 2.0 * (P @ n + d)[:, None] * n[None, :]


def reflect_occupancy(occ: np.ndarray, grid: Grid, n: np.ndarray, d: float) -> np.ndarray:
    """Reflected copy of `occ` on the SAME grid, nearest-neighbor, pull-based (no holes): target voxel c is set
    iff reflect(c) falls in an occupied source voxel. Reflection is an involution, so this is exact up to
    voxel rounding."""
    C = grid.centers()
    idx, ok = grid.index(reflect_points(C, n, d))
    out = np.zeros(C.shape[0], bool)
    out[ok] = occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out.reshape(grid.dims)


def self_symmetry_iou(occ: np.ndarray, grid: Grid, n: np.ndarray, d: float) -> float:
    """IoU(occ, reflect(occ)) across the plane. 1.0 = perfectly mirror-symmetric about that plane."""
    r = reflect_occupancy(occ, grid, n, d)
    u = (occ | r).sum()
    return float((occ & r).sum() / u) if u else 0.0


def plane_errors(n: np.ndarray, d: float, n_true: np.ndarray, d_true: float, at: np.ndarray) -> tuple[float, float]:
    """(signed offset error cm at point `at` measured along n_true: + = estimated plane is farther along n_true,
    normal angle error deg)."""
    n_e = n if n @ n_true >= 0 else -n
    d_e = d if n @ n_true >= 0 else -d
    # point on the estimated plane closest to `at`
    p_e = at - (at @ n_e + d_e) * n_e
    off = float(p_e @ n_true + d_true)
    ang = float(np.degrees(np.arccos(np.clip(n_e @ n_true, -1, 1))))
    return off, ang


def depth_beyond_deepest(n: np.ndarray, d: float, front: np.ndarray, pct: float = 100.0) -> float:
    """How far the plane sits BEHIND the deepest observed front point (cm, along +n which points away from the
    camera). Negative = the plane is in front of the deepest observed point."""
    nn, dd = (n, d) if n[2] >= 0 else (-n, -d)
    s = front @ nn + dd        # >0 means the point is behind the plane
    return float(-np.percentile(s, pct))


def _index_affine(grid: Grid, n: np.ndarray, d: float):
    """Reflection as an affine map on CONTINUOUS voxel coordinates q = ijk + 0.5 (grid frame):
    q' = A q + b. Avoids building camera-frame centers for every voxel."""
    ng = n if grid.R is None else grid.R @ n            # plane normal in the grid frame
    # plane in grid frame: ng . P_g + d = 0 (rotation preserves distances), P_g = origin + q * v
    A = np.eye(3) - 2.0 * np.outer(ng, ng)
    v, o = grid.voxel, grid.origin
    # P' = P - 2(ng.P + d) ng  ->  q' = (P' - o)/v
    b = -2.0 * (ng @ o + d) * ng / v
    return A, b


def reflect_union(occ: np.ndarray, colors: np.ndarray, grid: Grid, n: np.ndarray, d: float):
    """occ ∪ reflect(occ) on the same grid; voxels added by the reflection take their SOURCE voxel's color.
    Pull-based over the bounding box of the reflected occupied voxels, in index space (fast)."""
    occ_idx = np.argwhere(occ)
    out = occ.copy()
    col = colors.copy()
    if len(occ_idx) == 0:
        return out, col
    A, b = _index_affine(grid, n, d)
    dims = np.array(grid.dims)
    r = np.floor((occ_idx + 0.5) @ A.T + b).astype(np.int64)
    lo = np.clip(r.min(0) - 1, 0, dims - 1)
    hi = np.clip(r.max(0) + 2, 1, dims)
    sub = np.indices(tuple(hi - lo)).reshape(3, -1).T + lo
    src = np.floor((sub + 0.5) @ A.T + b).astype(np.int64)
    inb = np.all((src >= 0) & (src < dims), axis=1)
    sub, src = sub[inb], src[inb]
    hit = occ[src[:, 0], src[:, 1], src[:, 2]]
    sub, src = sub[hit], src[hit]
    new = ~occ[sub[:, 0], sub[:, 1], sub[:, 2]]
    out[sub[:, 0], sub[:, 1], sub[:, 2]] = True
    col[sub[new, 0], sub[new, 1], sub[new, 2]] = colors[src[new, 0], src[new, 1], src[new, 2]]
    return out, col
