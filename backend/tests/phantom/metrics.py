"""Voxel IoU, Chamfer (mm), watertightness."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pipeline.shapes.common import Grid
from pipeline.voxels import surface_mask


def to_union(a: np.ndarray, ga: Grid, b: np.ndarray, gb: Grid):
    """Place two lattice-aligned grids (same voxel) into one common grid."""
    assert abs(ga.voxel - gb.voxel) < 1e-9
    v = ga.voxel
    oa, ob = np.round(ga.origin / v).astype(int), np.round(gb.origin / v).astype(int)
    lo = np.minimum(oa, ob)
    hi = np.maximum(oa + np.array(ga.dims), ob + np.array(gb.dims))
    A = np.zeros(tuple(hi - lo), bool)
    B = np.zeros(tuple(hi - lo), bool)
    sa, sb = oa - lo, ob - lo
    A[sa[0]:sa[0] + ga.dims[0], sa[1]:sa[1] + ga.dims[1], sa[2]:sa[2] + ga.dims[2]] = a
    B[sb[0]:sb[0] + gb.dims[0], sb[1]:sb[1] + gb.dims[1], sb[2]:sb[2] + gb.dims[2]] = b
    return A, B, Grid(lo * v, tuple(hi - lo), v)


def iou(a, ga, b, gb) -> float:
    A, B, _ = to_union(a, ga, b, gb)
    return float((A & B).sum() / max(1, (A | B).sum()))


def chamfer_mm(occ: np.ndarray, grid: Grid, gt_pts_cm: np.ndarray) -> float:
    """Symmetric Chamfer: 0.5*(mean d(recon surface voxel centers -> GT surface) + mean d(GT -> recon)), mm.
    Floor from quantization alone is ~0.5 mm at 2 mm voxels."""
    rec = grid.centers(np.argwhere(surface_mask(occ)))
    if len(rec) == 0:
        return float("nan")
    d1, _ = cKDTree(gt_pts_cm).query(rec)
    d2, _ = cKDTree(rec).query(gt_pts_cm)
    return float(10 * 0.5 * (d1.mean() + d2.mean()))


def iou_centered(a, ga, b, gb) -> float:
    """IoU after translating `a` by whole voxels so its centroid matches `b`'s. Separates shape+size error
    from absolute placement (a 4 % depth-scale error at 60 cm is a 2.4 cm shift: IoU ~0 for a 6 cm object
    even when the shape is right). Diagnostic only; the app recenters models anyway."""
    A, B, g = to_union(a, ga, b, gb)
    if not A.any() or not B.any():
        return 0.0
    sh = np.round(np.argwhere(B).mean(0) - np.argwhere(A).mean(0)).astype(int)
    pad = int(np.abs(sh).max()) + 1
    Ap, Bp = np.pad(A, pad), np.pad(B, pad)
    Ap = np.roll(Ap, tuple(sh), axis=(0, 1, 2))
    return float((Ap & Bp).sum() / max(1, (Ap | Bp).sum()))
