"""Shared geometry for shape methods: voxel grid, ray filling, midplane fit, observed marking."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from ..intrinsics import pixel_rays, unproject

VOXEL_CM = 0.2
STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass
class Grid:
    """Axis-aligned voxel grid. By default in the CAMERA frame; with R set, in a rotated frame
    P_grid = R @ P_cam (the app uses a gravity-aligned world frame: y up). origin is lattice-aligned
    (a multiple of voxel) in the grid frame, so two grids with the same voxel size and R share voxel
    boundaries (IoU is an index offset). Voxel (i,j,k) center (grid frame) = origin + (ijk+0.5)*voxel."""
    origin: np.ndarray
    dims: tuple[int, int, int]
    voxel: float
    R: np.ndarray | None = None

    @staticmethod
    def around(lo: np.ndarray, hi: np.ndarray, voxel: float, margin: int = 2, R=None) -> "Grid":
        """lo/hi in the GRID frame."""
        o = (np.floor(lo / voxel) - margin) * voxel
        top = (np.ceil(hi / voxel) + margin) * voxel
        dims = tuple(int(x) for x in np.round((top - o) / voxel))
        return Grid(o, dims, voxel, R)

    @staticmethod
    def around_points(P_cam: np.ndarray, voxel: float, margin: int = 2, R=None) -> "Grid":
        Pg = P_cam if R is None else P_cam @ R.T
        return Grid.around(Pg.min(0), Pg.max(0), voxel, margin, R)

    def to_grid(self, P_cam: np.ndarray) -> np.ndarray:
        return P_cam if self.R is None else P_cam @ self.R.T

    def index(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points -> (idx (N,3) int, in-bounds bool (N,))."""
        idx = np.floor((self.to_grid(P) - self.origin) / self.voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(self.dims)), axis=1)
        return idx, ok

    def centers(self, ijk: np.ndarray | None = None) -> np.ndarray:
        """Voxel centers in the CAMERA frame."""
        if ijk is None:
            ijk = np.indices(self.dims).reshape(3, -1).T
        C = self.origin + (ijk + 0.5) * self.voxel
        return C if self.R is None else C @ self.R

    @property
    def transform(self) -> np.ndarray:
        """4x4: homogeneous voxel index (i,j,k,1) -> camera-frame cm of the voxel center."""
        T = np.eye(4)
        T[:3, :3] *= self.voxel
        T[:3, 3] = self.origin + 0.5 * self.voxel
        if self.R is not None:
            G = np.eye(4)
            G[:3, :3] = self.R.T
            T = G @ T
        return T


@dataclass
class ShapeResult:
    occ: np.ndarray            # (nx,ny,nz) bool
    observed: np.ndarray       # (nx,ny,nz) bool
    colors: np.ndarray         # (nx,ny,nz,3) uint8
    grid: Grid
    front_points: np.ndarray   # (N,3) cm, camera frame: the visible surface used for `observed`
    method: str
    thickness: float
    debug: dict = field(default_factory=dict)


def fill_rays(K, us, vs, z0, z1, rgb, voxel=VOXEL_CM, step=None, grid: Grid | None = None, R=None):
    """Sample each pixel ray from z0 to z1 (planar depths) at step < voxel/2; mark voxels, color them
    with the source pixel. Returns (occ, colors, grid)."""
    step = step or voxel * 0.45
    rays = pixel_rays(K, us.astype(np.float64), vs.astype(np.float64))
    # step is along the ray in 3D; convert to a z-step per pixel
    raylen = np.linalg.norm(rays, axis=1)
    dz = step / raylen
    n = np.maximum(1, np.ceil((z1 - z0) / dz).astype(np.int64) + 1)
    if grid is None:
        ends = np.concatenate([rays * z0[:, None], rays * z1[:, None]])
        grid = Grid.around_points(ends, voxel, R=R)
    pix = np.repeat(np.arange(len(us)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(pix.size) - start                     # sample number along its ray
    z = np.minimum(z0[pix] + k * dz[pix], z1[pix])
    P = rays[pix] * z[:, None]
    idx, ok = grid.index(P)
    idx, pix = idx[ok], pix[ok]
    occ = np.zeros(grid.dims, bool)
    colors = np.zeros(grid.dims + (3,), np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    colors[idx[:, 0], idx[:, 1], idx[:, 2]] = rgb[vs[pix], us[pix]]
    return occ, colors, grid


def compute_observed(occ: np.ndarray, grid: Grid, front_points: np.ndarray) -> np.ndarray:
    """observed = occupied voxel within 1 voxel (26-neighborhood) of a visible front-surface point.
    The dilation runs only on the bounding box of the hit voxels (+1), not the whole grid."""
    out = np.zeros(occ.shape, bool)
    idx, ok = grid.index(front_points)
    idx = idx[ok]
    if len(idx) == 0:
        return out
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, np.array(occ.shape))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    hit = np.zeros(tuple(hi - lo), bool)
    rel = idx - lo
    hit[rel[:, 0], rel[:, 1], rel[:, 2]] = True
    out[sl] = ndimage.binary_dilation(hit, STRUCT26) & occ[sl]
    return out


def mask_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.nonzero(mask)
    return us, vs


def ray_pixels(mask: np.ndarray, Z_f: np.ndarray, K: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask pixels on a regular stride so the pixel footprint * stride <= voxel/2 (every voxel column still gets
    a ray). Pixels are often far finer than voxels (real duck at estimate scale: ~6 px per voxel per axis), and
    ray filling cost is linear in rays. Stride is 1 for the phantoms (footprint 0.08 cm vs 0.2 cm voxels)."""
    foot = float(np.median(Z_f[mask])) / K[0, 0]
    k = max(1, int(np.floor(0.5 * voxel / max(foot, 1e-9))))
    if k == 1:
        return mask_pixels(mask)
    sub = np.zeros_like(mask)
    sub[::k, ::k] = mask[::k, ::k]
    return mask_pixels(sub)


def contour_band(mask: np.ndarray, band_px: int = 3) -> np.ndarray:
    """Mask pixels within band_px of the mask edge."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return mask & (dt <= band_px)


def ransac_plane(P: np.ndarray, thresh: float, iters: int = 400, seed: int = 0):
    """Plane n.x + d = 0 with |n|=1, n_z > 0. Vectorized RANSAC + SVD refine on inliers.
    Returns (n, d, inlier_mask, signed_residuals)."""
    rng = np.random.default_rng(seed)
    N = len(P)
    tri = rng.integers(0, N, size=(iters, 3))
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 1e-9
    nrm = nrm[good] / ln[good, None]
    d = -np.einsum("ij,ij->i", nrm, a[good])
    # chunk to keep (iters x N) residual matrix small
    best, best_cnt = None, -1
    for s in range(0, len(nrm), 100):
        r = np.abs(P @ nrm[s:s + 100].T + d[s:s + 100])
        cnt = (r < thresh).sum(0)
        j = int(np.argmax(cnt))
        if cnt[j] > best_cnt:
            best_cnt, best = cnt[j], s + j
    inl = np.abs(P @ nrm[best] + d[best]) < thresh
    for _ in range(2):  # refine
        cen = P[inl].mean(0)
        _, _, Vt = np.linalg.svd(P[inl] - cen, full_matrices=False)
        n = Vt[-1]
        if n[2] < 0:
            n = -n
        dd = -n @ cen
        res = P @ n + dd
        inl = np.abs(res) < thresh
    return n, dd, inl, res


@dataclass
class Midplane:
    n: np.ndarray
    d: float
    band_uv: np.ndarray        # (M,2) band pixel coords
    residuals: np.ndarray      # (M,) signed cm
    inliers: np.ndarray        # (M,) bool
    fallback: bool

    @property
    def inlier_frac(self) -> float:
        return float(self.inliers.mean()) if len(self.inliers) else 0.0


def fit_midplane(Z_f, mask, K, band_px=3, thresh_cm=0.5) -> Midplane:
    band = contour_band(mask, band_px)
    us, vs = mask_pixels(band)
    P = unproject(K, us, vs, Z_f[vs, us].astype(np.float64))
    if len(P) < 10:
        z = float(np.median(Z_f[mask]))
        return Midplane(np.array([0, 0, 1.0]), -z, np.stack([us, vs], 1), np.zeros(len(P)), np.ones(len(P), bool), True)
    n, d, inl, res = ransac_plane(P, thresh_cm)
    # Degenerate if the plane is nearly parallel to the viewing rays (would blow up the ray
    # intersection): fall back to a fronto-parallel plane at the median band depth.
    if abs(n[2]) < 0.3:
        z = float(np.median(P[:, 2]))
        n, d = np.array([0, 0, 1.0]), -z
        res = P @ n + d
        return Midplane(n, d, np.stack([us, vs], 1), res, np.abs(res) < thresh_cm, True)
    return Midplane(n, d, np.stack([us, vs], 1), res, inl, False)


def ray_plane_z(K, us, vs, n, d) -> np.ndarray:
    """Planar depth where each pixel's ray meets plane n.x + d = 0."""
    rays = pixel_rays(K, us.astype(np.float64), vs.astype(np.float64))
    denom = rays @ n
    return -d / np.where(np.abs(denom) < 1e-6, 1e-6, denom)


def nearest_mask_pixel(mask: np.ndarray):
    """Returns (iy, ix) maps: for every pixel, the coords of the nearest mask pixel."""
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return iy, ix
