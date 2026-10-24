"""Tiny software voxel renderer (orthographic z-buffer splatting) working from the API payload.
Used for 4-angle eval renders and server-side thumbnails. World frame: x right, y up, z toward viewer."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

BRASS = np.array([200, 152, 60], np.float64)
STEEL = np.array([88, 104, 122], np.float64)

VIEWS = {  # (yaw about y, pitch about x) in degrees; camera looks along -z after rotation
    "front": (0, 0),
    "side": (-90, 0),     # from +x
    "top": (0, -90),      # from +y
    "3/4": (-35, -25),
}


def payload_to_dense(p: dict):
    dims = tuple(p["dims"])
    ijk = np.asarray(p["voxels"], int).reshape(-1, 3)
    occ = np.zeros(dims, bool)
    occ[tuple(ijk.T)] = True
    solid = ndimage.binary_fill_holes(occ)  # payload is surface-only; refill the interior
    return ijk, solid


def render(p: dict, view: str = "3/4", size: int = 360, mode: str = "photo", plane: bool = False) -> np.ndarray:
    """mode: 'photo' (per-voxel photo color) or 'observed' (brass observed / steel inferred). Returns HxWx3 uint8."""
    ijk, solid = payload_to_dense(p)
    if mode == "photo":
        col = np.asarray(p["colors"], np.float64).reshape(-1, 3)
    else:
        obs = np.asarray(p["observed"], bool)
        col = np.where(obs[:, None], BRASS, STEEL)
    # normals from the blurred solid
    g = np.gradient(ndimage.gaussian_filter(solid.astype(np.float32), 1.0))
    n = -np.stack([gi[tuple(ijk.T)] for gi in g], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    P = (ijk + 0.5) - np.array(solid.shape) / 2.0
    yaw, pitch = (math.radians(a) for a in VIEWS[view])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R = Rx @ Ry
    Q, N = P @ R.T, n @ R.T
    light = np.array([-0.35, 0.6, 0.72])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(N @ light, 0, 1)
    # keep only voxels facing the camera (+z)
    vis = N[:, 2] > -0.2
    Q, c, shade = Q[vis], col[vis], shade[vis]
    span = max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]), 1) + 2
    s = (size * 0.9) / span
    k = max(1, int(math.ceil(s * 1.1)))
    qmean = Q.mean(0)
    px = ((Q[:, 0] - qmean[0]) * s + size / 2).astype(int)
    py = (size / 2 - (Q[:, 1] - qmean[1]) * s).astype(int)
    off = np.arange(k) - k // 2
    dx, dy = np.meshgrid(off, off)
    X = (px[:, None] + dx.ravel()).ravel()
    Y = (py[:, None] + dy.ravel()).ravel()
    Z = np.repeat(-Q[:, 2], k * k)            # smaller = nearer
    idx = np.repeat(np.arange(len(Q)), k * k)
    ok = (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
    X, Y, Z, idx = X[ok], Y[ok], Z[ok], idx[ok]
    pix = Y * size + X
    o = np.lexsort((Z, pix))
    first = np.unique(pix[o], return_index=True)[1]
    sel = o[first]
    img = np.full((size * size, 3), 245.0)
    img[pix[sel]] = c[idx[sel]] * shade[idx[sel], None]
    if plane and p.get("plane"):
        # translucent brass sheet + solid rim, z-tested against the voxels (index space, same projection)
        zbuf = np.full(size * size, np.inf)
        zbuf[pix[sel]] = Z[sel]
        vs_mm = p["voxel_size_mm"]
        c0 = (np.asarray(p["plane"]["point"]) - np.asarray(p["origin_mm"])) / vs_mm - np.array(solid.shape) / 2.0
        nrm = np.asarray(p["plane"]["normal"], float)
        a = np.cross(nrm, [0, 1, 0] if abs(nrm[1]) < 0.9 else [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(nrm, a)
        half = 0.62 * max(solid.shape)
        g = np.linspace(-half, half, 220)
        A, B = np.meshgrid(g, g)
        rim = (np.abs(A) > half * 0.985) | (np.abs(B) > half * 0.985)
        pts = c0 + A.reshape(-1, 1) * a + B.reshape(-1, 1) * b
        qp = pts @ R.T
        X2 = ((qp[:, 0] - qmean[0]) * s + size / 2).astype(int)
        Y2 = (size / 2 - (qp[:, 1] - qmean[1]) * s).astype(int)
        ok2 = (X2 >= 0) & (X2 < size) & (Y2 >= 0) & (Y2 < size)
        pp = Y2[ok2] * size + X2[ok2]
        front = -qp[ok2, 2] < zbuf[pp]
        rr = rim.reshape(-1)[ok2]
        brass = np.array([226, 186, 100.0])
        sheet = np.unique(pp[front & ~rr])
        img[sheet] = 0.72 * img[sheet] + 0.28 * brass
        img[pp[front & rr]] = brass * 0.8
    return img.reshape(size, size, 3).clip(0, 255).astype(np.uint8)


def render_grid(p: dict, size: int = 300, plane: bool = False) -> np.ndarray:
    """2 rows (photo colors / observed-inferred) x 4 views."""
    rows = [np.hstack([render(p, v, size, m, plane) for v in VIEWS]) for m in ("photo", "observed")]
    return np.vstack(rows)
