"""Symmetry-plane estimators for mirror v2. Each returns ((n, d), info): camera frame, plane n.X + d = 0,
|n| = 1, n_z > 0 (n points away from the camera).

  energy : (mirror v2 default) n within ±25° of an initial normal + offset, minimizing
           w_occ E_occ + w_sil E_sil + w_rim E_rim + w_prior E_prior on REFLECTED front points.
  rim    : rim RANSAC plane (old M0)                             -> baseline B1 with exact reflection
  B2     : rim orientation, offset = P95 front depth along n + k * median inflate half-thickness
  M1     : rim orientation, offset through Z_f + h_inflate

All energy terms are normalized by the object size L (cm, or L_px in pixels) so the weights are scale-free."""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import minimize

from ..intrinsics import pixel_rays, unproject
from .midplane import inflate_h, midplane_variant

DEFAULT_ESTIMATOR = "energy"
W = {"occ": 1.0, "sil": 0.3, "rim": 0.3, "prior": 3.0}
B2_K = 0.5
TILT_MAX_DEG = 25.0
OCC_TOL_FRAC = 0.03       # E_occ tolerance = 3 % of object size (fixed a priori)
SIL_DILATE_FRAC = 0.02    # silhouette slack = 2 % of object size in px (fixed a priori)
HUBER_FRAC = 0.05         # robust scale for E_rim / E_prior = 5 % of object size (fixed a priori)
MAX_PTS = 1200
MAX_RIM = 400


def _nd(mid):
    n, d = np.asarray(mid.n, float), float(mid.d)
    if n[2] < 0:
        n, d = -n, -d
    return n / np.linalg.norm(n), d / np.linalg.norm(mid.n)


def _huber(x, delta):
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def _basis(n0):
    a = np.array([1.0, 0, 0]) if abs(n0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n0, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n0, e1)


class PlaneProblem:
    """Precomputed data for fast batched energy evaluation."""

    def __init__(self, Z_f, mask, mask_full, K, n0, up_cam=None, seed=0, prior_scale=1.0):
        self.K = K
        H, W_ = Z_f.shape
        self.shape = (H, W_)
        rng = np.random.default_rng(seed)
        us, vs, h = inflate_h(Z_f, mask, mask_full, K)
        if len(us) > MAX_PTS:
            s = rng.choice(len(us), MAX_PTS, replace=False)
            us, vs, h = us[s], vs[s], h[s]
        self.zf_pts = Z_f[vs, us].astype(np.float64)
        self.rays = pixel_rays(K, us.astype(float), vs.astype(float))
        self.P = self.rays * self.zf_pts[:, None]
        self.h = h * prior_scale
        ys, xs = np.nonzero(mask_full)
        self.L_px = float(max(np.ptp(xs), np.ptp(ys)) + 1)
        self.L = self.L_px * float(np.median(self.zf_pts)) / K[0, 0]
        self.Zimg = Z_f.astype(np.float64)
        self.mask = mask
        r = max(1, int(round(SIL_DILATE_FRAC * self.L_px)))
        dil = cv2.dilate(mask_full.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        self.dt_out = cv2.distanceTransform((1 - dil).astype(np.uint8), cv2.DIST_L2, 5)
        self.tol = OCC_TOL_FRAC * self.L
        self.delta = HUBER_FRAC * self.L
        self.R = self._trusted_rim(Z_f, mask_full, K, rng)
        self.n0 = n0 / np.linalg.norm(n0)
        self.up = None if up_cam is None else up_cam / np.linalg.norm(up_cam)
        if self.up is not None:
            self.e1 = np.cross(self.up, self.n0)   # rotate about up only: plane stays perpendicular to the table
            self.e1 /= np.linalg.norm(self.e1)
            self.e2 = None
        else:
            self.e1, self.e2 = _basis(self.n0)
        self.c0 = np.median(self.P, axis=0)

    def _trusted_rim(self, Z_f, mask_full, K, rng):
        """Outer edge band of the uneroded mask, minus steep-gradient pixels and background-bled pixels."""
        dt_in = cv2.distanceTransform(mask_full.astype(np.uint8), cv2.DIST_L2, 5)
        dt_out = cv2.distanceTransform((~mask_full).astype(np.uint8), cv2.DIST_L2, 5)
        band = mask_full & (dt_in <= 2.0)
        Zf = Z_f.astype(np.float32)
        gx, gy = cv2.Sobel(Zf, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(Zf, cv2.CV_32F, 0, 1, ksize=3) / 8
        grad = np.hypot(gx, gy)
        inner = mask_full & (dt_in > 4)
        g_ref = np.median(grad[inner]) if inner.any() else np.median(grad[mask_full])
        steep = grad > 4 * max(g_ref, 1e-6)
        bg = (~mask_full & (dt_out >= 4) & (dt_out <= 10)).astype(np.float32)
        it = (mask_full & (dt_in >= 6) & (dt_in <= 12)).astype(np.float32)
        box = (25, 25)
        zbg = cv2.boxFilter(Zf * bg, -1, box, normalize=False) / np.maximum(cv2.boxFilter(bg, -1, box, normalize=False), 1e-6)
        zin = cv2.boxFilter(Zf * it, -1, box, normalize=False) / np.maximum(cv2.boxFilter(it, -1, box, normalize=False), 1e-6)
        keep = band & ~steep & (np.abs(Zf - zin) <= np.abs(Zf - zbg))
        vs, us = np.nonzero(keep)
        if len(us) > MAX_RIM:
            s = rng.choice(len(us), MAX_RIM, replace=False)
            us, vs = us[s], vs[s]
        self.rim_frac = float(keep.sum() / max(1, band.sum()))
        return unproject(K, us, vs, Z_f[vs, us].astype(np.float64)) if len(us) else np.zeros((0, 3))

    # ---- parametrization: x = (tilt_a, tilt_b, s) degrees/cm; plane passes through c0 + s*n
    def plane(self, x):
        a = np.radians(x[0])
        if self.up is not None:
            n = np.cos(a) * self.n0 + np.sin(a) * self.e1
        else:
            b = np.radians(x[1])
            n = self.n0 + np.tan(a) * self.e1 + np.tan(b) * self.e2
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        d = -float(n @ (self.c0 + x[2] * n))
        return n, d

    def terms(self, N, D):
        """Batched energy terms for planes N (k,3), D (k,). Returns dict of (k,) arrays."""
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        H, W_ = self.shape
        sd = self.P @ N.T + D[None, :]                                # (m,k) signed distance
        Pr = self.P[:, None, :] - 2 * sd[:, :, None] * N[None, :, :]  # (m,k,3) reflected front points
        Z = Pr[..., 2]
        good = Z > 1e-3
        Zs = np.where(good, Z, 1.0)
        u = fx * Pr[..., 0] / Zs + cx
        v = fy * Pr[..., 1] / Zs + cy
        ui = np.clip(np.round(u), 0, W_ - 1).astype(int)
        vi = np.clip(np.round(v), 0, H - 1).astype(int)
        off_img = np.hypot(np.maximum(0, np.maximum(-u, u - (W_ - 1))), np.maximum(0, np.maximum(-v, v - (H - 1))))
        # E_sil: reflected points must project inside the (slightly dilated) silhouette
        dsil = (self.dt_out[vi, ui] + off_img) / self.L_px
        dsil = np.where(good, dsil, 1.0)
        e_sil = np.mean(dsil ** 2, axis=0)
        # E_occ: a reflected point inside the mask but IN FRONT of the observed surface would have been seen
        inside = good & self.mask[vi, ui] & (off_img == 0)
        viol = np.maximum(0, self.Zimg[vi, ui] - self.tol - Z) * inside / self.L
        e_occ = np.mean(viol ** 2, axis=0)
        # E_rim: robust distance of trusted rim points to the plane
        if len(self.R):
            e_rim = np.mean(_huber((self.R @ N.T + D[None, :]) / self.L, HUBER_FRAC), axis=0)
        else:
            e_rim = np.zeros(len(D))
        # E_prior: half-thickness along each ray (plane hit - front) should match the inflate half-thickness
        den = self.rays @ N.T                                          # (m,k)
        zc = np.where(den > 1e-6, -D[None, :] / np.where(den > 1e-6, den, 1), self.zf_pts[:, None])
        e_prior = np.mean(_huber((zc - self.zf_pts[:, None] - self.h[:, None]) / self.L, HUBER_FRAC), axis=0)
        return {"occ": e_occ, "sil": e_sil, "rim": e_rim, "prior": e_prior}

    def energy(self, N, D, w):
        t = self.terms(N, D)
        return sum(w[k] * t[k] for k in w), t


def initial_normal(Z_f, mask, mask_full, K, up_cam=None):
    if up_cam is not None:
        z = np.array([0, 0, 1.0])
        up = up_cam / np.linalg.norm(up_cam)
        n0 = z - (z @ up) * up              # horizontal, perpendicular to the table normal, closest to the optical axis
        return n0 / np.linalg.norm(n0), "marker"
    n, _ = _nd(midplane_variant("M0", Z_f, mask, mask_full, K))
    return n, "rim"


CATEGORY_PRIOR = {"flat": 0.5, "bilateral": 1.0, "round": 1.0, "boxy": 1.27, "other": 1.0}
CATEGORY_PRIOR_ENABLED = True   # pre-registered rule passed: held-out mean IoU_c +0.041, worst 0.000 (car only)


def estimate_energy(Z_f, mask, mask_full, K, up_cam=None, w=None, prior_scale=1.0):
    t0 = time.perf_counter()
    w = w or W
    n0, src = initial_normal(Z_f, mask, mask_full, K, up_cam)
    pb = PlaneProblem(Z_f, mask, mask_full, K, n0, up_cam, prior_scale=prior_scale)
    L = pb.L
    tilts = np.linspace(-TILT_MAX_DEG, TILT_MAX_DEG, 7)
    offs = np.linspace(-0.4 * L, 0.8 * L, 25)
    if pb.up is not None:
        cand = [(a, 0.0, s) for a in tilts for s in offs]
    else:
        cand = [(a, b, s) for a in tilts for b in tilts for s in offs]
    planes = [pb.plane(x) for x in cand]
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    E = np.empty(len(D))
    for i in range(0, len(D), 256):                           # chunked batch evaluation
        E[i:i + 256] = pb.energy(N[i:i + 256], D[i:i + 256], w)[0]
    x0 = np.array(cand[int(np.argmin(E))])

    def f(x):
        x = np.array(x, float)
        x[0] = np.clip(x[0], -TILT_MAX_DEG, TILT_MAX_DEG)
        x[1] = np.clip(x[1], -TILT_MAX_DEG, TILT_MAX_DEG)
        n, d = pb.plane(x)
        return float(pb.energy(n[None], np.array([d]), w)[0][0])

    free = [0, 2] if pb.up is not None else [0, 1, 2]
    res = minimize(lambda z: f(np.array([z[0], 0.0, z[1]]) if pb.up is not None else z), x0[free], method="Powell",
                   options={"xtol": 1e-2, "ftol": 1e-6, "maxfev": 300})
    xb = np.array([res.x[0], 0.0, res.x[1]]) if pb.up is not None else res.x
    xb[:2] = np.clip(xb[:2], -TILT_MAX_DEG, TILT_MAX_DEG)
    n, d = pb.plane(xb)
    _, t = pb.energy(n[None], np.array([d]), w)
    info = {"estimator": "energy", "init_normal": src, "tilt_deg": [round(float(xb[0]), 2), round(float(xb[1]), 2)],
            "offset_from_median_front_cm": round(float(xb[2]), 3), "object_size_cm": round(L, 2),
            "terms": {k: round(float(v[0]), 6) for k, v in t.items()}, "weights": w,
            "trusted_rim_frac": round(pb.rim_frac, 3), "prior_scale": prior_scale,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return (n, d), info


def ratio_plane(n, Z_f, mask, mask_full, K, depth_ratio: float):
    """Keep orientation n; put the plane D_t/2 behind the nearest front point (P5 of n.P), where
    D_t = depth_ratio x silhouette height in cm (VLM estimate of depth / visible height). No fitted constants."""
    vs, us = np.nonzero(mask)
    P = unproject(K, us, vs, Z_f[vs, us].astype(np.float64))
    near = float(np.percentile(P @ n, 5))
    ys = np.nonzero(mask_full)[0]
    h_cm = float(np.ptp(ys) + 1) * float(np.median(Z_f[vs, us])) / K[1, 1]
    d_t = depth_ratio * h_cm
    return -(near + d_t / 2), {"depth_ratio": depth_ratio, "height_cm": round(h_cm, 2), "target_depth_cm": round(d_t, 2)}


def estimate_b2(Z_f, mask, mask_full, K, k=None):
    k = B2_K if k is None else k
    n, _ = _nd(midplane_variant("M0", Z_f, mask, mask_full, K))
    us, vs, h = inflate_h(Z_f, mask, mask_full, K)
    P = unproject(K, us, vs, Z_f[vs, us].astype(np.float64))
    s = float(np.percentile(P @ n, 95) + k * np.median(h))    # robust "deepest observed point" (P95, see step 1)
    return (n, -s), {"estimator": "B2", "k": k}


def estimate_plane(Z_f, mask, mask_full, K, estimator=None, up_cam=None, prior_scale=1.0, depth_ratio=None):
    est = estimator or DEFAULT_ESTIMATOR
    if est == "energy":
        (n, d), info = estimate_energy(Z_f, mask, mask_full, K, up_cam, prior_scale=prior_scale)
        if depth_ratio:
            d, info["depth_ratio_offset"] = ratio_plane(n, Z_f, mask, mask_full, K, float(depth_ratio))
        return (n, d), info
    if est == "rim":
        return _nd(midplane_variant("M0", Z_f, mask, mask_full, K)), {"estimator": "rim"}
    if est == "B2":
        return estimate_b2(Z_f, mask, mask_full, K)
    if est == "M1":
        return _nd(midplane_variant("M1", Z_f, mask, mask_full, K)), {"estimator": "M1"}
    raise ValueError(f"unknown plane estimator {est}")
