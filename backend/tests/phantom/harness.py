"""Phantom ground truth + evaluation."""
from __future__ import annotations

import time

import numpy as np
import open3d as o3d

from pipeline import mesh
from pipeline.intrinsics import pixel_rays
from pipeline.segment import clean_mask
from pipeline.shapes.common import Midplane, ShapeResult, fill_rays, ray_pixels
from pipeline.voxels import cleanup, finalize

from .metrics import chamfer_mm, iou, iou_centered
from .phantoms import _scene, gt_grid, gt_occupancy, gt_surface_points, render


class GT:
    def __init__(self, r):
        self.r = r
        self.grid = gt_grid(r)
        self.occ = gt_occupancy(r, self.grid)
        self.pts = gt_surface_points(r)
        alpha = (r.mask * 255).astype(np.uint8)
        self.mask, self.mask_full = clean_mask(alpha, 2)
        # morph-close can add gap pixels (e.g. the head/beak notch) with no surface behind them. A real
        # depth map is finite everywhere, so restrict to pixels that actually hit the object.
        self.mask &= r.mask
        self.mask_full &= r.mask

    @property
    def Z(self) -> np.ndarray:
        """GT planar depth with 0 off the object (what shape methods expect)."""
        return np.where(self.r.mask, self.r.Z, 0).astype(np.float32)

    def oracle_midplane(self) -> Midplane:
        """True symmetry plane (object z=0) in the camera frame, as a Midplane for mirror/inflate."""
        n = self.r.R @ np.array([0, 0, 1.0])
        p0 = self.r.to_cam(np.array([[0, 3.5, 0.0]]))[0]
        if n[2] < 0:
            n = -n
        return Midplane(n, float(-n @ p0), np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool), False)

    def back_depth(self, us, vs, far=200.0) -> np.ndarray:
        """Planar depth where each pixel ray last leaves the object (cast from behind, toward the camera)."""
        r = self.r
        d_w = pixel_rays(r.K, us.astype(float), vs.astype(float)) @ r.R     # z=1 camera rays, world frame
        rays = np.concatenate([r.C + far * d_w, -d_w], 1).astype(np.float32)
        t = _scene(r.parts).cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        return far - t                                                        # t is in units of |d_w|, z=1

    def oracle_fill(self, voxel=None, R=None) -> ShapeResult:
        """Front-to-back ray fill with the TRUE back surface: exercises Grid/fill_rays/ray_pixels/cleanup/finalize
        on real geometry with no shape assumption. Union-of-rays, so it can only miss along-ray concavities."""
        from pipeline.shapes.common import VOXEL_CM
        voxel = voxel or VOXEL_CM
        us, vs = ray_pixels(self.mask, self.Z, self.r.K, voxel)
        zf, zb = self.r.Z[vs, us].astype(np.float64), self.back_depth(us, vs)
        occ, colors, grid = fill_rays(self.r.K, us, vs, zf, zb, self.r.rgb, voxel, R=R)
        front = pixel_rays(self.r.K, us.astype(float), vs.astype(float)) * zf[:, None]
        return ShapeResult(occ, occ.copy(), colors, grid, front, "oracle", 1.0)


_gt_cache: dict = {}


def get_gt(name, elev=0.0, yaw=0.0, **kw) -> GT:
    k = (name, elev, yaw, tuple(sorted(kw.items())))
    if k not in _gt_cache:
        _gt_cache[k] = GT(render(name, elev, yaw, **kw))
    return _gt_cache[k]


def evaluate(gt: GT, res: ShapeResult, mesh_check=True) -> dict:
    """IoU (raw + centred) and Chamfer vs GT after cleanup; optionally build both STLs and check watertightness.
    Report centred IoU for anything that went through DA depth: a few % scale error is a cm-scale shift."""
    t0 = time.perf_counter()
    occ, _, _ = cleanup(res.occ, res.colors)
    out = {"method": res.method, "t": res.thickness, "iou": iou(occ, res.grid, gt.occ, gt.grid),
           "iou_c": iou_centered(occ, res.grid, gt.occ, gt.grid),
           "chamfer_mm": chamfer_mm(occ, res.grid, gt.pts), "cleanup_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if mesh_check:
        model = finalize(res)
        out["blocky_watertight"] = bool(mesh.blocky_mesh(model).is_watertight)
        out["smooth_watertight"] = mesh.smooth_stl(model)[2]["watertight"]
        out["model"] = model
    return out
