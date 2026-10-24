"""Synthetic phantoms with ground truth: meshes, camera, z-depth / mask / shaded RGB, occupancy, surface samples.

Object frame: x = length (duck faces +x), y = up (object sits on y=0), z = lateral (toward the camera when
side-on). Objects are unions of watertight parts. Open3D's compute_occupancy counts ray crossings (parity),
so overlapping parts must be tested separately and OR-ed; raycasting all parts together gives the union's
first hit correctly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import trimesh

from pipeline.intrinsics import build_K, pixel_rays
from pipeline.shapes.common import Grid, VOXEL_CM

W, H, HFOV, DIST = 1024, 768, 69.0, 60.0
YELLOW = np.array([235, 190, 40], np.float64)
GRAY = np.array([128, 128, 128], np.float64)


def _ellipsoid(center, semi, sub=4):
    m = trimesh.creation.icosphere(subdivisions=sub)
    m.apply_scale(semi)
    m.apply_translation(center)
    return m


def duck_parts():
    body = _ellipsoid((0, 3.5, 0), (5.0, 3.5, 3.0))
    head = _ellipsoid((3.6, 7.6, 0), (2.0, 2.0, 2.0))
    beak = trimesh.creation.cone(radius=0.8, height=1.8, sections=48)
    beak.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))  # +z -> +x
    beak.apply_scale((1.0, 0.45, 1.0))                                                       # flatten vertically
    beak.apply_translation((5.2, 7.4, 0))
    tail = _ellipsoid((0, 0, 0), (1.6, 1.0, 1.4), sub=3)
    tail.apply_transform(trimesh.transformations.rotation_matrix(math.radians(30), (0, 0, 1)))
    tail.apply_translation((-4.6, 5.2, 0))
    return [body, head, beak, tail]


def asym_duck_parts():
    handle = trimesh.creation.torus(major_radius=1.5, minor_radius=0.35, major_sections=48, minor_sections=16)
    handle.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))  # ring in y-z plane
    handle.apply_translation((-0.5, 3.8, 3.9))                                               # camera-facing flank
    return duck_parts() + [handle]


def bottle_parts():
    prof = np.array([[0, 0], [3.4, 0], [3.4, 15], [2.6, 17.5], [1.4, 19.5], [1.3, 24], [0, 24]], float)
    m = trimesh.creation.revolve(prof, sections=96)          # 2D y -> 3D z
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, (1, 0, 0)))  # z -> y
    return [m]


def _prism(tri_xy, z_half):
    """Triangle in the x-y plane extruded symmetrically along z (watertight, outward winding)."""
    t = np.asarray(tri_xy, float)
    V = np.vstack([np.c_[t, np.full(3, -z_half)], np.c_[t, np.full(3, z_half)]])
    F = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    m = trimesh.Trimesh(V, F, process=True)
    if m.volume < 0:
        m.invert()
    return m


def fish_parts():
    body = _ellipsoid((0, 2.8, 0), (6.0, 2.8, 1.6))
    tail = _prism([(-5.5, 2.8), (-8.5, 5.2), (-8.5, 0.6)], 0.3)
    dorsal = _prism([(-2.0, 5.2), (1.5, 5.3), (-1.5, 6.8)], 0.25)
    return [body, tail, dorsal]


def car_parts():
    body = trimesh.creation.box((12.0, 3.5, 5.5), trimesh.transformations.translation_matrix((0, 1.2 + 1.75, 0)))
    cabin = trimesh.creation.box((6.0, 2.7, 4.6), trimesh.transformations.translation_matrix((-0.5, 4.6 + 1.35, 0)))
    wheels = []
    for x in (-3.8, 3.8):
        for z in (-2.6, 2.6):
            w = trimesh.creation.cylinder(radius=1.4, height=1.0, sections=48)   # axis along z
            w.apply_translation((x, 1.4, z))
            wheels.append(w)
    return [body, cabin] + wheels


def teapot_parts():
    body = _ellipsoid((0, 3.8, 0), (4.5, 3.8, 4.5))
    handle = trimesh.creation.torus(major_radius=2.0, minor_radius=0.45, major_sections=48, minor_sections=16)  # ring in x-y
    handle.apply_translation((-5.0, 4.2, 0))
    knob = _ellipsoid((0, 7.9, 0), (0.8, 0.8, 0.8), sub=3)
    return [body, handle, knob]


PHANTOMS = {"duck": duck_parts, "bottle": bottle_parts, "asym_duck": asym_duck_parts,
            "fish": fish_parts, "car": car_parts, "teapot": teapot_parts}
HELDOUT = ("fish", "car", "teapot")


@dataclass
class Render:
    name: str
    K: np.ndarray
    R: np.ndarray            # world -> camera rotation (rows: right, down, forward)
    C: np.ndarray            # camera center, world
    parts: list              # trimesh parts in world frame (after yaw)
    Z: np.ndarray            # (H,W) planar depth cm, inf where no hit
    mask: np.ndarray         # (H,W) bool
    rgb: np.ndarray          # (H,W,3) uint8 shaded
    Z_table: np.ndarray | None = None   # (H,W) planar depth of the table plane (inf where not visible)
    marker: dict | None = None          # {"center_xz", "cm"} in world coords when a marker is rendered

    def to_cam(self, P):
        return (P - self.C) @ self.R.T

    def to_world(self, P):
        return P @ self.R + self.C


def _scene(parts):
    s = o3d.t.geometry.RaycastingScene()
    for m in parts:
        s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))))
    return s


def look_at(C, target, up=(0, 1, 0)):
    f = target - C
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


def _marker_bits():
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 6)


def _table_color(P, marker_xz, marker_cm, card_cm):
    """Table texture at world points P (N,3) on y=0: beige with low-frequency noise, white card, ArUco cells.
    Marker row 0 (top) is the far edge (-z), col 0 is -x."""
    x, z = P[:, 0], P[:, 2]
    noise = (np.sin(0.9 * x + 0.3) * np.cos(0.7 * z) + 0.5 * np.sin(2.3 * x - 1.7 * z + 1.1)
             + 0.25 * np.sin(5.1 * x + 4.3 * z))
    col = np.array([170.0, 150.0, 120.0])[None] * (1 + 0.07 * noise[:, None])
    mx, mz = marker_xz
    card = (np.abs(x - mx) <= card_cm / 2) & (np.abs(z - mz) <= card_cm / 2)
    col[card] = 238.0
    cell = marker_cm / 6
    inm = (np.abs(x - mx) < marker_cm / 2) & (np.abs(z - mz) < marker_cm / 2)
    bits = _marker_bits()
    r = np.clip(((z[inm] - (mz - marker_cm / 2)) / cell).astype(int), 0, 5)
    c = np.clip(((x[inm] - (mx - marker_cm / 2)) / cell).astype(int), 0, 5)
    col[inm] = np.where(bits[r, c][:, None] > 0, 238.0, 12.0)
    return col


def render(name: str, elevation_deg=0.0, yaw_deg=0.0, dist=DIST, w=W, h=H, hfov=HFOV, ssaa=2,
           table=False, marker_offset_z=9.0, marker_cm=6.0, card_cm=8.0) -> Render:
    parts = [p.copy() for p in PHANTOMS[name]()]
    lo = np.min([p.bounds[0] for p in parts], 0)
    hi = np.max([p.bounds[1] for p in parts], 0)
    center = (lo + hi) / 2
    if yaw_deg:
        Ry = trimesh.transformations.rotation_matrix(math.radians(yaw_deg), (0, 1, 0), point=center)
        for p in parts:
            p.apply_transform(Ry)
    e = math.radians(elevation_deg)
    C = center + dist * np.array([0, math.sin(e), math.cos(e)])
    R = look_at(C, center)
    K = build_K(w, h, hfov)
    scene = _scene(parts)

    def cast(Kc, ww, hh):
        vs, us = np.mgrid[0:hh, 0:ww]
        d_cam = pixel_rays(Kc, us.ravel().astype(float), vs.ravel().astype(float))   # z = 1
        d_w = d_cam @ R                                                               # R^T d
        rays = np.concatenate([np.broadcast_to(C, d_w.shape), d_w], 1).astype(np.float32)
        out = scene.cast_rays(o3d.core.Tensor(rays))
        t = out["t_hit"].numpy().reshape(hh, ww)                                      # == planar depth
        n = out["primitive_normals"].numpy().reshape(hh, ww, 3)
        dw = d_w.reshape(hh, ww, 3)
        with np.errstate(divide="ignore"):
            tp = np.where(dw[..., 1] < -1e-9, -C[1] / dw[..., 1], np.inf) if table else np.full((hh, ww), np.inf)
        return t, n, dw, tp

    Z, _, _, Ztab = cast(K, w, h)
    marker = {"center_xz": (center[0], center[2] + marker_offset_z), "cm": marker_cm} if table else None
    if table:
        Z = np.where(Z < Ztab, Z, np.inf)   # the object sits ON the table; table never occludes it here
    # shaded RGB, supersampled for anti-aliased edges (the depth model sees this)
    Ks = K.copy()
    Ks[:2] *= ssaa
    Ks[0, 2], Ks[1, 2] = (w * ssaa - 1) / 2, (h * ssaa - 1) / 2
    t2, n2, d2, tp2 = cast(Ks, w * ssaa, h * ssaa)
    hit = np.isfinite(t2) & (t2 < tp2)
    n2 = np.where((np.sum(n2 * d2, -1) > 0)[..., None], -n2, n2)                    # face the camera
    light = R.T @ np.array([-0.4, -0.6, -0.7])                                        # upper-left, from camera side
    light /= np.linalg.norm(light)
    lam = np.clip(n2 @ light, 0, 1)
    img = np.where(hit[..., None], YELLOW * (0.25 + 0.75 * lam[..., None]), GRAY)
    if table:
        tab = ~hit & np.isfinite(tp2)
        P = C + d2[tab] * tp2[tab][:, None]
        img[tab] = _table_color(P, marker["center_xz"], marker_cm, card_cm)
    rgb = cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)
    return Render(name, K, R, C, parts, Z, np.isfinite(Z), rgb, np.where(np.isfinite(Ztab) & ~np.isfinite(Z), Ztab, np.inf)
                  if table else None, marker)


def gt_grid(r: Render, voxel=VOXEL_CM) -> Grid:
    V = np.concatenate([r.to_cam(p.vertices) for p in r.parts])
    return Grid.around(V.min(0), V.max(0), voxel, margin=1)


def gt_occupancy(r: Render, grid: Grid) -> np.ndarray:
    Pw = r.to_world(grid.centers()).astype(np.float32)
    occ = np.zeros(len(Pw), bool)
    q = o3d.core.Tensor(Pw)
    for p in r.parts:   # per part, OR: parity-based occupancy breaks on overlapping parts
        occ |= _scene([p]).compute_occupancy(q, nsamples=3).numpy().astype(bool)
    return occ.reshape(grid.dims)


def gt_surface_points(r: Render, n=40000, seed=0) -> np.ndarray:
    """Points on the union surface (camera frame): sample each part, drop samples inside another part."""
    rng = np.random.default_rng(seed)
    areas = np.array([p.area for p in r.parts])
    out = []
    for i, p in enumerate(r.parts):
        pts, _ = trimesh.sample.sample_surface(p, int(n * areas[i] / areas.sum()), seed=int(rng.integers(1 << 31)))
        keep = np.ones(len(pts), bool)
        q = o3d.core.Tensor(pts.astype(np.float32))
        for j, o in enumerate(r.parts):
            if j != i:
                keep &= ~_scene([o]).compute_occupancy(q, nsamples=3).numpy().astype(bool)
        out.append(pts[keep])
    return r.to_cam(np.concatenate(out))
