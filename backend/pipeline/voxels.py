"""Voxel cleanup, world-frame conversion, measurements, payload."""
from __future__ import annotations

import logging

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .shapes.common import STRUCT26, ShapeResult, compute_observed

log = logging.getLogger("pipeline.voxels")
STRUCT6 = ndimage.generate_binary_structure(3, 1)


def largest_component(occ: np.ndarray, structure=STRUCT6) -> np.ndarray:
    lab, n = ndimage.label(occ, structure)
    if n <= 1:
        return occ
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def fix_edge_contacts(occ: np.ndarray, max_iter: int = 10) -> tuple[np.ndarray, int]:
    """Voxels touching only along an edge give non-manifold (4-face) mesh edges -> not watertight.
    Fill one of the two empty cells of every diagonal 2x2 pattern until none remain."""
    occ = occ.copy()
    added = 0
    for _ in range(max_iter):
        changed = 0
        for a, b in ((0, 1), (0, 2), (1, 2)):
            o = np.moveaxis(occ, (a, b), (0, 1))  # view: writes go through
            p00, p11, p10, p01 = o[:-1, :-1], o[1:, 1:], o[1:, :-1], o[:-1, 1:]
            diag = p00 & p11 & ~p10 & ~p01
            anti = p10 & p01 & ~p00 & ~p11
            if diag.any():
                o[1:, :-1] |= diag
                changed += int(diag.sum())
            if anti.any():
                o[:-1, :-1] |= anti
                changed += int(anti.sum())
        added += changed
        if not changed:
            break
    return occ, added


def cleanup(occ: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """binary open then close (3x3x3), keep largest component, fix edge contacts, fill colors of new voxels.
    Works on the occupied bounding box (+margin) and pastes back: same result, much less morphology work."""
    nz = np.argwhere(occ)
    if len(nz):
        lo = np.maximum(nz.min(0) - 3, 0)
        hi = np.minimum(nz.max(0) + 4, np.array(occ.shape))
        if np.prod(hi - lo) < 0.8 * occ.size:
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            o2, c2, info = cleanup(occ[sl], colors[sl])
            out_o = np.zeros_like(occ)
            out_c = np.zeros_like(colors)
            out_o[sl], out_c[sl] = o2, c2
            return out_o, out_c, info
    info = {"in": int(occ.sum())}
    p = np.pad(occ, 2)
    opened = ndimage.binary_opening(p, STRUCT26)
    if opened.sum() < 0.2 * p.sum():   # opening ate a thin object: skip it rather than delete it
        opened, info["open_skipped"] = p, True
    closed = ndimage.binary_closing(opened, STRUCT26)[2:-2, 2:-2, 2:-2]
    out = largest_component(closed)
    out, info["edge_fix_added"] = fix_edge_contacts(out)
    out = largest_component(out)
    colored = occ & out
    if (out & ~colored).any() and colored.any():
        _, (ii, jj, kk) = ndimage.distance_transform_edt(~colored, return_indices=True)
        colors = colors[ii, jj, kk]
    colors = np.where(out[..., None], colors, 0).astype(np.uint8)
    info["out"] = int(out.sum())
    return out, colors, info


def surface_mask(occ: np.ndarray) -> np.ndarray:
    """Voxels with at least one empty 6-neighbor."""
    return occ & ~ndimage.binary_erosion(occ, STRUCT6, border_value=0)


@dataclass
class VoxelModel:
    """World frame: x right, y up, z toward viewer. Object on y=0, bbox centered in x/z."""
    occ: np.ndarray
    colors: np.ndarray
    observed: np.ndarray
    voxel_cm: float
    origin_cm: np.ndarray     # world position of the min corner of voxel (0,0,0)
    method: str
    thickness: float
    cleanup_info: dict
    scale_source: str = "estimate"
    extras: dict = field(default_factory=dict)   # "plane" (mirror) / "axis" (revolve), world frame, mm

    @property
    def measurements_cm(self) -> dict:
        n = np.array(self.occ.shape) * self.voxel_cm
        return {"w": round(float(n[0]), 1), "h": round(float(n[1]), 1), "d": round(float(n[2]), 1)}

    def surface_voxels(self):
        """(ijk (N,3), colors (N,3), observed (N,)) for surface voxels only."""
        s = surface_mask(self.occ)
        ijk = np.argwhere(s)
        return ijk, self.colors[s], self.observed[s]

    def payload(self) -> dict:
        ijk, col, obs = self.surface_voxels()
        return {
            "voxel_size_mm": round(self.voxel_cm * 10, 4),
            "origin_mm": [round(float(x) * 10, 3) for x in self.origin_cm],
            "dims": [int(x) for x in self.occ.shape],
            "voxels": ijk.astype(int).ravel().tolist(),
            "colors": col.astype(int).ravel().tolist(),
            "observed": obs.astype(int).tolist(),
            "measurements_cm": self.measurements_cm,
            "method": self.method,
            "thickness": self.thickness,
            "scale_source": self.scale_source,
            "n_solid": int(self.occ.sum()),
            **self.extras,
        }


def clean_shape(res: ShapeResult):
    """cleanup + (for symmetric methods) re-symmetrize + edge-contact fix, in the result's own grid."""
    occ, colors, info = cleanup(res.occ, res.colors)
    if "symmetric_plane" in res.debug:
        # Morphological cleanup isn't reflection-equivariant on a grid that isn't aligned with an oblique plane
        # (real duck: 0.98 -> 0.93). Re-symmetrize: union with the exact reflection, colors from the source.
        from .shapes.symmetry import reflect_union, self_symmetry_iou
        n, d = res.debug["symmetric_plane"]
        occ, colors = reflect_union(occ, colors, res.grid, n, d)
        # the union can recreate edge-only contacts (non-manifold blocky mesh): fix them again
        fixed, added = fix_edge_contacts(occ)
        if added:
            new = fixed & ~occ
            occ = fixed
            if new.any():
                from scipy import ndimage as _nd
                nn = np.argwhere(new)
                lo = np.maximum(nn.min(0) - 4, 0)
                hi = np.minimum(nn.max(0) + 5, np.array(occ.shape))
                sl = tuple(slice(a, b) for a, b in zip(lo, hi))
                src = occ[sl] & ~new[sl]
                if src.any():
                    _, (ii, jj, kk) = _nd.distance_transform_edt(~src, return_indices=True)
                    sub = colors[sl]
                    colors[sl] = np.where(new[sl][..., None], sub[ii, jj, kk], sub)
        info["edge_fix_after_symmetrize"] = int(added)
        info["self_symmetry_iou"] = round(self_symmetry_iou(occ, res.grid, n, d), 4)
        if info["self_symmetry_iou"] < 0.97:
            log.warning("mirror result only %.3f self-symmetric", info["self_symmetry_iou"])
    return occ, colors, info


def finalize(res: ShapeResult) -> VoxelModel:
    occ, colors, info = clean_shape(res)
    observed = compute_observed(occ, res.grid, res.front_points)
    centroid_cam = res.grid.centers(np.argwhere(occ)).mean(0) if occ.any() else np.zeros(3)
    v = res.grid.voxel
    o = res.grid.origin
    dims = np.array(res.grid.dims)
    if res.grid.R is None:
        # camera (x right, y down, z fwd) -> world (x right, y up, z to viewer): rotate 180 deg about x
        occ, colors, observed = occ[:, ::-1, ::-1], colors[:, ::-1, ::-1], observed[:, ::-1, ::-1]
        origin_w = np.array([o[0], -o[1] - dims[1] * v, -o[2] - dims[2] * v])
    else:
        origin_w = o.copy()   # grid already in the gravity-aligned world frame
    # crop to occupied bbox
    nz = np.argwhere(occ)
    if len(nz) == 0:
        raise ValueError("empty voxel model after cleanup")
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    occ, colors, observed = occ[sl], colors[sl], observed[sl]
    cropped_origin = origin_w + lo * v
    size = (hi - lo) * v
    # sit on y=0, center x/z
    final_origin = np.array([-size[0] / 2, 0.0, -size[2] / 2])
    # world = R @ cam + T   (R = diag(1,-1,-1) when the grid was in the camera frame)
    info["world_T_cm"] = (final_origin - cropped_origin).tolist()
    Rw = np.diag([1.0, -1, -1]) if res.grid.R is None else np.asarray(res.grid.R)
    info["world_R"] = Rw.tolist()
    Tw = np.asarray(info["world_T_cm"])
    to_w = lambda p: [round(float(x) * 10, 2) for x in (Rw @ p + Tw)]   # noqa: E731  cam (cm) -> world (mm)
    extras = {}
    if "symmetric_plane" in res.debug:
        n, d = res.debug["symmetric_plane"]
        c = centroid_cam - (centroid_cam @ n + d) * n          # object centroid projected onto the plane
        extras["plane"] = {"point": to_w(c), "normal": [round(float(x), 5) for x in Rw @ n],
                           "offset_cm": res.debug.get("plane_offset_cm", 0.0)}
    if "axis" in res.debug:
        ax = res.debug["axis"]
        A0, dd, s_ = np.asarray(ax["A0"]), np.asarray(ax["d"]), np.asarray(ax["s"])
        extras["axis"] = {"p0": to_w(A0 + dd * s_[0]), "p1": to_w(A0 + dd * s_[-1])}
    return VoxelModel(occ, colors, observed, v, final_origin, res.method, res.thickness, info, extras=extras)
