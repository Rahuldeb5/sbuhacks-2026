"""STL export. Input VoxelModel is world frame (y up, cm); STL is mm and Z-UP (slicer convention):
(x, y, z)_world -> (x, -z, y)_stl, a proper rotation (det +1) so winding stays outward."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from .voxels import VoxelModel

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def _to_stl_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_transform(Y_UP_TO_Z_UP)
    return mesh


def blocky_mesh(model: VoxelModel) -> trimesh.Trimesh:
    """Cube faces only between solid and empty cells; shared corners merged."""
    occ = np.pad(model.occ, 1)
    verts, faces = [], []
    nv = 0
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3   # e_b x e_c = e_a
        for sign in (1, -1):
            nb = np.roll(occ, -sign, axis=a)
            ijk = np.argwhere(occ & ~nb)   # pad guarantees roll never wraps solid onto solid
            if len(ijk) == 0:
                continue
            base = ijk.astype(np.float64) - 1  # undo pad
            base[:, a] += 1 if sign > 0 else 0
            q = np.zeros((len(ijk), 4, 3))
            for n, (db, dc) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                q[:, n] = base
                q[:, n, b] += db
                q[:, n, c] += dc
            f = np.array([[0, 1, 2], [0, 2, 3]]) if sign > 0 else np.array([[0, 2, 1], [0, 3, 2]])
            idx = nv + np.arange(len(ijk))[:, None, None] * 4 + f[None]
            verts.append(q.reshape(-1, 3))
            faces.append(idx.reshape(-1, 3))
            nv += len(ijk) * 4
    V = np.concatenate(verts) * model.voxel_cm * 10 + model.origin_cm * 10
    m = trimesh.Trimesh(V, np.concatenate(faces), process=True)  # process=True merges duplicate verts
    return _to_stl_frame(m)


def blocky_stl(model: VoxelModel, path=None) -> tuple[trimesh.Trimesh, bytes]:
    m = blocky_mesh(model)
    assert m.is_watertight, "blocky mesh is not watertight"
    data = m.export(file_type="stl")
    if path:
        open(path, "wb").write(data)
    return m, data


def smooth_stl(model: VoxelModel, path=None, sigma: float = 0.8) -> tuple[trimesh.Trimesh, bytes, dict]:
    f = ndimage.gaussian_filter(np.pad(model.occ, 3).astype(np.float32), sigma)
    vmm = model.voxel_cm * 10
    V, F, _, _ = marching_cubes(f, level=0.5, spacing=(vmm, vmm, vmm))
    V = V - 3 * vmm + vmm / 2 + model.origin_cm * 10  # undo pad; sample points are voxel centers
    m = trimesh.Trimesh(V, F, process=True)
    if m.volume < 0:
        m.invert()
    _to_stl_frame(m)
    info = {"watertight": bool(m.is_watertight), "faces": int(len(m.faces)), "volume_cm3": float(abs(m.volume) / 1000)}
    data = m.export(file_type="stl")
    if path:
        open(path, "wb").write(data)
    return m, data, info
