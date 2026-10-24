"""Voxel core on phantoms (no depth model, no shape method): Grid, fill_rays, ray_pixels, cleanup, edge-contact fix,
finalize (world frame), payload, blocky/smooth STL, exact reflection. The oracle fill uses the TRUE back surface, so
anything below ~0.9 IoU is the voxel core's fault."""
import math

import numpy as np
import pytest
from scipy import ndimage

from pipeline import mesh
from pipeline.intrinsics import build_K
from pipeline.shapes.common import STRUCT26, Grid, ShapeResult, fill_rays, ray_pixels
from pipeline.shapes.symmetry import reflect_occupancy, reflect_union, self_symmetry_iou
from pipeline.voxels import finalize, fix_edge_contacts

from .harness import evaluate, get_gt
from .metrics import iou, to_union
from .phantoms import _scene

# measured oracle-fill IoU after cleanup; misses are a <=1-voxel quantization shell (any-sample vs centre-inside)
ORACLE_IOU = {"duck": 0.913, "bottle": 0.942, "asym_duck": 0.898, "fish": 0.880, "car": 0.897, "teapot": 0.927}
# front-to-back fill fills along-ray gaps: asym_duck's handle ring hole, the space under the car between the wheels
ALONG_RAY_CONCAVE = {"asym_duck", "car"}


def test_K_from_hfov():
    K = build_K(1024, 768, 69.0)
    assert math.isclose(2 * math.degrees(math.atan(511.5 / K[0, 0])), 69.0, abs_tol=0.2)
    assert K[0, 0] == K[1, 1]


def test_render_depth_is_planar_z():
    import open3d as o3d
    r = get_gt("bottle").r
    v, u = np.argwhere(r.mask)[::500].T
    z = r.Z[v, u]
    P = np.stack([(u - r.K[0, 2]) / r.K[0, 0] * z, (v - r.K[1, 2]) / r.K[1, 1] * z, z], 1)
    d = _scene(r.parts).compute_distance(o3d.core.Tensor(r.to_world(P).astype(np.float32))).numpy()
    assert d.max() < 0.01


def test_gt_self_iou():
    gt = get_gt("duck")
    assert iou(gt.occ, gt.grid, gt.occ, gt.grid) == 1.0


@pytest.mark.parametrize("name", list(ORACLE_IOU))
def test_oracle_fill(name):
    gt = get_gt(name)
    res = gt.oracle_fill()
    o = evaluate(gt, res)
    assert o["iou"] > ORACLE_IOU[name] - 0.03
    assert o["blocky_watertight"] and o["smooth_watertight"]
    assert o["chamfer_mm"] < 2.0
    if name in ALONG_RAY_CONCAVE:
        return
    A, B, _ = to_union(res.occ, res.grid, gt.occ, gt.grid)
    assert not (A & ~ndimage.binary_dilation(B, STRUCT26)).any(), "fill put voxels > 1 voxel from the true surface"


def test_edge_contact_fix_makes_watertight():
    occ = np.zeros((4, 4, 4), bool)
    occ[1, 1, 1] = occ[2, 2, 1] = True        # touch along an edge only
    fixed, added = fix_edge_contacts(occ)
    assert added >= 1
    res = ShapeResult(fixed, fixed, np.zeros(fixed.shape + (3,), np.uint8), Grid(np.zeros(3), fixed.shape, 0.2),
                      np.zeros((0, 3)), "test", 1.0)
    assert mesh.blocky_mesh(finalize(res)).is_watertight


def test_edge_contact_raw_is_not_watertight():
    """Guards the guard: without the fix, the 2-voxel edge contact really is non-manifold."""
    from pipeline.voxels import VoxelModel
    occ = np.zeros((2, 2, 1), bool)
    occ[0, 0, 0] = occ[1, 1, 0] = True
    m = VoxelModel(occ, np.zeros((2, 2, 1, 3), np.uint8), occ, 0.2, np.zeros(3), "t", 1.0, {})
    assert not mesh.blocky_mesh(m).is_watertight


def test_ray_pixels_stride():
    gt = get_gt("duck")
    us, vs = ray_pixels(gt.mask, gt.Z, gt.r.K, 0.2)
    assert len(us) == gt.mask.sum()                         # phantom: footprint 0.08 cm < voxel/2 -> stride 1
    Znear = gt.Z / 10                                       # macro: like the real duck at true scale (~7 cm away)
    us, vs = ray_pixels(gt.mask, Znear, gt.r.K, 0.2)
    k = round(math.sqrt(gt.mask.sum() / len(us)))
    foot = float(np.median(Znear[gt.mask])) / gt.r.K[0, 0]
    assert k > 1 and foot * k <= 0.1 + 1e-9                 # every voxel column still gets a ray


def test_payload_format_and_world_frame():
    gt = get_gt("bottle")
    m = finalize(gt.oracle_fill())
    p = m.payload()
    n = len(p["observed"])
    assert len(p["voxels"]) == 3 * n and len(p["colors"]) == 3 * n and n < p["n_solid"]
    assert set(p["measurements_cm"]) == {"w", "h", "d"} and p["voxel_size_mm"] == 2.0
    assert p["dims"] == list(m.occ.shape) and p["method"] == "oracle" and p["scale_source"] == "estimate"
    assert all(0 <= c <= 255 for c in p["colors"][:3000]) and set(p["observed"]) <= {0, 1}
    ox, oy, oz = p["origin_mm"]                             # sits on y=0, centred in x/z
    assert oy == 0 and math.isclose(ox, -p["dims"][0] * 1.0) and math.isclose(oz, -p["dims"][2] * 1.0)
    assert p["measurements_cm"]["h"] == pytest.approx(24.0, abs=0.4)    # bottle is 24 cm tall
    # world y up: the bottle's wide body is at the bottom, the neck at the top
    width = m.occ.any(2).sum(0)                             # per-height x extent
    assert width[: len(width) // 4].mean() > 2 * width[-len(width) // 8:].mean()
    # world z toward the viewer: observed (front) voxels sit at larger z than inferred ones
    ijk = np.argwhere(m.occ)
    obs = m.observed[m.occ]
    assert ijk[obs, 2].mean() > ijk[~obs, 2].mean() + 5


def test_rotated_grid_matches_camera_grid():
    """finalize's R branch: a grid in the level world frame (diag(1,-1,-1), what the app always uses) must give the
    same model as the camera-frame grid + 180 deg flip."""
    gt = get_gt("duck")
    a = finalize(gt.oracle_fill())
    b = finalize(gt.oracle_fill(R=np.diag([1.0, -1, -1])))
    assert a.occ.shape == b.occ.shape
    assert abs(int(a.occ.sum()) - int(b.occ.sum())) / a.occ.sum() < 0.01


def test_stl_units_axes_and_volume():
    gt = get_gt("bottle")
    m = finalize(gt.oracle_fill())
    blk = mesh.blocky_mesh(m)
    sm, _, info = mesh.smooth_stl(m)
    ext = blk.bounds[1] - blk.bounds[0]                     # STL is mm, Z-up: (w, d, h)
    w, h, d = (m.measurements_cm[k] * 10 for k in "whd")
    assert np.allclose(ext, [w, d, h], atol=1e-6) and abs(blk.bounds[0][2]) < 1e-6     # stands on z=0
    v_vox = m.occ.sum() * (m.voxel_cm * 10) ** 3
    assert blk.volume == pytest.approx(v_vox, rel=1e-6)
    assert info["watertight"] and info["volume_cm3"] * 1000 == pytest.approx(v_vox, rel=0.05)


def test_reflection_on_oracle_duck():
    """symmetry.py without mirror: the duck phantom is symmetric about its true plane, so the oracle fill already is
    (up to quantization); reflect_union makes it exact; reflecting twice is the identity."""
    gt = get_gt("duck")
    res = gt.oracle_fill()
    mp = gt.oracle_midplane()
    n, d = mp.n, mp.d
    assert self_symmetry_iou(res.occ, res.grid, n, d) > 0.85
    twice = reflect_occupancy(reflect_occupancy(res.occ, res.grid, n, d), res.grid, n, d)
    assert (twice & res.occ).sum() / (twice | res.occ).sum() > 0.97
    u, col = reflect_union(res.occ, res.colors, res.grid, n, d)
    ref = res.occ | reflect_occupancy(res.occ, res.grid, n, d)
    assert (u & ref).sum() / (u | ref).sum() > 0.99 and (u >= res.occ).all()
    assert self_symmetry_iou(u, res.grid, n, d) > 0.97
    assert (col[u & ~res.occ].sum(-1) > 0).all()            # reflected voxels got their source color


def test_resymmetrize_then_edge_fix_is_watertight():
    """clean_shape's mirror branch (debug['symmetric_plane']): cleanup -> reflect_union -> edge fix. Bug: the union recreated edge-only contacts and the blocky STL wasn't watertight."""
    gt = get_gt("duck")
    res = gt.oracle_fill()
    mp = gt.oracle_midplane()
    res.occ, res.colors = reflect_union(res.occ, res.colors, res.grid, mp.n, mp.d)
    res.debug["symmetric_plane"] = (mp.n, mp.d)
    m = finalize(res)
    assert m.cleanup_info["self_symmetry_iou"] > 0.97
    assert mesh.blocky_mesh(m).is_watertight and mesh.smooth_stl(m)[2]["watertight"]
    assert "plane" in m.payload() and abs(np.linalg.norm(m.payload()["plane"]["normal"]) - 1) < 1e-4
