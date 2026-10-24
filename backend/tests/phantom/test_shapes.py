"""Shape methods on phantoms with perfect depth."""
import numpy as np
import pytest

from pipeline import mesh
from pipeline.shapes import METHODS
from pipeline.voxels import clean_shape, finalize

from .harness import evaluate, get_gt
from .metrics import iou_centered

# (phantom, method, use oracle plane) -> experiment A IoU
EXP_A = {
    ("duck", "inflate", False): 0.726,
    ("duck", "inflate", True): 0.877,
    ("bottle", "inflate", False): 0.644,
    ("asym_duck", "inflate", False): 0.714,
    ("asym_duck", "inflate", True): 0.864,
}


@pytest.mark.parametrize("name,method,oracle", list(EXP_A))
def test_experiment_A(name, method, oracle):
    gt = get_gt(name)
    extra = {"midplane": gt.oracle_midplane()} if oracle else {}
    res = METHODS[method](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {"mask_full": gt.mask_full, **extra})
    o = evaluate(gt, res)
    assert abs(o["iou"] - EXP_A[(name, method, oracle)]) < 0.03
    assert o["blocky_watertight"] and o["smooth_watertight"]


def test_inflate_front_never_moves():
    """Front-fixed thickness: t only moves the back, so the visible surface is identical."""
    gt = get_gt("duck")
    p = {"mask_full": gt.mask_full}
    a = METHODS["inflate"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {**p, "thickness": 0.5})
    b = METHODS["inflate"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {**p, "thickness": 2.0})
    assert (a.front_points == b.front_points).all()
    assert b.occ.sum() > 1.5 * a.occ.sum()


EVAL_MIRROR = {
    ("duck", "perfect"): 0.927, ("duck", "flat0.5"): 0.859, ("duck", "flat0.75"): 0.903,
    ("fish", "perfect"): 0.576, ("fish", "flat0.5"): 0.573, ("fish", "flat0.75"): 0.571,
    ("car", "perfect"): 0.573, ("car", "flat0.5"): 0.575, ("car", "flat0.75"): 0.582,
    ("teapot", "perfect"): 0.680, ("teapot", "flat0.5"): 0.656, ("teapot", "flat0.75"): 0.676,
    ("asym_duck", "perfect"): 0.880, ("asym_duck", "flat0.5"): 0.844, ("asym_duck", "flat0.75"): 0.872,
}


def flattened(gt, f):
    """Relief scaled by f about the median depth (prep exp_fix_shapes conditions)."""
    r = gt.r
    pivot = float(np.median(r.Z[r.mask]))
    return np.where(r.mask, pivot + f * (r.Z - pivot), 0).astype(np.float32)


@pytest.mark.parametrize("name,cond", list(EVAL_MIRROR))
def test_mirror_v2_eval(name, cond):
    gt = get_gt(name)
    Z = gt.Z if cond == "perfect" else flattened(gt, float(cond[4:]))
    res = METHODS["mirror"](Z, gt.mask, gt.r.K, gt.r.rgb, {"mask_full": gt.mask_full})
    occ, _, info = clean_shape(res)
    assert abs(iou_centered(occ, res.grid, gt.occ, gt.grid) - EVAL_MIRROR[(name, cond)]) < 0.03
    assert info["self_symmetry_iou"] > 0.97


@pytest.mark.parametrize("offset", [-1.0, 0.0, 1.5])
def test_mirror_v2_symmetric_and_watertight(offset):
    gt = get_gt("duck")
    res = METHODS["mirror"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {"mask_full": gt.mask_full, "plane_offset_cm": offset})
    model = finalize(res)
    assert model.cleanup_info["self_symmetry_iou"] > 0.97
    assert mesh.blocky_mesh(model).is_watertight and mesh.smooth_stl(model)[2]["watertight"]
    assert model.payload()["plane"]["offset_cm"] == offset


def test_mirror_v2_front_never_moves():
    """A deeper plane keeps every ray that already reached it; the visible front points are the same set."""
    gt = get_gt("duck")
    p = {"mask_full": gt.mask_full}
    a = METHODS["mirror"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {**p, "plane_offset_cm": 0.0})
    b = METHODS["mirror"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {**p, "plane_offset_cm": 2.0})
    fa = {tuple(np.round(q, 4)) for q in a.front_points}
    fb = {tuple(np.round(q, 4)) for q in b.front_points}
    assert fa <= fb
    assert b.occ.sum() > a.occ.sum()
