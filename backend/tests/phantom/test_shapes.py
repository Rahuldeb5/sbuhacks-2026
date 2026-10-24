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


# revolve under camera tilt, perfect depth, camera-frame grid: prep results/revolve_tilt/results.json (reproduced
# axisup = no marker (up from the fitted axis); marker+clip = marker up vector + table clip.
REVOLVE_TILT = {(0, "axisup"): 0.957, (10, "axisup"): 0.933, (20, "axisup"): 0.889, (30, "axisup"): 0.842,
                (20, "v1"): 0.765, (20, "marker+clip"): 0.981, (30, "marker+clip"): 0.978}


@pytest.mark.parametrize("elev,mode", list(REVOLVE_TILT))
def test_revolve_tilt(elev, mode):
    from pipeline.run import clip_below_table
    from pipeline.scale import apply_marker
    from pipeline.voxels import cleanup
    from .metrics import iou
    gt = get_gt("bottle", float(elev), 0.0, table=True)
    r = gt.r
    ex = {"mask_full": gt.mask_full, "revolve_mode": "world" if mode == "marker+clip" else mode}
    mk = None
    if mode == "marker+clip":
        Zt = np.where(r.mask | np.isfinite(r.Z_table), np.where(r.mask, r.Z, r.Z_table), 0).astype(np.float32)
        mk = apply_marker(r.rgb, Zt, r.K, gt.mask_full, 6.0, "exif")    # pose (up, table) from the card corners
        up_true = r.R @ np.array([0, 1.0, 0])
        assert np.degrees(np.arccos(np.clip(mk[2].up_cam @ up_true, -1, 1))) < 0.25
        ex["up_cam"] = mk[2].up_cam
    res = METHODS["revolve"](gt.Z, gt.mask, r.K, r.rgb, ex)
    if mk:
        clip_below_table(res, *mk[2].table)
    occ, _, _ = cleanup(res.occ, res.colors)
    assert abs(iou(occ, res.grid, gt.occ, gt.grid) - REVOLVE_TILT[(elev, mode)]) < 0.03


def test_revolve_watertight_in_axisup_frame():
    """The app always passes a world frame; axisup then swaps in its own up -> the bottle comes out standing."""
    from pipeline.run import run_shape
    from pipeline.scale import LEVEL_UP, world_from_cam
    gt = get_gt("bottle", 20.0, 0.0, table=True)
    res, model = run_shape(gt.Z, gt.mask, gt.mask_full, gt.r.K, gt.r.rgb, "revolve", frame_R=world_from_cam(LEVEL_UP))
    assert res.debug["revolve_mode"] == "axisup" and mesh.blocky_mesh(model).is_watertight
    w, h, d = (model.measurements_cm[k] for k in "whd")
    assert abs(h - 24.0) < 2.5 and abs(w - 6.8) < 0.8 and abs(d - 6.8) < 0.8   # standing, not leaning 20 deg


def test_mirror_oracle_plane_duck():
    """prep test_phantom: with the TRUE symmetry plane, mirror v2 reconstructs the duck (> 0.85)."""
    gt = get_gt("duck")
    res = METHODS["mirror"](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {"mask_full": gt.mask_full, "plane": gt.oracle_midplane()})
    assert evaluate(gt, res)["iou"] > 0.85


@pytest.mark.parametrize("method", ["mirror", "revolve", "inflate"])
def test_all_methods_watertight_duck(method):
    gt = get_gt("duck")
    o = evaluate(gt, METHODS[method](gt.Z, gt.mask, gt.r.K, gt.r.rgb, {"mask_full": gt.mask_full}))
    assert o["blocky_watertight"] and o["smooth_watertight"]
