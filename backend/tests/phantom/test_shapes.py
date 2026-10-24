"""Shape methods on phantoms with perfect depth."""
import pytest

from pipeline.shapes import METHODS

from .harness import evaluate, get_gt

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
