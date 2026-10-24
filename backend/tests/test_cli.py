"""End to end on the real photos through the CLI"""
import json
import time

import pytest

from pipeline import cli
from pipeline.run import mirror_from_scan, mirror_offset_min, prepare, shape_from_scan

DATA = cli.RUNS.parent.parent / "data"
PREP_ESTIMATE = {"duck": (60.8, 66.1, 30.4), "bottle": (31.8, 83.2, 21.0)}


@pytest.mark.parametrize("name", list(PREP_ESTIMATE))
def test_cli_inflate_real(name, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    out = cli.main(["--image", str(DATA / f"{name}.jpeg"), "--method", "inflate", "--out", name])
    for f in ("payload.json", "blocky.stl", "smooth.stl", "debug.png", "views.png", "meta.json"):
        assert (out / f).stat().st_size > 0, f
    meta = json.loads((out / "meta.json").read_text())
    assert meta["blocky"]["watertight"] and meta["smooth"]["watertight"]
    p = json.loads((out / "payload.json").read_text())
    assert {"voxel_size_mm", "origin_mm", "dims", "voxels", "colors", "observed", "measurements_cm", "method",
            "thickness", "scale_source", "n_solid"} <= set(p)
    # soft cap: auto_voxel sizes from the MEDIAN depth; DA's deeper rim makes the real duck 131
    assert max(p["dims"]) <= 128 * 1.1
    m = p["measurements_cm"]
    for got, want in zip((m["w"], m["h"], m["d"]), PREP_ESTIMATE[name]):
        assert abs(got - want) <= max(1.0, 0.03 * want)


def test_reshape_under_1s():
    """A thickness change re-runs shape + cleanup only, < 1 s warm."""
    scan = prepare(DATA / "duck.jpeg", "estimate")
    shape_from_scan(scan, "inflate", 1.0)
    t = time.perf_counter()
    shape_from_scan(scan, "inflate", 1.5)
    assert time.perf_counter() - t < 1.0


def test_mirror_offset_reshape_capped():
    """The plane slider at estimate scale: the offset-aware voxel cap keeps a +11 cm plane (~+1 cm true) near 1 s
    (2.6 s and 145 voxels wide before the fix)."""
    scan, cache = prepare(DATA / "duck.jpeg", "estimate"), {}
    mirror_from_scan(scan, 11.0, cache)
    t = time.perf_counter()
    _, model = mirror_from_scan(scan, 11.0, cache)
    assert time.perf_counter() - t < 1.3 and max(model.occ.shape) <= 128


@pytest.mark.parametrize("name,dh", [("duck", 0.48), ("bottle", 0.29)])
def test_cli_mirror_real(name, dh, tmp_path, monkeypatch):
    """Real photos with mirror v2 at the auto plane: exactly symmetric, watertight, and as thin as prep found it
    (duck D/H 0.48: prep "D 3.0 at h=6", the known too-shallow default plane; bottle 0.29, prep 8.3/28)."""
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    out = cli.main(["--image", str(DATA / f"{name}.jpeg"), "--method", "mirror", "--out", name])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["self_symmetry_iou"] > 0.97 and meta["blocky"]["watertight"] and meta["smooth"]["watertight"]
    assert meta["plane"] is not None
    m = meta["measurements_cm"]
    assert abs(m["d"] / m["h"] - dh) < 0.05


@pytest.mark.parametrize("name,expected", [("duck", "mirror"), ("bottle", "revolve")])
def test_auto_real(name, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    out = cli.main(["--image", str(DATA / f"{name}.jpeg"), "--out", name])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["method"] == expected and meta["auto"]["selection_trace"]["final"] == expected
    assert meta["blocky"]["watertight"] and meta["smooth"]["watertight"]


PREP_KNOWN = {("duck", "mirror", "h=6"): (5.4, 5.8, 3.0), ("duck", "revolve", "h=6"): (5.8, 6.0, 5.8),
              ("duck", "inflate", "h=6"): (5.2, 5.8, 2.6), ("bottle", "mirror", "h=28"): (10.3, 28.0, 8.3),
              ("bottle", "revolve", "h=28"): (9.6, 28.0, 9.4), ("bottle", "inflate", "h=28"): (10.0, 28.7, 9.6)}


@pytest.mark.parametrize("name,method,known", list(PREP_KNOWN))
def test_known_dim_real(name, method, known):
    k, v = known.split("=")
    scan = prepare(DATA / f"{name}.jpeg", known={k: float(v)})
    _, model = mirror_from_scan(scan, 0.0, {}) if method == "mirror" else shape_from_scan(scan, method)
    m = model.measurements_cm
    assert model.scale_source == "known_dim" and model.cleanup_info["known_dim"]["dim"] == k
    for got, want in zip((m["w"], m["h"], m["d"]), PREP_KNOWN[(name, method, known)]):
        assert abs(got - want) <= 0.45, (m, PREP_KNOWN[(name, method, known)])


def test_mirror_offset_real_duck():
    """Prep's offset sweep at h=6 (fix_shapes/real/notes.md): +1 cm is the best-looking duck, D 3.0 -> 5.0; the plane
    slider minimum sits at ~0 on this photo (negative offsets collapse it: -0.5 -> 2.6 cm tall)."""
    scan, cache = prepare(DATA / "duck.jpeg", known={"h": 6.0}), {}
    _, m0 = mirror_from_scan(scan, 0.0, cache)
    _, m1 = mirror_from_scan(scan, 1.0, cache)
    assert abs(m0.measurements_cm["d"] - 3.0) <= 0.3 and abs(m1.measurements_cm["d"] - 5.0) <= 0.3
    assert m1.cleanup_info["known_dim"]["factor"] == m0.cleanup_info["known_dim"]["factor"]   # factor held at offset 0
    assert -0.2 < mirror_offset_min(scan, cache) <= 0.0
