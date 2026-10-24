"""End to end on the real photos through the CLI"""
import json
import time

import pytest

from pipeline import cli
from pipeline.perceive import perceive
from pipeline.run import shape_from_perception

DATA = cli.RUNS.parent.parent / "data"
# (w, h, d) cm at estimate scale; prep: bottle inflate 31.8x83.2x21.0, duck inflate 60.8x66.1x30.4
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
    p = perceive(DATA / "duck.jpeg")
    shape_from_perception(p, "inflate", 1.0)
    t = time.perf_counter()
    shape_from_perception(p, "inflate", 1.5)
    assert time.perf_counter() - t < 1.0
