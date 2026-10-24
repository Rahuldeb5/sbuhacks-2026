"""auto-select v2 decision rules + the real photos' traces."""
from pathlib import Path

import pytest

from pipeline.shapes import auto_select as AS

DATA = Path(__file__).resolve().parents[2] / "data"


def scores(st=0.1, sym=0.5, vol=1.0, reach=0.95, occ=1e-4, sil=0.0, tilt=5.0):
    return {"revolve": {"straightness": st, "symmetry": sym},
            "mirror": {"volume_ratio": vol, "ray_reach": reach, "e_occ": occ, "e_sil": sil, "tilt_max_deg": tilt,
                       "tilt_limit_deg": 25.0}}


def test_order_revolve_mirror_inflate():
    assert AS.decide(scores(st=0.99, sym=0.99))["recommended"] == "revolve"
    assert AS.decide(scores())["recommended"] == "mirror"
    assert AS.decide(scores(vol=0.1))["recommended"] == "inflate"          # collapse
    assert AS.decide(scores(reach=0.2))["recommended"] == "inflate"        # rays never reach the plane
    assert AS.decide(scores(occ=0.01))["recommended"] == "inflate"         # heavy occlusion violation
    assert AS.decide(scores(sil=0.01))["recommended"] == "inflate"         # reflection leaves the silhouette
    assert AS.decide(scores(tilt=25.0))["recommended"] == "inflate"        # plane pinned at the tilt limit


def test_trace_lists_every_check():
    d = AS.decide(scores(), "bilateral")
    tr = d["selection_trace"]
    assert {c["check"] for c in tr["checks"]} == {"revolve.straightness", "revolve.symmetry", "mirror.volume_ratio",
                                                  "mirror.ray_reach", "mirror.e_occ", "mirror.e_sil", "mirror.tilt_deg"}
    assert all({"value", "threshold", "pass", "op"} <= set(c) for c in tr["checks"])
    assert tr["hint"] == "bilateral" and tr["final"] == "mirror"
    assert "Chosen method: mirror" in AS.trace_text(d)


def test_hint_flips_only_borderline():
    thr_st, thr_sym = AS.REVOLVE_STRAIGHT_MIN, AS.REVOLVE_SYMMETRY_MIN
    border = scores(st=thr_st + 0.02, sym=thr_sym + 0.02)                    # revolve, but borderline
    assert AS.decide(border)["recommended"] == "revolve" and AS.decide(border)["borderline"]
    assert AS.decide(border, "bilateral")["recommended"] == "mirror"
    clear = scores(st=0.99, sym=0.99)                                        # strong revolve: hint ignored
    assert AS.decide(clear, "bilateral")["recommended"] == "revolve"
    near_miss = scores(st=thr_st - 0.02, sym=thr_sym - 0.02)                 # mirror, but revolve nearly passed
    assert AS.decide(near_miss)["recommended"] == "mirror"
    assert AS.decide(near_miss, "round")["recommended"] == "revolve"
    assert AS.decide(scores(), "round")["recommended"] == "mirror"           # far from revolve: ignored


def test_hint_never_overrides_failed_sanity_check():
    border_fail = scores(st=AS.REVOLVE_STRAIGHT_MIN + 0.02, sym=AS.REVOLVE_SYMMETRY_MIN + 0.02, vol=0.05)
    d = AS.decide(border_fail, "bilateral")
    assert d["recommended"] == "revolve" and "sanity" in d["selection_trace"]["hint_effect"]
    assert AS.decide(scores(vol=0.05), "bilateral")["recommended"] == "inflate"


@pytest.mark.parametrize("photo,expected,trace", [
    ("duck", "mirror", {"revolve.straightness": 0.1347, "revolve.symmetry": 0.7324, "mirror.volume_ratio": 0.8837}),
    ("bottle", "revolve", {})])
def test_real_photos(photo, expected, trace):
    from pipeline.run import prepare
    scan = prepare(DATA / f"{photo}.jpeg")
    assert scan.auto["recommended"] == expected, scan.auto["reason"]
    assert scan.auto["selection_trace"]["final"] == expected
    got = {c["check"]: c["value"] for c in scan.auto["selection_trace"]["checks"]}
    for k, v in trace.items():
        assert abs(got[k] - v) < 0.01, (k, got[k])
