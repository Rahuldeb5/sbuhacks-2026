"""VLM descriptions: strict-JSON validation (pure) + the live path through the app (needs LM Studio with the
model; skipped otherwise). The VLM only names/describes and picks a category; it never produces geometry numbers."""
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import vlm

DATA = Path(__file__).resolve().parents[2] / "data"
GOOD = {"name": "Rubber Duck Toy", "description": "A yellow toy duck.", "shape_category": "bilateral",
        "method_explanation": "Straightness 0.13 is below 0.616, so not revolve.", "suggested_method": "mirror"}


def test_validate_accepts_fenced_json_with_chatter():
    out = vlm.validate("Sure!\n```json\n" + json.dumps({**GOOD, "shape_category": "Bilateral"}) + "\n```\nDone.")
    assert out["shape_category"] == "bilateral" and out["name"] == "Rubber Duck Toy"


@pytest.mark.parametrize("bad,why", [
    ({k: v for k, v in GOOD.items() if k != "suggested_method"}, "missing"),
    ({**GOOD, "name": "Duck"}, "2-4 words"),
    ({**GOOD, "name": "A Very Long Rubber Duck Toy"}, "2-4 words"),
    ({**GOOD, "shape_category": "animal"}, "shape_category"),
    ({**GOOD, "suggested_method": "extrude"}, "suggested_method"),
    ({**GOOD, "description": "  "}, "non-empty"),
])
def test_validate_rejects(bad, why):
    with pytest.raises(ValueError, match=why):
        vlm.validate(json.dumps(bad))
    with pytest.raises(ValueError, match="no JSON"):
        vlm.validate("I think it is a duck.")


def test_fallback_shape():
    f = vlm.fallback("timeout")
    assert f["name"] == vlm.FALLBACK_NAME and f["fallback"] == "timeout"
    assert set(GOOD) <= set(f)


def _lmstudio_up() -> bool:
    from gpu_manager import LMSTUDIO, VLM_MODEL
    try:
        r = httpx.get(f"{LMSTUDIO}/api/v1/models", timeout=3)
        return any(m.get("key") == VLM_MODEL for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001
        return False


def live(fn):
    """Opt-in (pytest -m live): slow, and depends on LM Studio. Skipped if the model isn't reachable."""
    return pytest.mark.live(pytest.mark.skipif(not _lmstudio_up(), reason="LM Studio / VLM model not reachable")(fn))


@pytest.fixture(scope="module")
def vclient(tmp_path_factory):
    import app as app_module
    app_module.SCANS_DIR = tmp_path_factory.mktemp("vlm_scans")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("VLM_DISABLED", raising=False)
        with TestClient(app_module.app) as c:
            yield c


def _scan_and_describe(c, name):
    with open(DATA / f"{name}.jpeg", "rb") as f:
        t0 = time.perf_counter()
        r = c.post("/api/scan", files={"image": (f"{name}.jpeg", f, "image/jpeg")})
    scan_s = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    sid = r.json()["scan_id"]
    d = {"status": "pending"}
    while d["status"] == "pending" and time.perf_counter() - t0 < 30:
        time.sleep(0.25)
        d = c.get(f"/api/scans/{sid}/description").json()
    return r.json(), d, scan_s, time.perf_counter() - t0


@live
@pytest.mark.parametrize("name,expected", [("duck", "mirror"), ("bottle", "revolve")])
def test_live_description(vclient, name, expected):
    assert vclient.get("/api/health").json()["gpu"]["vlm_available"]
    scan, d, scan_s, total_s = _scan_and_describe(vclient, name)
    assert scan["label_source"] == "pending" and scan["recommended"] == expected
    assert scan_s < 3.0, "the scan must not wait for the VLM"
    assert d["status"] == "ready", d                          # not the fallback
    r = d["result"]
    assert 2 <= len(r["name"].split()) <= 4 and r["shape_category"] in vlm.CATEGORIES
    assert r["method_explanation"] and r["suggested_method"] in vlm.METHODS
    assert d["label"] == r["name"] and total_s < 25
    assert d["prior_scale"] in (0.5, 1.0, 1.27)               # only the category table, never a VLM number
    listed = {s["scan_id"]: s for s in vclient.get("/api/scans").json()}
    assert listed[scan["scan_id"]]["label"] == r["name"]      # the archive shows the VLM name
    print(f"\n{name}: '{r['name']}' ({r['shape_category']}, suggests {r['suggested_method']}) "
          f"scan {scan_s:.2f}s, notes at {total_s:.2f}s; hint {d['selection_hint'] and d['selection_hint']['hint_effect']}")


@live
def test_live_reshape_stays_fast_while_describing(vclient):
    """The describe job overlaps the first clicks after a scan; a pre-built big button must still be instant."""
    with open(DATA / "duck.jpeg", "rb") as f:
        sid = vclient.post("/api/scan", files={"image": ("duck.jpeg", f, "image/jpeg")}).json()["scan_id"]
    t0 = time.perf_counter()
    r = vclient.post("/api/shape", json={"scan_id": sid, "method": "revolve"})
    assert r.status_code == 200 and time.perf_counter() - t0 < 1.5
