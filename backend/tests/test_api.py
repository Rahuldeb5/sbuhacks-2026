"""Endpoint tests (real GPU model + rembg). Uses data/duck.jpeg and data/bottle.jpeg; scans land in a temp dir."""
import io
import json
import threading
import time
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

import app as app_module

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """No VLM here (deterministic, no LM Studio needed): descriptions take the fallback path. See test_vlm.py."""
    app_module.SCANS_DIR = tmp_path_factory.mktemp("scans")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLM_DISABLED", "1")
        with TestClient(app_module.app) as c:
            yield c


def scan(client, name, stream=False, **form):
    with open(DATA / f"{name}.jpeg", "rb") as f:
        return client.post("/api/scan" + ("?stream=1" if stream else ""),
                           files={"image": (f"{name}.jpeg", f, "image/jpeg")}, data=form)


@pytest.fixture(scope="module")
def duck(client):
    r = scan(client, "duck")
    assert r.status_code == 200, r.text
    return r.json()


def _settle(client, sid):
    """Wait until the post-scan pre-builds are done (both big buttons cached)."""
    for m in ("mirror", "revolve"):
        client.post("/api/shape", json={"scan_id": sid, "method": m})


def test_health(client):
    j = client.get("/api/health").json()
    assert j["ok"] and j["cuda"]


def test_scan_response(duck):
    for k in ("scan_id", "label", "recommended", "selection", "scores", "scale_source", "urls", "timings", "payload",
              "camera", "plane_offset_range"):
        assert k in duck
    assert duck["recommended"] == "mirror" and duck["method"] == "mirror"
    assert duck["selection"]["selection_trace"]["final"] == "mirror" and "Chosen method" in duck["selection"]["trace_text"]
    p = duck["payload"]
    n = len(p["observed"])
    assert n > 1000 and len(p["voxels"]) == 3 * n and len(p["colors"]) == 3 * n and "plane" in p
    lo, hi = duck["plane_offset_range"]
    assert lo <= 0 < hi


def test_files(client, duck):
    for k in ("photo", "depth", "mask", "thumbnail"):
        r = client.get(duck["urls"][k])
        assert r.status_code == 200 and len(r.content) > 1000


def test_big_buttons_instant_after_scan(client, duck):
    """Right after a scan, the other big button (Revolve) is a pre-built cache hit."""
    for m in ("mirror", "revolve"):
        t0 = time.perf_counter()
        r = client.post("/api/shape", json={"scan_id": duck["scan_id"], "method": m})
        assert r.status_code == 200 and time.perf_counter() - t0 < 1.0, m


def test_shape_under_1s(client, duck):
    """Uncached reshapes (slider values) at steady state."""
    _settle(client, duck["scan_id"])
    for method, t in (("inflate", 1.3), ("revolve", 0.8), ("inflate", 0.6)):
        t0 = time.perf_counter()
        r = client.post("/api/shape", json={"scan_id": duck["scan_id"], "method": method, "thickness": t})
        dt = time.perf_counter() - t0
        assert r.status_code == 200 and dt < 1.0, f"{method} t={t}: {dt:.2f}s"


def test_known_dim(client, duck):
    sid = duck["scan_id"]
    r = client.post("/api/shape", json={"scan_id": sid, "method": "inflate", "known_dim": "h=6"}).json()
    assert r["scale_source"] == "known_dim" and r["known_dim"] == {"h": 6.0}
    assert abs(r["payload"]["measurements_cm"]["h"] - 6.0) < 0.3
    r = client.post("/api/shape", json={"scan_id": sid, "method": "inflate", "known_dim": ""}).json()
    assert r["scale_source"] == "estimate" and r["payload"]["measurements_cm"]["h"] > 30


@pytest.mark.parametrize("style", ["blocky", "smooth"])
def test_export(client, duck, style):
    r = client.get(f"/api/export/{duck['scan_id']}", params={"style": style, "method": "inflate", "thickness": 1.0})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert trimesh.load(io.BytesIO(r.content), file_type="stl").is_watertight


def test_mirror_plane_offset(client, duck):
    """True scale (h=6): the slider moves the cached auto plane; deeper -> fatter; values outside the range clamp."""
    sid = duck["scan_id"]
    r = client.post("/api/shape", json={"scan_id": sid, "method": "mirror", "known_dim": "h=6"}).json()
    lo, hi = r["plane_offset_range"]
    assert -0.2 < lo <= 0 and 1.5 < hi < 4                    # duck: negatives collapse it; ~+1 cm looks right
    d = {}
    for off in (0.0, 0.7, 1.4):
        t0 = time.perf_counter()
        r = client.post("/api/shape", json={"scan_id": sid, "method": "mirror", "plane_offset_cm": off})
        assert r.status_code == 200 and time.perf_counter() - t0 < 1.0, off
        p = r.json()["payload"]
        assert abs(p["plane"]["offset_cm"] - off) < 1e-6 and p["measurements_cm"]["h"] < 8
        d[off] = p["measurements_cm"]["d"]
    assert d[1.4] > d[0.7] > d[0.0]
    r = client.post("/api/shape", json={"scan_id": sid, "method": "mirror", "plane_offset_cm": -3.0}).json()
    assert r["plane_offset_cm"] == lo and r["payload"]["measurements_cm"]["h"] > 5     # clamped, not collapsed
    client.post("/api/shape", json={"scan_id": sid, "method": "mirror", "known_dim": ""})


def test_reload_from_disk(client, duck):
    """A server restart forgets the in-memory state; the scan reloads from scans/<id>/ with its last shape."""
    sid = duck["scan_id"]
    client.post("/api/shape", json={"scan_id": sid, "method": "inflate", "thickness": 1.5})
    app_module.STATES.pop(sid)
    r = client.get(f"/api/scans/{sid}").json()
    assert r["method"] == "inflate" and r["thickness"] == 1.5 and r["recommended"] == "mirror"
    assert any(s["scan_id"] == sid for s in client.get("/api/scans").json())


def test_thumbnail_post(client, duck):
    png = client.get(duck["urls"]["photo"]).content
    r = client.post(f"/api/scans/{duck['scan_id']}/thumbnail", content=png, headers={"content-type": "image/jpeg"})
    assert r.status_code == 200
    assert client.post(f"/api/scans/{duck['scan_id']}/thumbnail", content=b"nope").status_code == 400


def test_stream_stages_and_cached_rescan(client):
    """Streamed stages in order; the bottle is revolve. Scanning the same photo again reuses depth + mask."""
    r = scan(client, "bottle", stream=True)
    ev = [json.loads(line) for line in r.text.strip().splitlines()]
    assert [e["stage"] for e in ev] == ["photo", "depth", "mask", "label", "done"]
    res = ev[-1]["result"]
    assert res["recommended"] == "revolve" and "axis" in res["payload"] and res["plane_offset_range"] is None
    t0 = time.perf_counter()
    again = scan(client, "bottle").json()
    assert again["timings"].get("perception_cached") and time.perf_counter() - t0 < 2.0
    assert again["scan_id"] != res["scan_id"] and again["recommended"] == "revolve"


def test_bad_requests(client, duck):
    assert client.post("/api/shape", json={"scan_id": duck["scan_id"], "method": "nope"}).status_code == 400
    assert client.post("/api/shape", json={"scan_id": "missing", "method": "mirror"}).status_code == 404
    assert client.post("/api/shape", json={"scan_id": "../../etc", "method": "mirror"}).status_code == 404
    assert client.get("/api/files/x/../../etc/passwd").status_code == 404
    assert client.get(f"/api/files/{duck['scan_id']}/meta.json").status_code == 404
    assert client.post("/api/shape", json={"scan_id": duck["scan_id"], "method": "inflate",
                                           "known_dim": "q=3"}).status_code == 400


def test_concurrent_identical_requests_build_once(client, duck, monkeypatch):
    """Two identical uncached requests at once: one builds, the other waits for it (in-flight dedupe)."""
    calls = []
    orig = app_module.shape_from_scan
    monkeypatch.setattr(app_module, "shape_from_scan", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    body = {"scan_id": duck["scan_id"], "method": "inflate", "thickness": 2.15}
    out = []
    ts = [threading.Thread(target=lambda: out.append(client.post("/api/shape", json=body).status_code)) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out == [200, 200] and len(calls) == 1


def test_description_fallback(client, duck):
    d = client.get(f"/api/scans/{duck['scan_id']}/description").json()
    assert d["status"] == "fallback"
    for k in ("name", "description", "shape_category", "method_explanation", "suggested_method"):
        assert k in d["result"]
    assert d["auto"]["recommended"] == "mirror" and "plane" not in d["auto"]
