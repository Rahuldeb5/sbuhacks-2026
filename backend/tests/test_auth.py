"""Sign-in, ownership and saved projects against the REAL Supabase project (opt-in: pytest -m supabase).
Creates two throwaway email/password users through the admin API (Google can't be scripted), signs them in for
genuine ES256 access tokens, and deletes the users, their rows and their Storage files at teardown."""
import os
import secrets
import shutil
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import config
import supa

DATA = Path(__file__).resolve().parents[2] / "data"
pytestmark = [pytest.mark.supabase,
              pytest.mark.skipif(not (config.supabase_configured() and os.environ.get("VITE_SUPABASE_PUBLISHABLE_KEY")),
                                 reason="Supabase not configured")]


def _admin(method, path, **kw):
    r = httpx.request(method, f"{config.SUPABASE_URL}/auth/v1{path}", headers=supa._headers(), timeout=20, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def _make_user(tag):
    email, pw = f"sns-test-{tag}-{secrets.token_hex(4)}@example.com", secrets.token_urlsafe(18)
    u = _admin("POST", "/admin/users", json={"email": email, "password": pw, "email_confirm": True,
                                             "user_metadata": {"full_name": f"Test {tag.upper()}"}})
    r = httpx.post(f"{config.SUPABASE_URL}/auth/v1/token", params={"grant_type": "password"}, timeout=20,
                   headers={"apikey": os.environ["VITE_SUPABASE_PUBLISHABLE_KEY"]}, json={"email": email, "password": pw})
    r.raise_for_status()
    return {"id": u["id"], "email": email, "h": {"Authorization": f"Bearer {r.json()['access_token']}"}}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import app as app_module
    app_module.SCANS_DIR = tmp_path_factory.mktemp("auth_scans")
    a, b = _make_user("a"), _make_user("b")
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("VLM_DISABLED", "1")
            with TestClient(app_module.app) as c:
                yield c, a, b, app_module
    finally:
        for u in (a, b):
            for p in supa.list_projects(u["id"]):
                supa.delete_dir(f"{u['id']}/{p['scan_id']}")
            _admin("DELETE", f"/admin/users/{u['id']}")       # cascades to the projects rows


def _scan(c, headers=None, name="duck"):
    with open(DATA / f"{name}.jpeg", "rb") as f:
        r = c.post("/api/scan", files={"image": (f"{name}.jpeg", f, "image/jpeg")}, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()


def test_me(env):
    c, a, _, _ = env
    assert c.get("/api/me").json() == {"user": None, "auth_configured": True}
    me = c.get("/api/me", headers=a["h"]).json()["user"]
    assert me["id"] == a["id"] and me["email"] == a["email"] and me["name"] == "Test A"
    assert c.get("/api/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/api/projects").status_code == 401


def test_owned_scan_is_private(env):
    c, a, b, _ = env
    s = _scan(c, a["h"])
    sid = s["scan_id"]
    assert s["owned"] and s["project"] is None
    body = {"scan_id": sid, "method": "revolve"}
    assert c.post("/api/shape", json=body, headers=a["h"]).status_code == 200
    for h in (b["h"], {}):                                   # another user, anonymous: same 404 as a missing scan
        assert c.post("/api/shape", json=body, headers=h).status_code == 404
        assert c.get(f"/api/scans/{sid}", headers=h).status_code == 404
        assert c.get(f"/api/scans/{sid}/description", headers=h).status_code == 404
        assert all(x["scan_id"] != sid for x in c.get("/api/scans", headers=h).json())
    # the file signature is the capability <img> and download links use; without it, 404
    assert c.get(s["urls"]["photo"]).status_code == 200
    assert c.get(s["urls"]["photo"].split("?")[0]).status_code == 404
    assert c.get(f"/api/export/{sid}", params={"style": "blocky"}).status_code == 404
    assert c.get(f"/api/export/{sid}", params={"style": "blocky", "s": s["file_sig"]}).status_code == 200


def test_save_claims_draft_and_lists_project(env):
    c, a, b, _ = env
    s = _scan(c)                                             # anonymous draft: open to anyone with the id
    sid = s["scan_id"]
    assert not s["owned"] and c.get(f"/api/scans/{sid}", headers=b["h"]).status_code == 200
    assert c.post("/api/projects", json={"scan_id": sid, "method": "mirror"}).status_code == 401
    t0 = time.perf_counter()
    r = c.post("/api/projects", json={"scan_id": sid, "method": "mirror", "plane_offset_cm": 5.0,
                                      "name": "Brass Duck"}, headers=a["h"])
    assert r.status_code == 200, r.text
    save_s = time.perf_counter() - t0
    p = r.json()
    assert p["name"] == "Brass Duck" and p["method"] == "mirror" and p["dims_cm"]["h"] > 0
    assert c.get(f"/api/scans/{sid}").status_code == 404      # claimed: no longer an open draft
    assert c.post("/api/projects", json={"scan_id": sid, "method": "mirror"}, headers=b["h"]).status_code == 404
    stored = set(supa.list_files(f"{a['id']}/{sid}"))
    assert {"photo.jpg", "depth_f.npy", "masks.npz", "meta.json", "thumbnail.png", "description.json"} <= stored
    lst = c.get("/api/projects", headers=a["h"]).json()
    mine = [x for x in lst if x["scan_id"] == sid]
    assert mine and mine[0]["thumbnail"].startswith(config.SUPABASE_URL)
    img = httpx.get(mine[0]["thumbnail"], timeout=20)
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    assert c.get("/api/projects", headers=b["h"]).json() == []
    print(f"\nsave (first upload) {save_s:.2f}s, stored {sorted(stored)}")


def test_reopen_from_storage_after_local_loss(env):
    """Saved on one server, opened on a fresh one: the folder comes back from Storage with the saved settings."""
    c, a, b, app_module = env
    sid = _scan(c, a["h"])["scan_id"]
    r = c.post("/api/projects", json={"scan_id": sid, "method": "inflate", "thickness": 1.6}, headers=a["h"])
    assert r.status_code == 200, r.text
    app_module.STATES.pop(sid, None)
    shutil.rmtree(app_module.SCANS_DIR / sid)
    assert c.get(f"/api/scans/{sid}", headers=b["h"]).status_code == 404     # B can't trigger a download
    assert not (app_module.SCANS_DIR / sid).exists()
    t0 = time.perf_counter()
    r = c.get(f"/api/scans/{sid}", headers=a["h"])
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["method"] == "inflate" and j["thickness"] == 1.6 and j["project"]["name"]
    assert c.get(j["urls"]["depth"]).status_code == 200
    print(f"\nreopen with download {time.perf_counter() - t0:.2f}s")


def test_rename_and_delete(env):
    c, a, b, app_module = env
    sid = _scan(c, a["h"])["scan_id"]
    assert c.post("/api/projects", json={"scan_id": sid, "method": "revolve"}, headers=a["h"]).status_code == 200
    assert c.patch(f"/api/projects/{sid}", json={"name": "Renamed"}, headers=b["h"]).status_code == 404
    assert c.patch(f"/api/projects/{sid}", json={"name": "Renamed"}, headers=a["h"]).json()["name"] == "Renamed"
    assert c.get(f"/api/scans/{sid}", headers=a["h"]).json()["project"]["name"] == "Renamed"
    assert c.delete(f"/api/projects/{sid}", headers=b["h"]).status_code == 404
    assert c.delete(f"/api/projects/{sid}", headers=a["h"]).json() == {"ok": True}
    assert supa.project_by_scan(sid) is None and supa.list_files(f"{a['id']}/{sid}") == []
    assert not (app_module.SCANS_DIR / sid).exists()
    assert c.get(f"/api/scans/{sid}", headers=a["h"]).status_code == 404
