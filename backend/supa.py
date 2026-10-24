"""Supabase: verify the browser's access token, the `projects` table (PostgREST) and the private `scans` bucket.

The browser signs in with Google through Supabase (supabase-js) and sends `Authorization: Bearer <access token>`.
Tokens are ES256 JWTs; they are verified HERE against the project's public JWKS (cached), so a request costs no
network round trip. The secret key is used only server-side, for the table and Storage, after that check.
Schema: supabase/schema.sql.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from joserfc import jwt
from joserfc.errors import JoseError
from joserfc.jwk import KeySet
from joserfc.jwt import JWTClaimsRegistry

import config

log = logging.getLogger("supa")
JWKS_TTL_S = 600
_jwks: tuple[float, KeySet] | None = None
_jwks_lock = threading.Lock()
_http = httpx.Client(timeout=30)


class AuthError(Exception):
    pass


def _headers(**extra) -> dict:
    return {"apikey": config.SUPABASE_KEY, "Authorization": f"Bearer {config.SUPABASE_KEY}", **extra}


# ------------------------------------------------------------------ tokens
def _key_set(force: bool = False) -> KeySet:
    global _jwks
    with _jwks_lock:
        if force or _jwks is None or time.time() - _jwks[0] > JWKS_TTL_S:
            r = _http.get(f"{config.SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=10)
            r.raise_for_status()
            _jwks = (time.time(), KeySet.import_key_set(r.json()))
        return _jwks[1]


def verify(token: str) -> dict:
    """Access token -> {id, email, name, avatar}. Raises AuthError on anything invalid or expired."""
    try:
        try:
            tok = jwt.decode(token, _key_set(), algorithms=["ES256"])
        except JoseError:
            tok = jwt.decode(token, _key_set(force=True), algorithms=["ES256"])   # signing key rotated
        JWTClaimsRegistry(iss={"essential": True, "value": f"{config.SUPABASE_URL}/auth/v1"},
                          aud={"essential": True, "value": "authenticated"},
                          exp={"essential": True}, sub={"essential": True}).validate(tok.claims)
    except (JoseError, ValueError) as e:
        raise AuthError(f"invalid token: {type(e).__name__}") from e
    c = tok.claims
    meta = c.get("user_metadata") or {}
    return {"id": c["sub"], "email": c.get("email") or meta.get("email"),
            "name": meta.get("full_name") or meta.get("name") or c.get("email"),
            "avatar": meta.get("avatar_url") or meta.get("picture")}


# ------------------------------------------------------------------ projects table
def _rest(method: str, params: dict | None = None, json=None, prefer: str | None = None) -> list[dict]:
    h = _headers(**({"Prefer": prefer} if prefer else {}))
    r = _http.request(method, f"{config.SUPABASE_URL}/rest/v1/projects", params=params, json=json, headers=h)
    if r.status_code >= 400:
        raise RuntimeError(f"projects {method} {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else []


def project_by_scan(scan_id: str) -> dict | None:
    rows = _rest("GET", {"scan_id": f"eq.{scan_id}", "select": "*"})
    return rows[0] if rows else None


def list_projects(user_id: str) -> list[dict]:
    return _rest("GET", {"user_id": f"eq.{user_id}", "select": "*", "order": "updated_at.desc"})


def save_project(user_id: str, scan_id: str, **fields) -> dict:
    """Insert or update the project for this scan. A scan saved by someone else is not ours to overwrite."""
    existing = project_by_scan(scan_id)
    if existing and existing["user_id"] != user_id:
        raise PermissionError(scan_id)
    if existing:
        return _rest("PATCH", {"scan_id": f"eq.{scan_id}"}, fields, "return=representation")[0]
    return _rest("POST", None, {"user_id": user_id, "scan_id": scan_id, **fields}, "return=representation")[0]


def rename_project(scan_id: str, name: str) -> dict:
    return _rest("PATCH", {"scan_id": f"eq.{scan_id}"}, {"name": name}, "return=representation")[0]


def delete_project(scan_id: str):
    _rest("DELETE", {"scan_id": f"eq.{scan_id}"})


# ------------------------------------------------------------------ storage: scans/<user_id>/<scan_id>/<file>
def _obj(path: str) -> str:
    return f"{config.SUPABASE_URL}/storage/v1/object/{config.SCANS_BUCKET}/{path}"


def upload_files(prefix: str, files: list[Path]):
    """Upload (overwrite) files under prefix/, a few at a time."""
    def put(f: Path):
        ct = {".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json"}.get(f.suffix, "application/octet-stream")
        r = _http.post(_obj(f"{prefix}/{f.name}"), content=f.read_bytes(),
                       headers=_headers(**{"Content-Type": ct, "x-upsert": "true"}), timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"upload {f.name} {r.status_code}: {r.text[:200]}")
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(put, files))


def list_files(prefix: str) -> list[str]:
    r = _http.post(f"{config.SUPABASE_URL}/storage/v1/object/list/{config.SCANS_BUCKET}",
                   json={"prefix": f"{prefix}/", "limit": 100}, headers=_headers())
    r.raise_for_status()
    return [o["name"] for o in r.json() if o.get("id")]          # folders have no id


def download_dir(prefix: str, dest: Path):
    """prefix/* -> dest/*. Written to a temp dir and renamed, so a half-downloaded scan never looks complete."""
    tmp = dest.with_name(dest.name + ".part")
    tmp.mkdir(parents=True, exist_ok=True)

    def get(name: str):
        r = _http.get(_obj(f"{prefix}/{name}"), headers=_headers(), timeout=120)
        r.raise_for_status()
        (tmp / name).write_bytes(r.content)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(get, list_files(prefix)))
    tmp.rename(dest)


def delete_dir(prefix: str):
    names = list_files(prefix)
    if names:
        r = _http.request("DELETE", f"{config.SUPABASE_URL}/storage/v1/object/{config.SCANS_BUCKET}",
                          json={"prefixes": [f"{prefix}/{n}" for n in names]}, headers=_headers())
        r.raise_for_status()


def signed_urls(paths: list[str], expires_s: int = 3600) -> dict[str, str]:
    """{path: absolute signed URL} for private objects (archive thumbnails)."""
    if not paths:
        return {}
    r = _http.post(f"{config.SUPABASE_URL}/storage/v1/object/sign/{config.SCANS_BUCKET}",
                   json={"expiresIn": expires_s, "paths": paths}, headers=_headers())
    r.raise_for_status()
    return {o["path"]: f"{config.SUPABASE_URL}/storage/v1{o['signedURL']}" for o in r.json() if o.get("signedURL")}
