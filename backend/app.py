"""FastAPI backend.  uvicorn app:app --host 0.0.0.0 --port 8000   (from backend/)

POST /api/scan              multipart image [+ known_dim "h=6", scale auto|estimate|marker|known_dim, marker_cm]
                            ?stream=1 -> NDJSON stage events (photo, depth, mask, label, done | error)
POST /api/shape             {scan_id, method, thickness?, known_dim?, plane_offset_cm?} -> reruns ONLY the shape step
GET  /api/export/{id}       ?style=blocky|smooth&method=&thickness=&plane_offset_cm=  -> STL download
GET  /api/scans             your recent scans on this server;  GET /api/scans/{id} reload one (from Storage if saved)
POST /api/scans/{id}/thumbnail   PNG/JPEG body (canvas snapshot) replaces the server-rendered thumbnail
GET  /api/scans/{id}/description  async object description from the VLM (poll until status != pending)
GET  /api/files/{id}/{f}?s= photo.jpg / depth.png / mask.png / thumbnail.png  (s = the scan's file signature)
GET  /api/me                the signed-in user (or null)
POST /api/projects          save (sign-in required): scan folder -> Supabase Storage, row -> projects table
GET  /api/projects          your saved projects;  PATCH/DELETE /api/projects/{scan_id}  rename / delete
GET  /api/health

Sign-in: Supabase (Google). The browser sends `Authorization: Bearer <access token>`; supa.verify checks it locally.
A scan made while signed in, or saved, belongs to that user: anyone else gets 404. Anonymous drafts are open to
whoever has the id. <img> and download links can't send the header, so file/export URLs carry `s`, an HMAC of the
scan id that only an authorized response hands out.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import queue
import re
import secrets
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

import config
import supa
import vlm
from gpu_manager import GpuManager
from pipeline import depth, mesh, segment
from pipeline.intrinsics import Camera, load_image
from pipeline.perceive import CACHE_DIR, Perception, cache_key, load_cached, store
from pipeline.render import render as render_voxels
from pipeline.run import Scan, mirror_from_scan, mirror_offset_min, scan_from_perception, shape_from_scan, timed
from pipeline.scale import ScaleInfo, parse_known
from pipeline.shapes.auto_select import decide, trace_text
from pipeline.shapes.plane_fit import CATEGORY_PRIOR, CATEGORY_PRIOR_ENABLED
from pipeline.voxels import VoxelModel

log = logging.getLogger("app")
SCANS_DIR = Path(__file__).resolve().parent / "scans"
FILES = {"photo.jpg", "depth.png", "mask.png", "thumbnail.png"}
METHODS = ("mirror", "revolve", "inflate")
LABEL_FALLBACK = vlm.FALLBACK_NAME
THICKNESS_RANGE = (0.25, 2.5)
MODEL_CACHE_SIZE = 48
SID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
POOL = ThreadPoolExecutor(max_workers=6)
# stable across restarts when Supabase is configured (derived from the secret key), random otherwise
FILE_KEY = hashlib.sha256(b"scan-files|" + (config.SUPABASE_KEY.encode() or secrets.token_bytes(32))).digest()
# saved scans: files uploaded once (they never change) vs on every save
STORE_ONCE = ("photo.jpg", "depth.png", "mask.png", "depth_f.npy", "masks.npz")
STORE_EACH = ("meta.json", "description.json", "thumbnail.png")


@dataclass
class ScanState:
    """Everything the server keeps for one scan. `lock` guards scan.known/scale changes and meta writes."""
    sid: str
    scan: Scan
    meta: dict
    mirror_cache: dict = field(default_factory=dict)   # auto plane + known_dim factor (run.mirror_from_scan)
    description: dict = field(default_factory=lambda: {"status": "none"})
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def dir(self) -> Path:
        return SCANS_DIR / self.sid

    def save_meta(self):
        with self.lock:
            (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=float))


STATES: dict[str, ScanState] = {}
STATES_LOCK = threading.Lock()
MODEL_CACHE: "OrderedDict[tuple, VoxelModel]" = OrderedDict()
INFLIGHT: dict[tuple, threading.Event] = {}            # model key -> done event (dedupes identical concurrent builds)
CACHE_LOCK = threading.Lock()
GPU: GpuManager | None = None
FETCH_LOCK = threading.Lock()                           # one Storage download per scan at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    SCANS_DIR.mkdir(exist_ok=True)
    t = time.perf_counter()
    depth.get_model()
    depth.infer(np.zeros((256, 256, 3), np.uint8))    # the first CUDA call is ~1 s; pay it at startup, not on a scan
    segment.segment(np.zeros((64, 64, 3), np.uint8))  # same for onnxruntime's first rembg call
    global GPU
    GPU = GpuManager()
    if config.vlm_disabled():
        GPU.loop = asyncio.get_running_loop()           # depth still goes through the GPU lock; no LM Studio calls
    else:
        await GPU.startup()                            # coresident: (re)load the VLM with the small settings
    log.info("models ready in %.1fs (gpu: %s)", time.perf_counter() - t, GPU.status())
    yield
    await GPU.client.aclose()


app = FastAPI(title="Steel & Steam scanner", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://[^/]+:(5173|4173)$|https?://(localhost|127\.0\.0\.1)(:\d+)?$",
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ images, (de)serialization
def file_sig(sid: str) -> str:
    return hmac.new(FILE_KEY, sid.encode(), hashlib.sha256).hexdigest()[:32]


def urls(sid: str) -> dict:
    s = file_sig(sid)
    return {k: f"/api/files/{sid}/{f}?s={s}" for k, f in
            (("photo", "photo.jpg"), ("depth", "depth.png"), ("mask", "mask.png"), ("thumbnail", "thumbnail.png"))}


# ------------------------------------------------------------------ sign-in + ownership
def current_user(request: Request) -> dict | None:
    """The verified Supabase user, or None when the request carries no token. A bad token is 401, not anonymous."""
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        return None
    if not config.supabase_configured():
        raise HTTPException(503, "sign-in is not configured on this server")
    try:
        return supa.verify(h[7:].strip())
    except supa.AuthError:
        raise HTTPException(401, "invalid or expired sign-in")


def require_user(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "sign in required")
    return u


def fetch_saved(sid: str, user: dict | None):
    """A saved scan that isn't on this server's disk: download its folder from Storage (owner only)."""
    if not user or not config.supabase_configured() or (SCANS_DIR / sid / "meta.json").exists():
        return
    with FETCH_LOCK:
        if (SCANS_DIR / sid / "meta.json").exists():
            return
        proj = supa.project_by_scan(sid)
        if proj and proj["user_id"] == user["id"]:
            shutil.rmtree(SCANS_DIR / f"{sid}.part", ignore_errors=True)
            supa.download_dir(f"{user['id']}/{sid}", SCANS_DIR / sid)


def authorize(request: Request, sid: str, sig: str | None = None) -> ScanState:
    """The scan, if this caller may see it: its owner (Bearer token), anyone holding its file signature, or anyone
    at all for an ownerless draft. Everyone else gets the same 404 as for a missing scan (no id probing)."""
    if not SID_RE.match(sid or ""):
        raise HTTPException(404, "unknown scan")
    user = current_user(request)
    fetch_saved(sid, user)
    st = get_state(sid)
    owner = st.meta.get("owner")
    if owner and not (user and user["id"] == owner) and not (sig and hmac.compare_digest(sig, file_sig(sid))):
        raise HTTPException(404, "unknown scan")
    return st


def depth_png(Z: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(Z, 1), np.percentile(Z, 99)
    n = np.clip((Z - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_INFERNO)   # near = bright


def mask_png(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ov = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    ov[mask] = (0.45 * rgb[mask] + 0.55 * np.array([200, 152, 60])).astype(np.uint8)   # brass tint
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    ov[edge] = (255, 220, 140)
    return cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)


def scale_to_json(sc: ScaleInfo) -> dict:
    d = sc.to_json()
    if sc.table is not None:
        d["table_n"] = [float(x) for x in sc.table[0]]
        d["table_d"] = float(sc.table[1])
    return d


def scale_from_json(d: dict) -> ScaleInfo:
    info = {k: v for k, v in d.items() if k not in ("scale_source", "up_cam", "table_n", "table_d")}
    table = (np.array(d["table_n"]), d["table_d"]) if "table_n" in d else None
    return ScaleInfo(d.get("scale_source", "estimate"), np.array(d["up_cam"]), table, info)


def camera_json(cam: Camera) -> dict:
    return {"K": cam.K.tolist(), "width": cam.width, "height": cam.height, "hfov_deg": cam.hfov_deg,
            "source": cam.source, "f35": cam.f35}


def get_state(sid: str) -> ScanState:
    """In memory, else reloaded from scans/<sid>/. Unknown or malformed ids are 404 (no path tricks)."""
    if not SID_RE.match(sid or ""):
        raise HTTPException(404, "unknown scan")
    with STATES_LOCK:
        if sid in STATES:
            return STATES[sid]
    d = SCANS_DIR / sid
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        raise HTTPException(404, "unknown scan")
    rgb = cv2.cvtColor(cv2.imread(str(d / "photo.jpg")), cv2.COLOR_BGR2RGB)
    c = meta["camera"]
    cam = Camera(np.array(c["K"]), c["width"], c["height"], c["hfov_deg"], c["source"], c.get("f35"))
    m = np.load(d / "masks.npz")
    scan = Scan(rgb, cam, np.load(d / "depth_f.npy"), m["mask"], m["mask_full"], meta["auto"], meta["timings"],
                None, scale_from_json(meta["scale"]), meta.get("known"), meta.get("prior_scale", 1.0), meta.get("key", ""))
    st = ScanState(sid, scan, meta)
    p = d / "description.json"
    if p.exists():
        st.description = json.loads(p.read_text())
        if st.description.get("status") == "pending":   # the server stopped mid-request
            st.description = {"status": "fallback", "result": vlm.fallback("interrupted by a restart"), "meta": {}}
    with STATES_LOCK:
        return STATES.setdefault(sid, st)


# ------------------------------------------------------------------ models (cache + in-flight dedupe)
def plane_offset_range(st: ScanState) -> list[float]:
    """Mirror slider bounds in the model's cm: min keeps the plane behind ~95 % of the visible surface (lower values
    collapse the object), max = half the object size (beyond that the beak/edges smear)."""
    get_model(st, "mirror", 1.0)                       # the cached auto plane + known_dim factor must exist
    key = (st.scan.scale.source, json.dumps(st.scan.known), st.scan.prior_scale)
    c = st.mirror_cache.get(key, {})
    # the plane was estimated on the final (already rescaled) depth, so its object size is in model cm as is
    L = float(((c.get("estimator") or {}).get("object_size_cm")) or 10.0)
    return [round(mirror_offset_min(st.scan, st.mirror_cache), 2), round(0.5 * L, 2)]


def get_model(st: ScanState, method: str, t: float, tm: dict | None = None, plane_offset: float = 0.0) -> VoxelModel:
    t = 1.0 if method == "mirror" else float(np.clip(t, *THICKNESS_RANGE))   # mirror depth comes from its plane
    plane_offset = float(plane_offset) if method == "mirror" else 0.0
    with st.lock:
        scan = replace(st.scan)                        # consistent snapshot of known/scale for this build
    key = (st.sid, method, round(t, 3), round(plane_offset, 2), json.dumps(scan.known), scan.scale.source,
           scan.prior_scale)
    while True:
        with CACHE_LOCK:
            if key in MODEL_CACHE:
                MODEL_CACHE.move_to_end(key)
                return MODEL_CACHE[key]
            ev = INFLIGHT.get(key)
            if ev is None:
                ev = INFLIGHT[key] = threading.Event()
                break
        ev.wait(30)          # someone else (e.g. the post-scan pre-build) is building this exact model
    try:
        if method == "mirror" and abs(plane_offset) > 1e-9:
            get_model(st, "mirror", 1.0)               # the shared auto plane first (deduped)
        if method == "mirror":
            _, model = mirror_from_scan(scan, plane_offset, st.mirror_cache, tm)
        else:
            _, model = shape_from_scan(scan, method, t, tm)
        with CACHE_LOCK:
            MODEL_CACHE[key] = model
            while len(MODEL_CACHE) > MODEL_CACHE_SIZE:
                MODEL_CACHE.popitem(last=False)
        return model
    finally:
        with CACHE_LOCK:
            INFLIGHT.pop(key, None)
        ev.set()


def _prebuild(st: ScanState, method: str):
    try:
        get_model(st, method, 1.0)
    except Exception:  # noqa: BLE001
        log.exception("pre-build of %s failed", method)


def selection_json(auto: dict) -> dict:
    """What the UI needs from a decision: the method, whether the revolve call was borderline, and the trace."""
    return {"recommended": auto["recommended"], "geometric": auto.get("geometric", auto["recommended"]),
            "borderline": auto.get("borderline", False), "reason": auto["reason"],
            "selection_trace": auto.get("selection_trace"),
            "trace_text": trace_text(auto) if "selection_trace" in auto else ""}


def scan_response(st: ScanState, method: str, t: float, tm: dict, model: VoxelModel, plane_offset: float = 0.0) -> dict:
    scan, meta = st.scan, st.meta
    auto = {k: v for k, v in scan.auto.items() if k != "plane"}
    return {"scan_id": st.sid, "label": meta["label"], "label_source": meta.get("label_source"),
            "recommended": scan.auto["recommended"], "reason": scan.auto["reason"], "scores": auto["scores"],
            "selection": selection_json(scan.auto), "scale_source": scan.scale.source, "scale": scale_to_json(scan.scale),
            "known_dim": scan.known, "camera": meta["camera"], "urls": urls(st.sid), "timings": tm,
            "method": method, "thickness": t, "plane_offset_cm": plane_offset,
            "plane_offset_range": plane_offset_range(st) if method == "mirror" else None, "payload": model.payload(),
            "created": meta["created"], "file_sig": file_sig(st.sid), "owned": bool(meta.get("owner")),
            "project": {"name": meta["project_name"], "saved_at": meta.get("saved_at")} if meta.get("project_name") else None}


# ------------------------------------------------------------------ scan
def run_scan(data: bytes, known: dict | None, scale_mode: str, marker_cm: float, emit, owner: str | None = None) -> dict:
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = SCANS_DIR / sid
    d.mkdir(parents=True)
    (d / "input.jpg").write_bytes(data)
    tm: dict = {}
    t0 = time.perf_counter()
    p = load_cached(data)                              # same photo scanned before: depth + mask are free
    with timed(tm, "load_ms"):
        rgb, cam = (p.rgb, p.cam) if p else load_image(data)
        cv2.imwrite(str(d / "photo.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    emit({"stage": "photo", "url": urls(sid)["photo"], "ms": tm["load_ms"], "scan_id": sid})
    if p is None:
        def run(fn):
            t = time.perf_counter()
            r = fn(rgb)
            return r, round((time.perf_counter() - t) * 1000, 1)

        # depth holds the GPU lock (never during a VLM load/unload); rembg runs on the CPU side by side
        infer = (lambda x: GPU.run_depth_sync(depth.infer, x)) if GPU else depth.infer
        fd, fs = POOL.submit(run, infer), POOL.submit(run, segment.segment)
        Z_raw, tm["depth_ms"] = fd.result()
        cv2.imwrite(str(d / "depth.png"), depth_png(Z_raw))
        emit({"stage": "depth", "url": urls(sid)["depth"], "ms": tm["depth_ms"]})
        (mask, mask_full), tm["segment_ms"] = fs.result()
        cv2.imwrite(str(d / "mask.png"), mask_png(rgb, mask_full))
        emit({"stage": "mask", "url": urls(sid)["mask"], "ms": tm["segment_ms"]})
        p = Perception(rgb, cam, Z_raw, mask, mask_full, cache_key(data), False)
        store(p)
    else:
        tm["perception_cached"] = True
        cv2.imwrite(str(d / "depth.png"), depth_png(p.Z_raw))
        emit({"stage": "depth", "url": urls(sid)["depth"], "ms": 0.0, "cached": True})
        cv2.imwrite(str(d / "mask.png"), mask_png(rgb, p.mask_full))
        emit({"stage": "mask", "url": urls(sid)["mask"], "ms": 0.0, "cached": True})
    scan = scan_from_perception(p, scale_mode, marker_cm, known, tm)
    tm = scan.timings
    meta = {"scan_id": sid, "owner": owner, "key": p.key, "created_ts": time.time(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": LABEL_FALLBACK, "label_source": "pending", "camera": camera_json(scan.cam),
            "scale": scale_to_json(scan.scale), "known": scan.known, "auto": scan.auto, "timings": tm,
            "last": {"method": scan.auto["recommended"], "thickness": 1.0, "plane_offset_cm": 0.0}}
    st = ScanState(sid, scan, meta)
    with STATES_LOCK:
        STATES[sid] = st
    method = scan.auto["recommended"]
    model = get_model(st, method, 1.0, tm)
    for other in ("mirror", "revolve"):
        # pre-build the other big-button method AFTER the recommended one (before, it competes for the CPU), so
        # switching is a cache hit; the mirror plane comes from auto_select, so mirror is just the half-solid + reflect
        if other != method:
            POOL.submit(_prebuild, st, other)
    tm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    np.save(d / "depth_f.npy", scan.Z_f)
    np.savez_compressed(d / "masks.npz", mask=scan.mask, mask_full=scan.mask_full)
    meta["measurements_cm"] = model.measurements_cm
    img = render_voxels(model.payload(), "3/4", 320, "photo")
    cv2.imwrite(str(d / "thumbnail.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    meta["thumbnail_source"] = "server"
    st.save_meta()
    schedule_describe(st)
    emit({"stage": "label", "label": meta["label"], "source": meta["label_source"]})
    return scan_response(st, method, 1.0, tm, model)


def _persist_description(st: ScanState):
    try:
        (st.dir / "description.json").write_text(json.dumps(st.description, indent=1, default=str))
    except OSError:
        pass


def schedule_describe(st: ScanState):
    """Fire-and-forget VLM description on the server loop; /api/scan never waits for it. The UI polls
    /api/scans/{id}/description."""
    if GPU is None or GPU.loop is None or config.vlm_disabled() or not GPU.available:
        st.description = {"status": "fallback", "result": vlm.fallback("vlm disabled or unavailable"), "meta": {}}
        _persist_description(st)
        return
    st.description = {"status": "pending", "started": time.time()}
    photo = cv2.cvtColor(st.scan.rgb, cv2.COLOR_RGB2BGR)
    fut = asyncio.run_coroutine_threadsafe(
        vlm.describe(GPU, photo, cv2.imread(str(st.dir / "depth.png")), cv2.imread(str(st.dir / "mask.png")),
                     st.scan.auto), GPU.loop)
    fut.add_done_callback(lambda f: _store_description(st, f))


def _store_description(st: ScanState, fut):
    try:
        out = fut.result()
    except Exception as e:  # noqa: BLE001
        out = {"result": vlm.fallback(str(e)[:200]), "meta": {}}
    res = out["result"]
    st.description = {"status": "fallback" if res.get("fallback") else "ready", **out}
    _persist_description(st)
    if res.get("fallback"):
        return
    with st.lock:
        st.meta["label"], st.meta["label_source"] = res["name"], "vlm"
        # the category only scales the mirror plane's thickness prior (never a number from the VLM itself);
        # prior_scale is part of the model + mirror-plane cache keys, so the old mirror results simply stop matching
        scale = CATEGORY_PRIOR.get(res.get("shape_category"), 1.0) if CATEGORY_PRIOR_ENABLED else 1.0
        rewarm = scale != st.scan.prior_scale
        st.scan.prior_scale = st.meta["prior_scale"] = scale
        # the hint may flip a BORDERLINE revolve call, never a failed mirror sanity check; the UI decides whether to
        # apply it (only if the user hasn't picked a method), here it is recorded and its model pre-built
        hinted = decide(st.scan.auto["scores"], res.get("shape_category"))
        st.meta["selection_hint"] = {**hinted, "switch": hinted["recommended"] != st.scan.auto["recommended"]}
    st.save_meta()
    if rewarm:
        POOL.submit(_prebuild, st, "mirror")
    if st.meta["selection_hint"]["switch"]:
        POOL.submit(_prebuild, st, hinted["recommended"])


def hint_json(h: dict | None) -> dict | None:
    if not h:
        return None
    return {"recommended": h["recommended"], "switch": h["switch"], "reason": h["reason"],
            "hint": h["selection_trace"]["hint"], "hint_effect": h["selection_trace"]["hint_effect"],
            "selection_trace": h["selection_trace"]}


@app.post("/api/scan")
async def api_scan(request: Request, image: UploadFile = File(...), known_dim: str | None = Form(None),
                   scale: str = Form("auto"), marker_cm: float = Form(6.0), stream: int = Query(0)):
    user = current_user(request)
    owner = user["id"] if user else None
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if scale not in ("auto", "estimate", "marker", "known_dim"):
        raise HTTPException(400, "scale must be auto|estimate|marker|known_dim")
    try:
        known = parse_known(known_dim) if known_dim else None
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not stream:
        return JSONResponse(await asyncio.to_thread(run_scan, data, known, scale, marker_cm, lambda e: None, owner))
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            q.put({"stage": "done", "result": run_scan(data, known, scale, marker_cm, q.put, owner)})
        except Exception as e:  # noqa: BLE001
            log.exception("scan failed")
            q.put({"stage": "error", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            ev = q.get()
            yield json.dumps(ev, default=float) + "\n"
            if ev["stage"] in ("done", "error"):
                break

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ shape / export
class ShapeReq(BaseModel):
    scan_id: str
    method: str
    thickness: float = 1.0
    known_dim: str | None = None    # "h=6"; "" clears it; None leaves it as is
    plane_offset_cm: float = 0.0    # mirror only: symmetry plane shift (cm) relative to the auto estimate


def apply_known(st: ScanState, known_dim: str | None):
    """Set or clear the user's known dimension. It wins over the estimate; a marker keeps its gravity + table."""
    if known_dim is None:
        return
    with st.lock:
        scan, meta = st.scan, st.meta
        orig = scale_from_json(meta.get("scale_orig", meta["scale"]))
        meta.setdefault("scale_orig", meta["scale"])
        if known_dim == "":
            scan.known, scan.scale = None, orig
        else:
            try:
                scan.known = parse_known(known_dim)
            except ValueError as e:
                raise HTTPException(400, str(e))
            scan.scale = ScaleInfo("known_dim", orig.up_cam, orig.table, {"known_cm": scan.known, "base": orig.source})
        meta["known"] = scan.known
        meta["scale"] = scale_to_json(scan.scale)


@app.post("/api/shape")
def api_shape(req: ShapeReq, request: Request):
    if req.method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    st = authorize(request, req.scan_id)
    apply_known(st, req.known_dim)
    tm: dict = {}
    t0 = time.perf_counter()
    rng = plane_offset_range(st) if req.method == "mirror" else None
    off = float(np.clip(req.plane_offset_cm, *rng)) if rng else 0.0
    model = get_model(st, req.method, req.thickness, tm, off)
    with timed(tm, "payload_ms"):
        payload = model.payload()
    tm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t = 1.0 if req.method == "mirror" else float(np.clip(req.thickness, *THICKNESS_RANGE))
    with st.lock:
        st.meta["last"] = {"method": req.method, "thickness": t, "plane_offset_cm": off}
        st.meta["measurements_cm"] = model.measurements_cm
    st.save_meta()
    return {"scan_id": st.sid, "method": req.method, "thickness": t, "plane_offset_cm": off, "plane_offset_range": rng,
            "scale_source": st.scan.scale.source, "known_dim": st.scan.known, "timings": tm, "payload": payload}


@app.get("/api/export/{sid}")
def api_export(sid: str, request: Request, style: str = "blocky", method: str | None = None,
               thickness: float | None = None, plane_offset_cm: float | None = None, s: str | None = None):
    st = authorize(request, sid, s)
    last = st.meta["last"]
    method = method or last["method"]
    t = thickness if thickness is not None else last["thickness"]
    off = plane_offset_cm if plane_offset_cm is not None else last.get("plane_offset_cm", 0.0)
    if method not in METHODS or style not in ("blocky", "smooth"):
        raise HTTPException(400, "bad method/style")
    model = get_model(st, method, t, plane_offset=off)
    data = mesh.blocky_stl(model)[1] if style == "blocky" else mesh.smooth_stl(model)[1]
    name = "".join(ch if ch.isalnum() else "_" for ch in st.meta["label"]).strip("_").lower() or "scan"
    return Response(data, media_type="model/stl",
                    headers={"Content-Disposition": f'attachment; filename="{name}_{method}_{style}.stl"'})


# ------------------------------------------------------------------ scans, files, description
@app.get("/api/scans")
def api_scans(request: Request, limit: int = 50):
    """Scans on this server's disk that the caller may open: ownerless drafts + the caller's own."""
    user = current_user(request)
    out = []
    for d in sorted((p for p in SCANS_DIR.iterdir() if SID_RE.match(p.name)), reverse=True)[:max(1, min(limit, 200))]:
        try:
            m = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            continue                                   # still scanning, or a failed scan
        if m.get("owner") and not (user and user["id"] == m["owner"]):
            continue
        out.append({"scan_id": d.name, "label": m["label"], "created": m["created"], "method": m["last"]["method"],
                    "dims_cm": m.get("measurements_cm"), "scale_source": m["scale"].get("scale_source"),
                    "saved": bool(m.get("project_name")),
                    "thumbnail": f"{urls(d.name)['thumbnail']}&v={int((d / 'thumbnail.png').stat().st_mtime)}"})
    return out


@app.get("/api/scans/{sid}")
def api_scan_get(sid: str, request: Request):
    st = authorize(request, sid)
    last = st.meta["last"]
    tm: dict = {}
    model = get_model(st, last["method"], last["thickness"], tm, last.get("plane_offset_cm", 0.0))
    return scan_response(st, last["method"], last["thickness"], tm, model, last.get("plane_offset_cm", 0.0))


@app.post("/api/scans/{sid}/thumbnail")
async def api_thumb(sid: str, request: Request):
    st = await asyncio.to_thread(authorize, request, sid)
    if request.headers.get("content-type", "").startswith("multipart/"):
        data = await (await request.form())["file"].read()
    else:
        data = await request.body()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    cv2.imwrite(str(st.dir / "thumbnail.png"), img)
    with st.lock:
        st.meta["thumbnail_source"] = "canvas"
    st.save_meta()
    return {"ok": True}


@app.get("/api/files/{sid}/{name}")
def api_file(sid: str, name: str, s: str = ""):
    # signature only, no get_state(): stage images are fetched while the scan is still running (no meta.json yet)
    if name not in FILES or not SID_RE.match(sid) or not hmac.compare_digest(s, file_sig(sid)):
        raise HTTPException(404)
    p = SCANS_DIR / sid / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@app.get("/api/scans/{sid}/description")
def api_description(sid: str, request: Request):
    st = authorize(request, sid)
    d = {k: v for k, v in st.description.items() if k != "started"}
    return {"scan_id": sid, **d, "label": st.meta["label"],
            "auto": {k: v for k, v in st.scan.auto.items() if k != "plane"}, "prior_scale": st.scan.prior_scale,
            "selection": selection_json(st.scan.auto), "selection_hint": hint_json(st.meta.get("selection_hint"))}


@app.get("/api/health")
def health():
    import torch
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "scans_in_memory": len(STATES), "models_cached": len(MODEL_CACHE), "gpu": GPU.status() if GPU else None}


# ------------------------------------------------------------------ account + saved projects (Supabase)
@app.get("/api/me")
def api_me(request: Request):
    return {"user": current_user(request), "auth_configured": config.supabase_configured()}


class SaveReq(BaseModel):
    scan_id: str
    name: str | None = None
    method: str
    thickness: float = 1.0
    plane_offset_cm: float = 0.0


class RenameReq(BaseModel):
    name: str


def project_json(p: dict, thumb: str | None = None) -> dict:
    return {"scan_id": p["scan_id"], "name": p["name"], "method": p["method"], "thickness": p["thickness"],
            "plane_offset_cm": p["plane_offset_cm"], "known_dim": p.get("known_dim"), "scale_source": p["scale_source"],
            "dims_cm": p.get("dims_cm"), "created_at": p["created_at"], "updated_at": p["updated_at"], "thumbnail": thumb}


def own_project(request: Request, sid: str) -> tuple[dict, dict]:
    u = require_user(request)
    if not SID_RE.match(sid or ""):
        raise HTTPException(404)
    p = supa.project_by_scan(sid)
    if not p or p["user_id"] != u["id"]:
        raise HTTPException(404)
    return u, p


@app.post("/api/projects")
def api_save(req: SaveReq, request: Request):
    """Save (or re-save) a scan: the folder goes to Storage under <user>/<scan>/, the settings to `projects`.
    Saving claims an anonymous draft for the signed-in user."""
    u = require_user(request)
    if req.method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    st = authorize(request, req.scan_id)
    existing = supa.project_by_scan(st.sid)
    if existing and existing["user_id"] != u["id"]:
        raise HTTPException(404, "unknown scan")
    rng = plane_offset_range(st) if req.method == "mirror" else None
    off = float(np.clip(req.plane_offset_cm, *rng)) if rng else 0.0
    t = 1.0 if req.method == "mirror" else float(np.clip(req.thickness, *THICKNESS_RANGE))
    model = get_model(st, req.method, t, plane_offset=off)
    name = ((req.name or "").strip() or (existing or {}).get("name") or st.meta["label"])[:80]
    with st.lock:
        first = not st.meta.get("stored")
        st.meta.update(owner=u["id"], project_name=name, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), stored=True,
                       last={"method": req.method, "thickness": t, "plane_offset_cm": off},
                       measurements_cm=model.measurements_cm)
    st.save_meta()
    files = [st.dir / f for f in (STORE_ONCE if first else ()) + STORE_EACH if (st.dir / f).exists()]
    try:
        supa.upload_files(f"{u['id']}/{st.sid}", files)
        row = supa.save_project(u["id"], st.sid, name=name, method=req.method, thickness=t, plane_offset_cm=off,
                                known_dim=st.scan.known, scale_source=st.scan.scale.source,
                                dims_cm=model.measurements_cm, description=st.description.get("result"))
    except PermissionError:
        raise HTTPException(404, "unknown scan")
    except Exception as e:  # noqa: BLE001
        with st.lock:
            st.meta["stored"] = not first and st.meta.get("stored", False)
        log.exception("save failed")
        raise HTTPException(502, f"could not save to Supabase: {e}")
    return project_json(row, urls(st.sid)["thumbnail"])


@app.get("/api/projects")
def api_projects(request: Request):
    u = require_user(request)
    rows = supa.list_projects(u["id"])
    signed = supa.signed_urls([f"{u['id']}/{p['scan_id']}/thumbnail.png" for p in rows])
    return [project_json(p, signed.get(f"{u['id']}/{p['scan_id']}/thumbnail.png")) for p in rows]


@app.patch("/api/projects/{sid}")
def api_rename(sid: str, req: RenameReq, request: Request):
    own_project(request, sid)
    name = req.name.strip()[:80]
    if not name:
        raise HTTPException(400, "empty name")
    row = supa.rename_project(sid, name)
    with STATES_LOCK:
        st = STATES.get(sid)
    if st:
        with st.lock:
            st.meta["project_name"] = name
        st.save_meta()
    return project_json(row)


@app.delete("/api/projects/{sid}")
def api_delete(sid: str, request: Request):
    """Delete the saved project everywhere: Storage, the table, this server's copy and caches."""
    u, _ = own_project(request, sid)
    supa.delete_dir(f"{u['id']}/{sid}")
    supa.delete_project(sid)
    with STATES_LOCK:
        STATES.pop(sid, None)
    with CACHE_LOCK:
        for k in [k for k in MODEL_CACHE if k[0] == sid]:
            MODEL_CACHE.pop(k, None)
    shutil.rmtree(SCANS_DIR / sid, ignore_errors=True)
    return {"ok": True}
