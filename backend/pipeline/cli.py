"""python -m pipeline.cli --image X.jpg [--method auto|inflate] [--thickness 1.0]
Writes runs/<id>/: payload.json (PLAYBOOK §2.13 model payload), blocky.stl, smooth.stl, debug.png, views.png, meta.json."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

from . import mesh
from .debug import save_debug
from .perceive import CACHE_DIR, perceive
from .render import render_grid
from .run import shape_from_perception, timed
from .shapes import METHODS

RUNS = Path(__file__).resolve().parent.parent / "runs"


def pick_method(requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "requested"
    return "inflate", "auto_select not built yet: fallback shape"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="auto", choices=["auto", *METHODS])
    ap.add_argument("--thickness", type=float, default=1.0)
    ap.add_argument("--voxel-mm", type=float, default=2.0, help="minimum voxel; the 128 cap can make it coarser")
    ap.add_argument("--out", default=None, help="run id (default: <time>-<image stem>-<method>)")
    ap.add_argument("--no-cache", action="store_true", help="recompute depth + mask")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    p = perceive(a.image, None if a.no_cache else CACHE_DIR)
    tm = {"perceive_ms": round((time.perf_counter() - t0) * 1000, 1), **p.timings}
    method, why = pick_method(a.method)
    res, model = shape_from_perception(p, method, a.thickness, tm, a.voxel_mm / 10)

    out = RUNS / (a.out or f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(a.image).stem}-{method}")
    out.mkdir(parents=True, exist_ok=True)
    with timed(tm, "blocky_stl_ms"):
        bm, _ = mesh.blocky_stl(model, out / "blocky.stl")
    with timed(tm, "smooth_stl_ms"):
        _, _, sinfo = mesh.smooth_stl(model, out / "smooth.stl")
    with timed(tm, "payload_ms"):
        payload = model.payload()
        (out / "payload.json").write_text(json.dumps(payload))
    tm["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    with timed(tm, "debug_png_ms"):
        save_debug(out / "debug.png", p.rgb, p.Z_raw, p.mask, p.mask_full, p.cam.K, res, model,
                   f"{Path(a.image).name}  scale=estimate  method={method} ({why})")
    with timed(tm, "views_png_ms"):
        cv2.imwrite(str(out / "views.png"), cv2.cvtColor(render_grid(payload, 300, plane=True), cv2.COLOR_RGB2BGR))
    meta = {"image": str(a.image), "perception_key": p.key, "perception_cached": p.cached,
            "camera": {"hfov_long_deg": p.cam.hfov_deg, "source": p.cam.source, "f35": p.cam.f35,
                       "size": [p.cam.width, p.cam.height]},
            "method": method, "method_reason": why, "thickness": a.thickness, "scale_source": model.scale_source,
            "voxel_mm": round(model.voxel_cm * 10, 3), "measurements_cm": model.measurements_cm,
            "cleanup": model.cleanup_info, "blocky": {"watertight": bool(bm.is_watertight), "faces": len(bm.faces)},
            "smooth": sinfo, "surface_voxels": len(payload["observed"]), "solid_voxels": payload["n_solid"],
            "timings_ms": tm}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    print(f"run: {out}")
    print(f"camera {p.cam.width}x{p.cam.height}, fov {p.cam.hfov_deg:.1f} deg ({p.cam.source}); "
          f"perception {'cached' if p.cached else 'computed'} ({p.key})")
    print(f"method {method} ({why}), t={a.thickness}, scale estimate (DA raw: expect 3-11x too big)")
    print(f"voxel {model.voxel_cm * 10:.2f} mm, dims cm {model.measurements_cm}, solid {payload['n_solid']}, "
          f"surface {len(payload['observed'])}")
    print(f"blocky watertight={bm.is_watertight} faces={len(bm.faces)}; smooth {sinfo}")
    print("timings ms: " + ", ".join(f"{k} {v:.0f}" for k, v in tm.items()))
    return out


if __name__ == "__main__":
    main()
