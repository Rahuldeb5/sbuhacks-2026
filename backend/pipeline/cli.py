"""python -m pipeline.cli --image X.jpg [--method auto|mirror|revolve|inflate] [--thickness 1.0] [--plane-offset 0]
                          [--scale auto|estimate|known_dim|marker] [--known-dim h=7.5] [--marker-cm 6]
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
from .perceive import CACHE_DIR
from .render import render_grid
from .run import mirror_from_scan, prepare, shape_from_scan, timed
from .scale import MARKER_CM, parse_known
from .shapes import METHODS
from .shapes.auto_select import trace_text

RUNS = Path(__file__).resolve().parent.parent / "runs"


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="auto", choices=["auto", *METHODS])
    ap.add_argument("--thickness", type=float, default=1.0, help="revolve/inflate (mirror's depth comes from its plane)")
    ap.add_argument("--plane-offset", type=float, default=0.0,
                    help="mirror only: move the auto symmetry plane along its normal, cm (+ = deeper / fatter)")
    ap.add_argument("--scale", default="auto", choices=["auto", "estimate", "known_dim", "marker"])
    ap.add_argument("--known-dim", default=None, help="one real dimension, e.g. h=7.5 (cm; w = image-horizontal)")
    ap.add_argument("--marker-cm", type=float, default=MARKER_CM, help="ArUco 4x4_50 id 0 outer side, cm")
    ap.add_argument("--voxel-mm", type=float, default=2.0, help="minimum voxel; the 128 cap can make it coarser")
    ap.add_argument("--out", default=None, help="run id (default: <time>-<image stem>-<method>)")
    ap.add_argument("--no-cache", action="store_true", help="recompute depth + mask")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")

    known = parse_known(a.known_dim)
    if a.scale == "known_dim" and not known:
        ap.error("--scale known_dim needs --known-dim")
    t0 = time.perf_counter()
    scan = prepare(a.image, a.scale, a.marker_cm, known, None if a.no_cache else CACHE_DIR)
    tm = scan.timings
    method = scan.auto["recommended"] if a.method == "auto" else a.method
    why = scan.auto["reason"] if a.method == "auto" else "requested"
    if a.plane_offset and method != "mirror":
        ap.error(f"--plane-offset only applies to mirror (method is {method})")
    thickness = 1.0 if method == "mirror" else a.thickness
    if method == "mirror":
        res, model = mirror_from_scan(scan, a.plane_offset, {}, tm, a.voxel_mm / 10)
    else:
        res, model = shape_from_scan(scan, method, thickness, tm, a.voxel_mm / 10)

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
    kd = model.cleanup_info.get("known_dim")
    with timed(tm, "debug_png_ms"):
        save_debug(out / "debug.png", scan.rgb, scan.Z_f * (kd["factor"] if kd else 1.0), scan.mask, scan.mask_full,
                   scan.K, res, model, f"{Path(a.image).name}  scale={scan.scale.source}  method={method} ({why})")
    with timed(tm, "views_png_ms"):
        cv2.imwrite(str(out / "views.png"), cv2.cvtColor(render_grid(payload, 300, plane=True), cv2.COLOR_RGB2BGR))
    meta = {"image": str(a.image), "perception_key": scan.key,
            "camera": {"hfov_long_deg": scan.cam.hfov_deg, "source": scan.cam.source, "f35": scan.cam.f35,
                       "size": [scan.cam.width, scan.cam.height]},
            "method": method, "method_reason": why, "auto": {k: v for k, v in scan.auto.items() if k != "plane"},
            "thickness": thickness, "plane": payload.get("plane"), "axis": payload.get("axis"),
            "plane_estimator": res.debug.get("plane_estimator"), "revolve_mode": res.debug.get("revolve_mode"),
            "self_symmetry_iou": model.cleanup_info.get("self_symmetry_iou"),
            "scale_source": scan.scale.source, "scale": scan.scale.to_json(), "known_dim": kd,
            "voxel_mm": round(model.voxel_cm * 10, 3), "measurements_cm": model.measurements_cm,
            "cleanup": model.cleanup_info, "blocky": {"watertight": bool(bm.is_watertight), "faces": len(bm.faces)},
            "smooth": sinfo, "surface_voxels": len(payload["observed"]), "solid_voxels": payload["n_solid"],
            "timings_ms": tm}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    print(f"run: {out}")
    print(f"camera {scan.cam.width}x{scan.cam.height}, fov {scan.cam.hfov_deg:.1f} deg ({scan.cam.source}); "
          f"scale {scan.scale.source}" + (f", known_dim {kd}" if kd else ""))
    print(trace_text(scan.auto))
    print(f"using {method} ({why}), t={thickness}" + (f", plane offset {a.plane_offset} cm" if method == "mirror" else ""))
    print(f"voxel {model.voxel_cm * 10:.2f} mm, dims cm {model.measurements_cm}, solid {payload['n_solid']}, "
          f"surface {len(payload['observed'])}")
    if "self_symmetry_iou" in model.cleanup_info:
        print(f"self-symmetry {model.cleanup_info['self_symmetry_iou']}")
    print(f"blocky watertight={bm.is_watertight} faces={len(bm.faces)}; smooth {sinfo}")
    print("timings ms: " + ", ".join(f"{k} {v:.0f}" for k, v in tm.items()))
    return out


if __name__ == "__main__":
    main()
