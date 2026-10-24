"""Pick a shape method with explicit, logged checks (auto-select-v2).

All candidates are cheap to score, so we score all of them, then decide in a fixed order:
  1. REVOLVE if the revolve fit is strong: row-center straightness AND left/right edge symmetry both pass.
  2. Otherwise MIRROR V2, unless one of its sanity checks fails:
       collapse   - volume_ratio: mirror volume / the volume the silhouette implies (inflate half-thickness)
                  - ray_reach:   fraction of masked pixels whose ray reaches the plane behind the front surface
       violation  - E_occ, E_sil of the optimized plane (plane_fit energy terms)
       degenerate - plane tilt at the edge of its allowed range (plane_fit.TILT_MAX_DEG)
  3. INFLATE only as the fallback when both fail.
VLM hint (decide(..., hint=category)): "round" favors revolve, "bilateral"/"boxy"/"flat" favor mirror. It can only
flip a BORDERLINE revolve decision (a revolve check within REVOLVE_MARGIN of its threshold); it never overrides a
failed mirror sanity check.

"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..intrinsics import pixel_rays
from .common import mask_pixels
from .midplane import inflate_h
from .revolve import robust_line, row_profile

log = logging.getLogger("pipeline.auto_select")

# ---- thresholds (set on the duck + bottle phantoms only; see module docstring)
REVOLVE_STRAIGHT_MIN = 0.616     # duck max 0.245 / bottle min 0.987 -> midpoint
REVOLVE_SYMMETRY_MIN = 0.846     # duck max 0.705 / bottle min 0.987 -> midpoint
REVOLVE_MARGIN = 0.05            # a priori: a revolve check within this of its threshold is "borderline"
MIRROR_VOLUME_RATIO_MIN = 0.488  # 0.5 x duck min 0.976
MIRROR_RAY_REACH_MIN = 0.432     # 0.5 x duck min 0.863
MIRROR_E_OCC_MAX = 1.9e-3         # 2 x duck max 9.4e-4 (above the 9e-4 floor)
MIRROR_E_SIL_MAX = 9e-4           # floor 0.03^2 (2 x duck max 3.3e-4 is below it)
TILT_EDGE_FRAC = 0.95            # a priori: |tilt| >= 95 % of the allowed range = "at the edge"

HINT_FAVORS = {"round": "revolve", "bilateral": "mirror", "boxy": "mirror", "flat": "mirror"}


def reflect_mask(mask: np.ndarray, a: float, b: float) -> np.ndarray:
    """Reflect the mask about the image line u = a*v + b."""
    H, W = mask.shape
    vs, us = np.mgrid[0:H, 0:W].astype(np.float32)
    n = np.array([1.0, -a]) / np.hypot(1.0, a)            # normal in (u, v)
    dist = (us - (a * vs + b)) * n[0]
    map_u = (us - 2 * dist * n[0]).astype(np.float32)   # cv2.remap needs float32 maps
    map_v = (vs - 2 * dist * n[1]).astype(np.float32)
    return cv2.remap(mask.astype(np.uint8), map_u, map_v, cv2.INTER_NEAREST, borderValue=0).astype(bool)


def score_revolve(mask_full: np.ndarray) -> dict:
    """straightness: fraction of mask rows whose center lies within max(1.5 px, 4 % of the half-width) of the
    robust axis line; symmetry: IoU of the mask with its reflection about that line."""
    v, l, r, c, hw = row_profile(mask_full)
    a, b = robust_line(v, c)
    resid = np.abs(c - (a * v + b))
    tol = np.maximum(1.5, 0.04 * hw)
    straight = float((resid <= tol).mean())
    refl = reflect_mask(mask_full, a, b)
    sym = float((refl & mask_full).sum() / max(1, (refl | mask_full).sum()))
    return {"straightness": straight, "symmetry": sym, "score": 0.5 * (straight + sym),
            "axis_slope": a, "median_center_resid_px": float(np.median(resid))}


def score_mirror(Z_f, mask, K, mask_full, up_cam=None, prior_scale=1.0) -> dict:
    """Estimate the mirror v2 auto plane and the collapse / violation / degeneracy statistics."""
    from .plane_fit import TILT_MAX_DEG, estimate_plane
    (n, d), info = estimate_plane(Z_f, mask, mask_full, K, up_cam=up_cam, prior_scale=prior_scale)
    us, vs = mask_pixels(mask)
    zf = Z_f[vs, us].astype(np.float64)
    rays = pixel_rays(K, us.astype(np.float64), vs.astype(np.float64))
    den = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = np.where(den > 1e-6, -d / den, -np.inf)
    ok = zc > zf
    foot = (zf / K[0, 0]) * (zf / K[1, 1])                       # pixel footprint (cm^2) at the front surface
    v_mirror = 2.0 * float(np.sum(np.where(ok, zc - zf, 0.0) * foot))
    _, _, h = inflate_h(Z_f, mask, mask_full, K)                  # same pixel order (mask_pixels)
    v_sil = 2.0 * float(np.sum(h * foot))
    tilt = [abs(float(t)) for t in info["tilt_deg"]]
    return {"volume_ratio": v_mirror / max(v_sil, 1e-9), "ray_reach": float(ok.mean()),
            "e_occ": float(info["terms"]["occ"]), "e_sil": float(info["terms"]["sil"]),
            "tilt_max_deg": max(tilt), "tilt_limit_deg": TILT_MAX_DEG, "plane": (n.tolist(), float(d)),
            "plane_info": info}


def _check(name, value, op, thr, group, margin=None):
    ok = value >= thr if op == ">=" else value <= thr
    c = {"check": name, "group": group, "value": round(float(value), 6), "op": op, "threshold": thr, "pass": bool(ok)}
    if margin is not None:
        c["borderline"] = bool(abs(value - thr) < margin)
    return c


def decide(scores: dict, hint: str | None = None) -> dict:
    """Pure decision from precomputed scores (+ optional VLM shape_category). Returns recommended + trace."""
    rv, mi = scores["revolve"], scores["mirror"]
    checks = [
        _check("revolve.straightness", rv["straightness"], ">=", REVOLVE_STRAIGHT_MIN, "revolve", REVOLVE_MARGIN),
        _check("revolve.symmetry", rv["symmetry"], ">=", REVOLVE_SYMMETRY_MIN, "revolve", REVOLVE_MARGIN),
        _check("mirror.volume_ratio", mi["volume_ratio"], ">=", MIRROR_VOLUME_RATIO_MIN, "mirror.collapse"),
        _check("mirror.ray_reach", mi["ray_reach"], ">=", MIRROR_RAY_REACH_MIN, "mirror.collapse"),
        _check("mirror.e_occ", mi["e_occ"], "<=", MIRROR_E_OCC_MAX, "mirror.violation"),
        _check("mirror.e_sil", mi["e_sil"], "<=", MIRROR_E_SIL_MAX, "mirror.violation"),
        _check("mirror.tilt_deg", mi["tilt_max_deg"], "<=", round(TILT_EDGE_FRAC * mi["tilt_limit_deg"], 2),
               "mirror.degenerate"),
    ]
    rchecks = [c for c in checks if c["group"] == "revolve"]
    mchecks = [c for c in checks if c["group"].startswith("mirror")]
    revolve_ok = all(c["pass"] for c in rchecks)
    mirror_ok = all(c["pass"] for c in mchecks)
    # borderline = the revolve decision would flip if a borderline check went the other way
    if revolve_ok:
        borderline = any(c["borderline"] for c in rchecks)
    else:
        failing = [c for c in rchecks if not c["pass"]]
        borderline = all(c["borderline"] for c in failing)
    if revolve_ok:
        geo, why = "revolve", "revolve fit strong (straightness and symmetry pass)"
    elif mirror_ok:
        geo, why = "mirror", "not a solid of revolution; mirror v2 passes every sanity check"
    else:
        bad = ", ".join(c["check"] for c in mchecks if not c["pass"])
        geo, why = "inflate", f"not a solid of revolution and mirror v2 failed: {bad}"
    final, hint_effect = geo, "none"
    fav = HINT_FAVORS.get(hint or "")
    if hint is None:
        hint_effect = "no hint"
    elif fav is None:
        hint_effect = f"category '{hint}' favors nothing"
    elif fav == geo:
        hint_effect = f"agrees ({hint} -> {fav})"
    elif not borderline:
        hint_effect = f"ignored: {hint} favors {fav} but the revolve decision is not borderline"
    elif fav == "mirror" and not mirror_ok:
        hint_effect = f"ignored: {hint} favors mirror but mirror failed a sanity check"
    elif fav == "revolve" and geo in ("mirror", "inflate"):
        final, hint_effect = "revolve", f"override: {hint} favors revolve and the revolve checks are borderline"
    elif fav == "mirror" and geo == "revolve":
        final, hint_effect = "mirror", f"override: {hint} favors mirror and the revolve checks are borderline"
    reason = why if final == geo else f"{why}; Inspector hint '{hint}' flipped a borderline call to {final}"
    return {"recommended": final, "geometric": geo, "reason": reason, "borderline": bool(borderline),
            "selection_trace": {"checks": checks, "revolve_ok": revolve_ok, "mirror_ok": mirror_ok,
                                "geometric": geo, "borderline": bool(borderline), "hint": hint,
                                "hint_favors": fav, "hint_effect": hint_effect, "final": final}}


def auto_select(Z_f, mask, K, mask_full=None, label: str | None = None, up_cam=None) -> dict:
    """label is unused (kept for the CLI signature); the VLM hint is applied later via decide(scores, hint)."""
    t0 = time.perf_counter()
    full = mask if mask_full is None else mask_full
    rv = score_revolve(full)
    mi = score_mirror(Z_f, mask, K, full, up_cam)
    scores = {"revolve": rv, "mirror": {k: v for k, v in mi.items() if k not in ("plane", "plane_info")}}
    out = decide(scores)
    out["scores"] = scores
    out["plane"] = {"n": mi["plane"][0], "d": mi["plane"][1], "info": mi["plane_info"]}
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("auto_select -> %s (%s) %s", out["recommended"], out["reason"],
             [(c["check"], c["value"], c["pass"]) for c in out["selection_trace"]["checks"]])
    return out


def trace_text(auto: dict) -> str:
    """Human-readable selection trace (for the VLM prompt and the UI tooltip)."""
    tr = auto["selection_trace"]
    lines = [f"Chosen method: {tr['final']}. Reason: {auto['reason']}."]
    for c in tr["checks"]:
        lines.append(f"- {c['check']} = {c['value']:.4g} (needs {c['op']} {c['threshold']}): "
                     f"{'pass' if c['pass'] else 'FAIL'}{' (borderline)' if c.get('borderline') else ''}")
    lines.append(f"Order: revolve if both revolve checks pass; else mirror if every mirror check passes; else inflate. "
                 f"Revolve call borderline: {'yes' if tr['borderline'] else 'no'}.")
    return "\n".join(lines)
