"""VLM "Inspector's notes": photo + depth heatmap + mask overlay + auto_select's selection trace -> strict JSON.

Validation: JSON object with exactly the required keys and enums; name 2-4 words. One retry (with the error
fed back) on invalid output; 20 s total budget; fallback name "Unidentified contraption".
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time

import cv2
import numpy as np

log = logging.getLogger("vlm")
FALLBACK_NAME = "Unidentified contraption"
CATEGORIES = ("round", "bilateral", "boxy", "flat", "other")
METHODS = ("mirror", "revolve", "inflate")
TIMEOUT_S = 20.0

SYSTEM = ("You are the inspector of a steampunk workshop that turns ONE photo of an object into a voxel model. "
          "You are given three images of the same scan: (1) the photo, (2) a depth heatmap (bright = near the "
          "camera), (3) the segmentation mask overlay (the object tinted brass). You are also given the numbers our "
          "shape selector computed. Reply with ONLY a JSON object, no markdown, no extra text.")

SCHEMA = ('{"name": "2-4 words", "description": "1-2 sentences about the object", '
          '"shape_category": "round | bilateral | boxy | flat | other", '
          '"method_explanation": "1-2 sentences explaining why the CHOSEN method was picked, citing the checks in '
          'the selection trace (their values and thresholds); do not invent other reasons", '
          '"suggested_method": "mirror | revolve | inflate"}')

CATEGORY_DOC = ("shape_category, judged from the PHOTO ONLY (ignore the selector numbers for this field): "
                "round = rotationally symmetric around one axis (bottle, cup, vase, can, ball); "
                "bilateral = left/right mirror-symmetric with a distinct front and back (animals, toys, figurines, "
                "vehicles, shoes, chairs); boxy = mostly flat faces and right angles (box, book, brick, gadget); "
                "flat = much thinner than it is wide and tall (coin, card, leaf, plate); other = none of these.")

METHOD_DOC = ("Methods: revolve = solid of revolution around a vertical axis (for bottles, cups, vases); "
              "mirror = reflect the visible half across a symmetry plane (for bilaterally symmetric things); "
              "inflate = puff the outline up like a balloon (fallback when neither assumption fits).")


def _data_url(img_bgr: np.ndarray, long_side: int = 512) -> str:
    h, w = img_bgr.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def validate(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object found")
    d = json.loads(m.group(0))
    req = ("name", "description", "shape_category", "method_explanation", "suggested_method")
    missing = [k for k in req if k not in d]
    if missing:
        raise ValueError(f"missing keys {missing}")
    for k in req:
        if not isinstance(d[k], str) or not d[k].strip():
            raise ValueError(f"{k} must be a non-empty string")
    d = {k: d[k].strip() for k in req}
    n = len(d["name"].split())
    if not 2 <= n <= 4:
        raise ValueError(f"name must be 2-4 words, got {n}")
    d["shape_category"] = d["shape_category"].lower()
    d["suggested_method"] = d["suggested_method"].lower()
    if d["shape_category"] not in CATEGORIES:
        raise ValueError(f"shape_category must be one of {CATEGORIES}")
    if d["suggested_method"] not in METHODS:
        raise ValueError(f"suggested_method must be one of {METHODS}")
    return d


def fallback(reason: str) -> dict:
    return {"name": FALLBACK_NAME, "description": "", "shape_category": "other", "method_explanation": "",
            "suggested_method": "", "fallback": reason}


async def describe(manager, photo_bgr, depth_bgr, mask_bgr, auto: dict, timeout_s: float = TIMEOUT_S) -> dict:
    """Returns {"result": validated dict or fallback, "meta": {...}}."""
    t0 = time.perf_counter()
    images = [{"type": "image", "data_url": _data_url(x)} for x in (photo_bgr, depth_bgr, mask_bgr)]
    from pipeline.shapes.auto_select import trace_text
    user = (f"{METHOD_DOC}\n{CATEGORY_DOC}\n\nSelection trace (use these exact numbers):\n{trace_text(auto)}\n\n"
            f"Return JSON with exactly these keys: {SCHEMA}")
    base = {"system_prompt": SYSTEM, "temperature": 0.2, "max_output_tokens": 300, "store": False}
    attempts, err = [], None
    for attempt in range(2):
        remaining = timeout_s - (time.perf_counter() - t0)
        if remaining <= 0.5:
            err = "timeout"
            break
        text = user if attempt == 0 else (user + f"\n\nYour previous reply was invalid ({err}). Reply with ONLY "
                                                 f"the JSON object, exactly those 5 keys.")
        body = {**base, "input": [{"type": "text", "content": text}, *images]}
        try:
            import asyncio
            r = await asyncio.wait_for(manager.chat(body, timeout=remaining), timeout=remaining)
            out = " ".join(o.get("content", "") for o in r.get("output", []) if o.get("type") == "message")
            attempts.append({"stats": r.get("stats"), "raw": out[:1200]})
            res = validate(out)
            return {"result": res, "meta": {"ms": round((time.perf_counter() - t0) * 1000), "attempts": attempts,
                                            "agrees_with_auto": res["suggested_method"] == auto["recommended"]}}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
            log.warning("VLM attempt %d failed: %s", attempt + 1, err)
            if "Timeout" in type(e).__name__ or "unavailable" in str(e):
                break
    return {"result": fallback(err or "unknown"), "meta": {"ms": round((time.perf_counter() - t0) * 1000),
                                                          "attempts": attempts, "error": err}}
