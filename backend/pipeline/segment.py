from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

log = logging.getLogger("pipeline.segment")

REMBG_MODEL = "u2net"
_session = None
_lock = threading.Lock()


def get_session():
    global _session
    with _lock:
        if _session is None:
            from rembg import new_session
            _session = new_session(REMBG_MODEL, providers=["CPUExecutionProvider"])
    return _session


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def clean_mask(alpha: np.ndarray, erode_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """alpha (H,W) uint8 -> (eroded mask, uneroded mask), both bool."""
    m = (alpha > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    full = largest_component(m)
    if erode_px > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        er = cv2.erode(full.astype(np.uint8), ke).astype(bool)
    else:
        er = full.copy()
    return er, full


def segment(rgb: np.ndarray, erode_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    from rembg import remove
    sess = get_session()
    out = remove(Image.fromarray(rgb), session=sess, only_mask=True)
    alpha = np.asarray(out, dtype=np.uint8)
    if alpha.ndim == 3:
        alpha = alpha[..., -1]
    return clean_mask(alpha, erode_px)
