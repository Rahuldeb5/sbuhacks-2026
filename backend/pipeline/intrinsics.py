from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

from PIL import Image, ImageOps

DEFAULT_HFOV_DEG = 69.0 # guestimate of theta for a 36mm long side with focal length 26 mm
MAX_LONG_SIDE = 1024
EXIF_FOCAL_35MM = 0xA405 
EXIF_IFD = 0x8769

@dataclass
class Camera:
    K: np.ndarray   
    width: int
    height: int
    hfov_deg: float 
    source: str     
    f35: float | None = None

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

def build_K(width: int, height: int, fov_long_deg: float) -> np.ndarray:
    long_side = max(width, height)
    f = (long_side / 2.0) / math.tan(math.radians(fov_long_deg) / 2.0)
    return np.array([[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]], dtype=np.float64)


def _exif_f35(img: Image.Image) -> float | None:
    try:
        exif = img.getexif()
        v = exif.get(EXIF_FOCAL_35MM)
        if v is None:
            v = exif.get_ifd(EXIF_IFD).get(EXIF_FOCAL_35MM)
        v = float(v) if v is not None else None
        return v if v and v > 0 else None
    except Exception:
        return None


def load_image(data: bytes | str, max_long_side: int = MAX_LONG_SIDE) -> tuple[np.ndarray, Camera]:
    img = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    f35 = _exif_f35(img)  # read before transpose (exif_transpose strips orientation only)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(1.0, max_long_side / max(w, h))
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    rgb = np.asarray(img, dtype=np.uint8).copy()
    H, W = rgb.shape[:2]
    if f35:
        fov = math.degrees(2 * math.atan(36.0 / (2 * f35)))
        src = "exif"
    else:
        fov, src = DEFAULT_HFOV_DEG, "default"
    return rgb, Camera(build_K(W, H, fov), W, H, fov, src, f35)


def camera_from_array(rgb: np.ndarray, fov_long_deg: float = DEFAULT_HFOV_DEG) -> Camera:
    H, W = rgb.shape[:2]
    return Camera(build_K(W, H, fov_long_deg), W, H, fov_long_deg, "given")


def pixel_rays(K: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, dtype=np.float64)], axis=-1)


def unproject(K: np.ndarray, us: np.ndarray, vs: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return pixel_rays(K, us.astype(np.float64), vs.astype(np.float64)) * Z[:, None]

def project(K: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = P[:, 2]
    return K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]
