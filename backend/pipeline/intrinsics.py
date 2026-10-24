import math

import numpy as np

def build_K(width: int, height: int, fov_long_deg: float) -> np.ndarray:
    long_side = max(width, height)
    f = (long_side / 2.0) / math.tan(math.radians(fov_long_deg) / 2.0)
    return np.array([[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]], dtype=np.float64)

def pixel_rays(K: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, dtype=np.float64)], axis=-1)


def unproject(K: np.ndarray, us: np.ndarray, vs: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return pixel_rays(K, us.astype(np.float64), vs.astype(np.float64)) * Z[:, None]

def project(K: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = P[:, 2]
    return K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]
