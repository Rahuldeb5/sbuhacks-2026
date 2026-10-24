import math

import numpy as np

def build_K(width: int, height: int, fov_long_deg: float) -> np.ndarray:
    long_side = max(width, height)
    f = (long_side / 2.0) / math.tan(math.radians(fov_long_deg) / 2.0)
    return np.array([[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]], dtype=np.float64)
