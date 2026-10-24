"""Shape assumptions for the hidden side.

Common signature: fn(Z_f, mask, K, rgb, params) -> ShapeResult
  Z_f: (H,W) planar depth, cm.  mask: (H,W) bool (eroded).  K: 3x3.  rgb: (H,W,3) uint8.
  params: dict; common keys: thickness (t), voxel_cm, mask_full (uneroded mask), frame_R.
"""
from .common import Grid, ShapeResult  # noqa: F401
from .inflate import inflate

METHODS = {"inflate": inflate}
