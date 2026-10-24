"""debug.png: photo | depth | mask + rim band | plane/axis overlay.  views.png: render.render_grid (4 views x
photo colors / brass=observed, steel=inferred)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .intrinsics import project, unproject
from .shapes.common import contour_band, mask_pixels


def shape_plane(res):
    """(n, d, label) of the plane a shape built from, camera frame: mirror's symmetry plane, else inflate's midplane."""
    if "symmetric_plane" in res.debug:
        n, d = res.debug["symmetric_plane"]
        return np.asarray(n), float(d), "symmetry plane"
    if "midplane" in res.debug:
        m = res.debug["midplane"]
        return np.asarray(m.n), float(m.d), "rim midplane" + (" (fallback)" if m.fallback else "")
    return None


def save_debug(path, rgb, Z, mask, mask_full, K, res, model, title=""):
    fig, ax = plt.subplots(1, 4, figsize=(20, 7.2))
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    ax[0].imshow(rgb)
    ax[0].set_title(f"photo {rgb.shape[1]}x{rgb.shape[0]}")

    zm = np.where(mask, Z, np.nan)
    lo, hi = np.nanpercentile(zm, 1), np.nanpercentile(zm, 99)
    im = ax[1].imshow(zm, cmap="turbo", vmin=lo, vmax=hi)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="Z cm (P1-P99)")
    ax[1].set_title(f"depth in mask, median {np.nanmedian(zm):.1f} cm")

    band = contour_band(mask_full)
    ov = rgb / 255 * 0.5
    ov[mask] = ov[mask] * 0.5 + np.array([0.2, 0.6, 1.0]) * 0.5
    ov[mask_full & ~mask] = [1.0, 0.85, 0.1]           # erosion ring: depth not trusted
    ov[band & mask] = [1.0, 0.2, 0.1]                  # rim band (midplane / plane fit input)
    ax[2].imshow(ov)
    ax[2].set_title(f"mask {int(mask.sum())} px (blue), erosion (yellow), rim band (red)")

    ax[3].imshow(rgb.mean(-1), cmap="gray", alpha=0.45)
    ttl = []
    pl = shape_plane(res)
    if pl is not None:
        n, d, label = pl
        us, vs = mask_pixels(mask)
        k = max(1, len(us) // 20000)
        us, vs = us[::k], vs[::k]
        s = unproject(K, us, vs, Z[vs, us].astype(np.float64)) @ n + d    # + = front point behind the plane
        lim = max(0.5, float(np.percentile(np.abs(s), 95)))
        sc = ax[3].scatter(us, vs, c=s, s=1, cmap="coolwarm", vmin=-lim, vmax=lim)
        fig.colorbar(sc, ax=ax[3], fraction=0.046, label=f"cm from {label} (+ = behind)")
        ttl.append(f"{label}, n={np.round(n, 2).tolist()}")
    if "axis" in res.debug:
        a = res.debug["axis"]
        A0, dd, sv = np.asarray(a["A0"]), np.asarray(a["d"]), np.asarray(a["s"])
        P = A0 + dd * np.array([sv[0], sv[-1]])[:, None]
        u, v = project(K, P)
        ax[3].plot(u, v, "-", color="lime", lw=2)
        ttl.append("revolve axis (lime)")
    ax[3].set_xlim(0, rgb.shape[1]), ax[3].set_ylim(rgb.shape[0], 0)
    ax[3].set_title(" | ".join(ttl) or "no plane / axis")

    fig.suptitle(f"{title}\n{model.method} t={model.thickness} voxel {model.voxel_cm * 10:.2f} mm  "
                 f"dims cm {model.measurements_cm}  solid {int(model.occ.sum())}  "
                 f"observed {model.observed.sum() / max(1, model.occ.sum()):.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
