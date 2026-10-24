"""print/marker_card.pdf: ArUco DICT_4X4_50 id 0 at exactly 60 mm (outer black square) + 50 mm ruler,
in a brass 'calibration plate' frame. US Letter, vector. Also writes a 300-dpi PNG used to self-check
the marker size and detectability."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "print"
PAGE_W, PAGE_H = 215.9, 279.4
MARKER_MM = 60.0
BRASS, BRASS_DK, INK = "#c9a24a", "#7a5a1e", "#2b2116"


def main():
    OUT.mkdir(exist_ok=True)
    bits = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 6)  # 6x6 cells
    fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # plate: 130 x 150 mm, centered
    pw, ph = 130.0, 150.0
    px, py = (PAGE_W - pw) / 2, (PAGE_H - ph) / 2 + 20
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=8", fc=BRASS, ec=BRASS_DK, lw=2))
    ax.add_patch(FancyBboxPatch((px + 5, py + 5), pw - 10, ph - 10, boxstyle="round,pad=0,rounding_size=5",
                                fc="none", ec=BRASS_DK, lw=0.8))
    for cx, cy in ((px + 10, py + 10), (px + pw - 10, py + 10), (px + 10, py + ph - 10), (px + pw - 10, py + ph - 10)):
        ax.add_patch(Circle((cx, cy), 2.6, fc="#e8cf8a", ec=BRASS_DK, lw=1))
        ax.add_patch(Circle((cx - 0.7, cy + 0.7), 0.9, fc="#fff5d6", ec="none"))

    # white quiet zone (>= 1 cell = 10 mm) + marker at exactly 60 mm
    mx, my = PAGE_W / 2 - MARKER_MM / 2, py + ph - 32 - MARKER_MM
    q = 10.0
    ax.add_patch(Rectangle((mx - q, my - q), MARKER_MM + 2 * q, MARKER_MM + 2 * q, fc="white", ec=INK, lw=0.3))
    cell = MARKER_MM / 6
    for r in range(6):
        for c in range(6):
            if bits[r, c] == 0:  # black cell; row 0 is the TOP of the marker
                ax.add_patch(Rectangle((mx + c * cell, my + (5 - r) * cell), cell, cell, fc="black", ec="none", lw=0))

    ax.text(PAGE_W / 2, py + ph - 12, "CALIBRATION PLATE", ha="center", va="center", fontsize=13,
            family="serif", weight="bold", color=INK)

    # 50 mm ruler bar on white
    rx, ry = PAGE_W / 2 - 25, py + 30
    ax.add_patch(Rectangle((rx - 4, ry - 9), 58, 21, fc="white", ec=INK, lw=0.3))
    ax.add_patch(Rectangle((rx, ry), 50, 3, fc=INK, ec="none"))
    for mm in range(51):
        h = 6 if mm % 10 == 0 else (4.5 if mm % 5 == 0 else 2.5)
        ax.plot([rx + mm, rx + mm], [ry + 3, ry + 3 + h], color=INK, lw=0.35 if mm % 5 else 0.6)
        if mm % 10 == 0:
            ax.text(rx + mm, ry - 2.5, str(mm), ha="center", va="top", fontsize=6, color=INK)
    ax.text(PAGE_W / 2, ry - 7.2, "this bar must measure exactly 50 mm", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(PAGE_W / 2, py + 14, "ArUco DICT_4X4_50 · id 0 · marker 60 mm (outer black square)",
            ha="center", va="center", fontsize=7.5, family="monospace", color=INK)

    ax.text(PAGE_W / 2, py - 12, "Print at 100 % / Actual size (no 'fit to page'). Check the ruler, then lay the card FLAT "
            "on the table\nnext to the object, fully visible in the photo. Shoot from 20–40° above the table.",
            ha="center", va="top", fontsize=8, color=INK, linespacing=1.5)
    # cut line
    ax.add_patch(Rectangle((px - 6, py - 6), pw + 12, ph + 12, fc="none", ec="#999999", lw=0.5, ls=(0, (3, 3))))
    fig.savefig(OUT / "marker_card.pdf")
    fig.savefig(OUT / "marker_card_check_300dpi.png", dpi=300)
    plt.close(fig)

    # self-check: detect + measure the marker in the 300 dpi raster
    img = cv2.imread(str(OUT / "marker_card_check_300dpi.png"))
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert ids is not None and 0 in ids.ravel(), "marker not detected in the check raster"
    c = corners[list(ids.ravel()).index(0)].reshape(4, 2)
    side_px = sum(((c[i] - c[(i + 1) % 4]) ** 2).sum() ** 0.5 for i in range(4)) / 4
    print(f"detected id 0; side {side_px:.1f} px at 300 dpi = {side_px / 300 * 25.4:.2f} mm (expect 60.00)")


if __name__ == "__main__":
    main()
