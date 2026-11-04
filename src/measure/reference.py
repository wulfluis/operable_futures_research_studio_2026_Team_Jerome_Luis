"""Reference object: turn image points on an object of known size into a pixel -> millimetre mapping.

Spec: spec/features/measurement.md.
- A rectangle (4 corners: scale card, bank card, A4 sheet) gives a homography. It also corrects a camera that is
  not held perpendicular to the wall.
- Two points a known distance apart (scale bar, coin diameter) give a single scale. They assume the camera is
  perpendicular to the wall.
Either mapping holds only in the plane of the reference object, so the crack must be in that plane, close to the
reference [46] ch. 5.2.
"""

import math
from dataclasses import dataclass

import numpy as np

CLICK_ERROR_PX = 1.0  # assumption: a careful click (in the zoomed view) lands within about 1 pixel of the point

# name: (kind, size in mm, description). Sizes are those of the standards named; check every print with a ruler.
PRESETS = {
    "card": ("rect", (85.60, 53.98), "bank or ID card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm"),
    "a4": ("rect", (297.0, 210.0), "A4 sheet, ISO 216, 297 x 210 mm"),
    "a6": ("rect", (148.0, 105.0), "A6 sheet, ISO 216, 148 x 105 mm"),
    "square50": ("rect", (50.0, 50.0), "50 mm square printed on the scale card (proposal, not yet on the card)"),
    "bar50": ("length", 50.0, "50 mm scale bar on the scale card, both ends"),
    "euro1": ("length", 23.25, "1 euro coin, 23.25 mm, two points across its middle"),
}


@dataclass(frozen=True)
class Scale:
    """H maps image pixels (x, y, 1) to millimetres in the plane of the reference object."""
    H: np.ndarray
    kind: str  # rect or length
    points_px: tuple
    size_mm: tuple
    rel_uncertainty: float  # relative error of the scale from the clicks (CLICK_ERROR_PX)
    description: str = ""

    def to_mm(self, xy) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xy, dtype=float))
        q = np.c_[p, np.ones(len(p))] @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def mm_per_px(self, xy, direction=(1.0, 0.0)) -> float:
        """Size in mm of one pixel step from xy along direction (perspective makes it vary across the image)."""
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        a = np.asarray(xy, dtype=float)
        return float(np.linalg.norm(self.to_mm(a + d) - self.to_mm(a)))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "description": self.description, "size_mm": list(self.size_mm),
                "points_px": [[round(float(x), 2), round(float(y), 2)] for x, y in self.points_px],
                "scale_rel_uncertainty": round(self.rel_uncertainty, 4)}


def _normaliser(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0)
    d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def homography(src, dst) -> np.ndarray:
    """Normalised direct linear transform: H with dst ~ H @ src, from 4 or more point pairs."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("homography needs at least 4 point pairs of shape (n, 2)")
    ts, td = _normaliser(src), _normaliser(dst)
    s = np.c_[src, np.ones(len(src))] @ ts.T
    d = np.c_[dst, np.ones(len(dst))] @ td.T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(td) @ h @ ts
    return H / H[2, 2]


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def from_rectangle(corners_px, size_mm, description: str = "", long_first: bool | None = None) -> Scale:
    """Four corners in order around the rectangle (either direction, any starting corner).
    long_first: the side from corner 0 to corner 1 is the long side; None means the longer side in the photo is
    matched to the longer side in mm (the card detector sets it from the camera pose instead)."""
    q = np.asarray(corners_px, dtype=float)
    if q.shape != (4, 2):
        raise ValueError("a rectangle needs exactly 4 corners")
    turns = [_cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError("the 4 corners do not form a convex shape: click them in order around the rectangle")
    side = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    if min(side) < 10:
        raise ValueError("reference is smaller than 10 pixels: move closer")
    long_mm, short_mm = max(size_mm), min(size_mm)
    if long_first is None:
        long_first = side[0] + side[2] >= side[1] + side[3]
    if long_first:
        dst = [(0, 0), (long_mm, 0), (long_mm, short_mm), (0, short_mm)]
    else:
        dst = [(0, 0), (short_mm, 0), (short_mm, long_mm), (0, long_mm)]
    rel = CLICK_ERROR_PX * math.sqrt(2) / min(side)
    return Scale(homography(q, dst), "rect", tuple(map(tuple, q)), (long_mm, short_mm), rel, description)


def from_length(p1, p2, length_mm: float, description: str = "") -> Scale:
    """Two points a known distance apart; one scale for the whole image (camera perpendicular to the wall)."""
    a, b = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    d = float(np.linalg.norm(b - a))
    if d < 10:
        raise ValueError("reference is shorter than 10 pixels: move closer")
    if not length_mm > 0:
        raise ValueError("reference length must be positive")
    s = length_mm / d
    H = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]])
    return Scale(H, "length", (tuple(a), tuple(b)), (float(length_mm),), CLICK_ERROR_PX * math.sqrt(2) / d,
                 description)


def from_preset(name: str, points_px) -> Scale:
    kind, size, text = PRESETS[name]
    if kind == "rect":
        return from_rectangle(points_px, size, text)
    return from_length(points_px[0], points_px[1], size, text)
