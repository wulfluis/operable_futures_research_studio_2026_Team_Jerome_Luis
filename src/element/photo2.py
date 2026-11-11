"""What the photo of the whole element shows about the crack: where it is on the element, how large it is next to it,
its shape, and (for test cases only) an estimated width (D61, D62). Spec: spec/features/photo2.md.

Assumption: the resident framed the whole element (the page asks for it), so the photo is the element and the crack's
place is read relative to the photo. The crack model was trained on close-ups (D48): on a wide photo it can mark frames
and edges, so this is a place, an extent and a shape; a width only as a labelled estimate for test cases (D62).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

END_ZONE = 0.20  # share of the height (column, wall) or length (beam) at each end that counts as an end zone (illustrative)
MAX_POINTS = 120
RED = np.array([214, 39, 40], dtype=float)


@dataclass
class CrackOnElement:
    found: bool
    path_uv: list | None = None  # [[u, v], ...]: u across 0 (left) to 1 (right), v up 0 (bottom) to 1 (top)
    extent: float | None = None  # the larger of the crack's share of the element's width and height
    centre_uv: tuple | None = None
    reason: str = ""
    size_px: tuple | None = None  # (w, h) of the photo as analysed
    width_px_max: float | None = None  # the crack's width on photo 2, in its pixels (for the test estimate, D62)
    width_px_median: float | None = None
    length_px: float | None = None
    bbox_px: tuple | None = None  # (x0, y0, x1, y1) of the crack on photo 2
    shape: dict | None = None  # direction and form, for the pattern rules when photo 1 has no measurement
    overlay: Image.Image | None = field(default=None, repr=False)  # photo 2 with the crack in red, for the wall

    def position(self, element: str | None) -> str:
        """The risk matrix's location: end_zone or middle from where the crack's centre lies; unknown without one."""
        if not self.found or self.centre_uv is None:
            return "unknown"
        u, v = self.centre_uv
        along = u if element == "beam" else v  # a beam runs left-right; a column or a wall up-down
        return "end_zone" if along <= END_ZONE or along >= 1 - END_ZONE else "middle"

    def to_dict(self) -> dict:
        r = lambda v, d=3: None if v is None else round(float(v), d)  # noqa: E731
        return {"found": self.found, "path_uv": self.path_uv, "extent": r(self.extent),
                "centre_uv": None if self.centre_uv is None else [round(c, 3) for c in self.centre_uv],
                "reason": self.reason, "size_px": None if self.size_px is None else list(self.size_px),
                "width_px_max": r(self.width_px_max, 2), "width_px_median": r(self.width_px_median, 2),
                "length_px": r(self.length_px, 1), "shape": self.shape,
                "assumption": "the photo shows the whole element; place, extent and shape, never a measured width"}


def measure_px(grey: np.ndarray, mask: np.ndarray):
    """The crack measurement of the longest piece, in pixels (to_mm = identity)."""
    from measure.analyse import longest_piece
    from measure.crack import measure
    if not mask.any():
        return None
    m = measure(grey, longest_piece(mask), lambda p: np.asarray(p, dtype=float), 0.0)
    return m if m.found and m.main_path_px is not None and len(m.main_path_px) >= 2 else None


def path_from_mask(mask: np.ndarray) -> np.ndarray | None:
    m = measure_px(np.full(mask.shape, 128.0), mask)
    return None if m is None else np.asarray(m.main_path_px, dtype=float)


def place(path_px: np.ndarray | None, size: tuple[int, int]) -> CrackOnElement:
    """The crack line (pixels of a photo of size (w, h)) in the element's coordinates."""
    if path_px is None or len(path_px) < 2:
        return CrackOnElement(False, reason="No crack line found on the photo of the element: its place is not known.")
    w, h = size
    uv = np.column_stack([path_px[:, 0] / max(w - 1, 1), 1 - path_px[:, 1] / max(h - 1, 1)])
    if len(uv) > MAX_POINTS:
        uv = uv[np.linspace(0, len(uv) - 1, MAX_POINTS).round().astype(int)]
    extent = float(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])))
    centre = (float(uv[:, 0].mean()), float(uv[:, 1].mean()))
    return CrackOnElement(True, np.round(uv, 4).tolist(), extent, centre,
                          f"On the photo of the element the crack spans about {extent:.0%} of it.", size_px=(w, h))


def red_overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=float).copy()
    grown = mask
    a[grown] = 0.25 * a[grown] + 0.75 * RED
    return Image.fromarray(a.astype(np.uint8))


def read(model, photo: Image.Image, device: str = "cpu") -> CrackOnElement:
    """The crack model and its written rules on photo 2, then the place, extent, shape and pixel widths."""
    from scipy import ndimage

    from segmentation.decide import CRACK, UNSURE, decide
    from segmentation.predict import probability, working_image
    if model is None:
        return CrackOnElement(False, reason="No crack model: the crack's place is not known.")
    full = ImageOps.exif_transpose(photo).convert("RGB")
    img, (w, h) = working_image(full)
    work = img.crop((0, 0, w, h))
    prob = probability(model, img, device)[:h, :w]
    d = decide(prob, scale=max(w, h) / 416)
    if d.statement not in (CRACK, UNSURE) or not d.mask.any():
        out = CrackOnElement(False, reason="No crack line found on the photo of the element: its place is not known.")
        out.overlay = work
        return out
    m = measure_px(np.asarray(work.convert("L"), dtype=float), d.mask)
    out = place(None if m is None else np.asarray(m.main_path_px, dtype=float), (w, h))
    out.overlay = red_overlay(work, ndimage.binary_dilation(d.mask, iterations=1))
    if m is not None:
        out.width_px_max, out.width_px_median, out.length_px = m.width_max_mm, m.width_median_mm, m.length_main_mm
        ys, xs = np.nonzero(d.mask)
        out.bbox_px = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        out.shape = m.to_dict()["shape"]
    return out
