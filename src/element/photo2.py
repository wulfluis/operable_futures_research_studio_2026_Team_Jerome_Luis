"""What the photo of the whole element shows about the crack: where it is on the element, how large it is next to it,
its shape, and (for test cases only) an estimated width (D61, D62). Spec: spec/features/photo2.md.

The element's frame: SAM 3.1's outline of the element holding the crack when SAM ran (D74), else the whole photo, as
the page asks the resident to frame the whole element (an assumption, said so). Every branch of the crack is kept (the
network), not only its longest line, so the 3D model shows the crack as photo 2 does (D74). The crack model was trained on close-ups (D48): on a wide photo it can mark frames
and edges, so this is a place, an extent and a shape; a width only as a labelled estimate for test cases (D62).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

END_ZONE = 0.20  # share of the height (column, wall) or length (beam) at each end that counts as an end zone (illustrative)
MAX_POINTS = 120
MAX_NETWORK_POINTS = 600  # all branches together, for the 3D model
MIN_BRANCH_SHARE = 0.03  # branches shorter than this share of the photo's long side are left out (illustrative)
WHOLE_PHOTO = "the whole photo (the element assumed to fill it)"
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
    network_uv: list | None = None  # every branch: [[[u, v], ...], ...] in the element's frame (D74)
    frame: tuple | None = None  # the element in the photo: (x0, y0, x1, y1) as shares of its width and height
    frame_source: str = WHOLE_PHOTO
    source: str = "photo 2's own crack model"  # or photo 1's crack carried across by the link (D75)
    path_px: np.ndarray | None = field(default=None, repr=False)  # the longest line, photo pixels
    network_px: list | None = field(default=None, repr=False)  # every branch, photo pixels

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
                "length_px": r(self.length_px, 1), "shape": self.shape, "network_uv": self.network_uv,
                "frame": None if self.frame is None else [round(c, 4) for c in self.frame],
                "frame_source": self.frame_source, "source": self.source,
                "assumption": f"the element is {self.frame_source}; place, extent and shape, never a measured width"}

    def reframe(self, frame: tuple | None, source: str) -> None:
        """Read the crack again relative to the element's outline in the photo (shares of the photo), D74."""
        if self.path_px is None or not self.size_px:
            return
        new = place(self.path_px, self.size_px, frame, self.network_px)
        for k in ("path_uv", "extent", "centre_uv", "reason", "network_uv", "frame"):
            setattr(self, k, getattr(new, k))
        self.frame_source = source if frame is not None else WHOLE_PHOTO


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


def _to_uv(px: np.ndarray, size: tuple[int, int], frame: tuple) -> np.ndarray:
    """Photo pixels to the element's frame: u 0 (left) to 1 (right), v 0 (bottom) to 1 (top), clipped to it."""
    w, h = size
    x0, y0, x1, y1 = frame
    u = (px[:, 0] / max(w - 1, 1) - x0) / max(x1 - x0, 1e-6)
    v = 1 - (px[:, 1] / max(h - 1, 1) - y0) / max(y1 - y0, 1e-6)
    return np.clip(np.column_stack([u, v]), 0.0, 1.0)


def place(path_px: np.ndarray | None, size: tuple[int, int], frame: tuple | None = None,
          network_px: list | None = None) -> CrackOnElement:
    """The crack (pixels of a photo of size (w, h)) in the element's coordinates. frame: the element's outline box
    in the photo as shares (x0, y0, x1, y1); None: the whole photo. network_px: every branch; None: the line only."""
    if path_px is None or len(path_px) < 2:
        return CrackOnElement(False, reason="No crack line found on the photo of the element: its place is not known.")
    w, h = size
    f = tuple(float(c) for c in frame) if frame is not None else (0.0, 0.0, 1.0, 1.0)
    path_px = np.asarray(path_px, float)
    uv = _to_uv(path_px, size, f)
    if len(uv) > MAX_POINTS:
        uv = uv[np.linspace(0, len(uv) - 1, MAX_POINTS).round().astype(int)]
    branches = [np.asarray(b, float) for b in (network_px or [path_px]) if len(b) >= 2]
    total = sum(len(b) for b in branches)
    step = max(1, int(np.ceil(total / MAX_NETWORK_POINTS)))
    net = []
    for b in branches:
        idx = sorted(set(range(0, len(b), step)) | {len(b) - 1})
        net.append(np.round(_to_uv(b[idx], size, f), 4).tolist())
    pts = np.vstack([np.asarray(n) for n in net])
    extent = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    return CrackOnElement(True, np.round(uv, 4).tolist(), extent, centre,
                          f"On the photo of the element the crack spans about {extent:.0%} of it.", size_px=(w, h),
                          network_uv=net, frame=f, path_px=path_px, network_px=branches)


OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_network(mask: np.ndarray, min_len_px: float) -> list[np.ndarray]:
    """Every branch of the crack: the mask's centre line cut at its ends and junctions into polylines (x, y pixels),
    each at least min_len_px long. A closed loop is kept as one polyline."""
    from scipy import ndimage

    from measure.crack import skeletonize
    sk = skeletonize(mask).astype(bool)
    if not sk.any():
        return []
    h, w = sk.shape
    deg = np.where(sk, ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1, 0)
    node = sk & (deg != 2)
    used = set()

    def nbrs(p):
        y, x = p
        return [(y + a, x + b) for a, b in OFFS if 0 <= y + a < h and 0 <= x + b < w and sk[y + a, x + b]]

    def walk(start, nxt):
        path, prev, cur = [start, nxt], start, nxt
        used.add((start, nxt)); used.add((nxt, start))
        while not node[cur] and cur != start:
            cand = [q for q in nbrs(cur) if q != prev and (cur, q) not in used]
            if not cand:
                break
            q = cand[0]
            used.add((cur, q)); used.add((q, cur))
            path.append(q)
            prev, cur = cur, q
        return path

    paths = []
    for p in zip(*np.nonzero(node)):
        p = (int(p[0]), int(p[1]))
        for q in nbrs(p):
            if (p, q) not in used:
                paths.append(walk(p, q))
    rest = sk.copy()
    for path in paths:
        for p in path:
            rest[p] = False
    for p in zip(*np.nonzero(rest)):
        p = (int(p[0]), int(p[1]))
        if rest[p]:
            n = [q for q in nbrs(p) if rest[q]]
            if n:
                path = walk(p, n[0])
                for r in path:
                    rest[r] = False
                paths.append(path)
    lines = []
    for path in paths:
        a = np.array([[x, y] for y, x in path], float)
        if len(a) >= 2 and np.linalg.norm(np.diff(a, axis=0), axis=1).sum() >= min_len_px:
            lines.append(a)
    return lines


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
    network = trace_network(d.mask, MIN_BRANCH_SHARE * max(w, h)) if m is not None else None
    out = place(None if m is None else np.asarray(m.main_path_px, dtype=float), (w, h), None, network)
    out.overlay = red_overlay(work, ndimage.binary_dilation(d.mask, iterations=1))
    if m is not None:
        out.width_px_max, out.width_px_median, out.length_px = m.width_max_mm, m.width_median_mm, m.length_main_mm
        ys, xs = np.nonzero(d.mask)
        out.bbox_px = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        out.shape = m.to_dict()["shape"]
    return out
