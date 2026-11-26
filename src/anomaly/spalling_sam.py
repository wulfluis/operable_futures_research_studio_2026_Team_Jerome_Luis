"""Possible spalling on a close-up of a plastered wall (D93): SAM 3.1 asked for "missing plaster" and "broken plaster",
then written rules keep only areas that look like broken-away plaster. Shown only, labelled "unverified": nothing here
changes a level (rules 5, 7). Spec: spec/features/anomaly_ae.md (Possible spalling).

The rules, all illustrative until calibrated on real close-ups of spalling:
- SAM's score at least MIN_SCORE;
- the traced cracks' thin lines are removed first (SAM also outlines a crack as "broken plaster"); a wide area
  the crack model marked stays (D111: the crack model often marks the spall itself as a very wide crack);
- what is left must be an area, not a line: it survives an erosion of AREA_RADIUS_SHARE of the photo's long side;
- between MIN_AREA_SHARE and MAX_AREA_SHARE of the photo;
- its colour differs clearly from the wall around it (CIELAB difference at least MIN_DELTA_E): broken plaster shows
  the material below; a door frame or a crack's edge looks like the wall;
- not an open crack's dark gap (D111): not much darker than the wall (MAX_DARKER), and not a small, long, thin and
  dark strip (STRIP_*)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

PROMPTS = ["missing plaster", "broken plaster"]
MIN_SCORE = 0.5
AREA_RADIUS_SHARE = 0.012
MIN_AREA_SHARE = 0.005
MAX_AREA_SHARE = 0.40
RING_SHARE = 0.02
MIN_DELTA_E = 15.0  # 21 Nov, 12 photos: real spalls 16 to 18.5, a door frame 13.3 (illustrative)
# D111, an open crack's dark gap is not a spall (26 Nov, 76 real cases checked by eye; illustrative until calibrated):
MAX_DARKER = 35.0  # much darker than the wall around it: a shadow in a gap, not broken plaster
STRIP_ELONGATION = 3.5  # a long, thin area ...
STRIP_DARKER = 18.0  # ... clearly darker than the wall ...
STRIP_MAX_SHARE = 0.05  # ... and small: an open crack, not a spall
# D114: exposed concrete in shade can be close in colour to the plaster around it; it is rougher (26 Nov, real
# columns). Illustrative until calibrated:
ROUGH_MIN_SCORE = 0.75  # SAM sure of it ...
ROUGH_MIN_RATIO = 1.5  # ... and the area this many times rougher than the wall around it
# D114: a spall's edge is broken; a door frame or panel is a straight shape. SAM fairly sure and a jagged outline
# (area / convex hull area at most JAGGED_MAX_SOLIDITY) is enough, whatever the colour (26 Nov, two real cases)
JAGGED_MIN_SCORE = 0.6
JAGGED_MAX_SOLIDITY = 0.85
SIDE = 1024  # px, the photo's long side for SAM
LABEL = "Possible spalling or missing plaster (SAM 3.1, unverified)"


def delta_e(rgb: np.ndarray, area: np.ndarray, around: np.ndarray) -> float:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return float(np.linalg.norm(np.median(lab[area], axis=0) - np.median(lab[around], axis=0)))


def near_of(a: np.ndarray, side: int) -> np.ndarray:
    return ndimage.binary_dilation(a, iterations=max(2, round(RING_SHARE * side))) | a


def solidity(area: np.ndarray) -> float:
    """Area / convex hull area of the largest outline: 1 for a rectangle, lower for a broken, jagged edge."""
    cs, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 0 else 1.0


def roughness_ratio(rgb: np.ndarray, area: np.ndarray, around: np.ndarray) -> float:
    """How much rougher the area is than the wall around it: median fine detail (grey minus its blur) inside / around."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))
    return float(np.median(hp[area]) / max(1.0, float(np.median(hp[around]))))  # a floor: a flat wall has no detail


def shape_and_light(rgb: np.ndarray, area: np.ndarray, around: np.ndarray) -> tuple[float, float]:
    """How long and thin the area is (long side / short side of its smallest rectangle) and how much darker it is than
    the wall around it (CIELAB lightness, 0 to 100; above 0: darker)."""
    cs, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (a, b), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    el = max(a, b) / max(1.0, min(a, b))
    L = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    return float(el), float(np.median(L[around]) - np.median(L[area]))


def areas(rgb: np.ndarray, sam_out: list, crack: np.ndarray | None = None, exclude: np.ndarray | None = None) -> list:
    """The spalling areas, largest first: each a dict with its mask, score and colour difference."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    cand = np.zeros((h, w), bool)
    best = np.zeros((h, w), np.float32)
    for word, m, s in sam_out:
        if word in PROMPTS and s >= MIN_SCORE:
            cand |= m
            best = np.maximum(best, m * s)
    sam_area = cand.copy()  # what SAM outlined, before the cracks are taken out (D111: the outline drawn)
    if crack is not None and crack.any():
        # D111: only the crack's thin lines are removed; a wide area the crack model also marked (often the spall
        # itself, which it reads as a very wide crack) stays a candidate
        r0 = max(1, round(AREA_RADIUS_SHARE * side))
        wide = ndimage.binary_dilation(ndimage.binary_erosion(crack, iterations=r0), iterations=r0) & crack
        cand &= ~ndimage.binary_dilation(crack & ~wide, iterations=max(1, round(0.006 * side)))
    if exclude is not None:
        cand &= ~exclude
        sam_area &= ~exclude
    r = max(1, round(AREA_RADIUS_SHARE * side))
    core = ndimage.binary_erosion(cand, iterations=r)
    if not core.any():
        return []
    cand = ndimage.binary_dilation(core, iterations=r) & cand  # an opening: lines and thin strips are gone
    labels, n = ndimage.label(cand)
    out = []
    for i in range(1, n + 1):
        a = labels == i
        share = a.mean()
        if not MIN_AREA_SHARE <= share <= MAX_AREA_SHARE:
            continue
        around = ndimage.binary_dilation(a, iterations=max(2, round(RING_SHARE * side))) & ~cand & ~sam_area  # D114: outside SAM's whole area
        if crack is not None:
            around &= ~crack
        if exclude is not None:
            around &= ~exclude
        if around.sum() < 50:
            continue
        de = delta_e(rgb, a, around)
        el, dl = shape_and_light(rgb, a, around)
        gap = dl >= MAX_DARKER or (el >= STRIP_ELONGATION and dl >= STRIP_DARKER and share < STRIP_MAX_SHARE)
        rough = float(best[a].max()) >= ROUGH_MIN_SCORE and roughness_ratio(rgb, a, around) >= ROUGH_MIN_RATIO
        jagged = float(best[a].max()) >= JAGGED_MIN_SCORE and solidity(sam_area & near_of(a, side)) <= JAGGED_MAX_SOLIDITY
        if (de >= MIN_DELTA_E or rough or jagged) and not gap:  # D114: or SAM sure and rougher, or a broken edge
            # with the crack lines along its rim, but only near it: SAM may outline a crack's tail too (D111)
            near = ndimage.binary_dilation(a, iterations=max(2, round(RING_SHARE * side)))
            whole = (sam_area & near) | a
            out.append({"mask": whole, "score": float(best[a].max()), "delta_e": round(de, 1), "share": round(float(share), 4),
                        "elongation": round(el, 1), "darker": round(dl, 1)})
    return sorted(out, key=lambda d: -d["share"])


PATCH_BAND_MIN_SHARE = 0.03  # D114: a long area this large and unlike the wall (colour or roughness) is a patch
PATCH_MAX_ELONGATION = 5.0  # D113: a patch is a blob, not a long band along a crack (illustrative)
PATCH_LABEL = "Possible spalling: a wide patch the crack model marked (written rule, unverified)"


def wide_patches(rgb: np.ndarray, crack: np.ndarray) -> list:
    """D113: areas of the crack model's mask too wide to be a crack line (they survive an erosion of AREA_RADIUS_SHARE
    of the long side) and not an open crack's black hole (less than MAX_DARKER darker than the wall around them):
    a spalled patch or broken plaster the crack model painted. As defect dicts like find(); no SAM needed."""
    if crack is None or not crack.any():
        return []
    h, w = crack.shape
    side = max(h, w)
    r = max(1, round(AREA_RADIUS_SHARE * side))
    core = ndimage.binary_erosion(crack, iterations=r)
    if not core.any():
        return []
    wide = ndimage.binary_dilation(core, iterations=r + 1) & crack
    labels, n = ndimage.label(wide)
    out = []
    for i in range(1, n + 1):
        a = labels == i
        if a.mean() < MIN_AREA_SHARE:  # D114: a small piece is a wide stretch of crack, not a patch
            continue
        around = ndimage.binary_dilation(a, iterations=max(2, round(RING_SHARE * side))) & ~crack
        if around.sum() < 50:
            continue
        el, dl = shape_and_light(rgb[:h, :w], a, around)
        if dl >= MAX_DARKER:  # a black hole of an open crack, kept as crack (spalled concrete is often darker too)
            continue
        if el >= STRIP_ELONGATION and dl >= STRIP_DARKER and a.mean() < STRIP_MAX_SHARE:  # an open crack's dark strip
            continue
        differs = delta_e(rgb[:h, :w], a, around) >= MIN_DELTA_E or roughness_ratio(rgb[:h, :w], a, around) >= ROUGH_MIN_RATIO
        if el >= PATCH_MAX_ELONGATION and not (a.mean() >= PATCH_BAND_MIN_SHARE and differs):
            continue  # a thick band along a crack line, kept as crack; a large band unlike the wall is a patch (D114)
        cs, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2.0, True).reshape(-1, 2)
        ys, xs = np.nonzero(a)
        out.append({"kind": "spalling", "label": PATCH_LABEL, "share": round(float(a.mean()), 4),
                    "centre": [round(float(xs.mean()) / w, 5), round(float(ys.mean()) / h, 5)],
                    "outline": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in c],
                    "score": None, "delta_e": None, "elongation": None, "darker": round(dl, 1)})
    return out


def find(photo: Image.Image, crack: np.ndarray | None = None, exclude: np.ndarray | None = None) -> list | None:
    """Possible spalling on the photo, as defect dicts for the crack viewer (kind "spalling", outline in shares of the
    photo); None when SAM 3.1 is not on this laptop. crack, exclude: full-size masks (traced cracks, bank card)."""
    from element.parts import sam_segmenter, shared_predictor
    p = shared_predictor()
    if p is None:
        return None
    im = photo.convert("RGB")
    k = min(1.0, SIDE / max(im.size))
    small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR) if k < 1 else im
    sw, sh = small.size

    def fit(m):
        return None if m is None else np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((sw, sh))) > 0
    rgb = np.asarray(small)
    found = areas(rgb, sam_segmenter(p, PROMPTS)(rgb), fit(crack), fit(exclude))
    out = []
    for f in found:
        cs, _ = cv2.findContours(f["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2.0, True).reshape(-1, 2)
        ys, xs = np.nonzero(f["mask"])
        out.append({"kind": "spalling", "label": LABEL, "share": f["share"],
                    "centre": [round(float(xs.mean()) / sw, 5), round(float(ys.mean()) / sh, 5)],
                    "outline": [[round(float(x) / sw, 5), round(float(y) / sh, 5)] for x, y in c],
                    "score": round(f["score"], 2), "delta_e": f["delta_e"],
                    "elongation": f["elongation"], "darker": f["darker"]})
    return out
