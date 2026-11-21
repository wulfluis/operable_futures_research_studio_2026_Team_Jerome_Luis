"""Possible spalling on a close-up of a plastered wall (D93): SAM 3.1 asked for "missing plaster" and "broken plaster",
then written rules keep only areas that look like broken-away plaster. Shown only, labelled "unverified": nothing here
changes a level (rules 5, 7). Spec: spec/features/anomaly_ae.md (Possible spalling).

The rules, all illustrative until calibrated on real close-ups of spalling:
- SAM's score at least MIN_SCORE;
- the traced cracks are removed first (SAM also outlines a crack as "broken plaster");
- what is left must be an area, not a line: it survives an erosion of AREA_RADIUS_SHARE of the photo's long side;
- between MIN_AREA_SHARE and MAX_AREA_SHARE of the photo;
- its colour differs clearly from the wall around it (CIELAB difference at least MIN_DELTA_E): broken plaster shows
  the material below; a door frame or a crack's edge looks like the wall."""

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
SIDE = 1024  # px, the photo's long side for SAM
LABEL = "Possible spalling or missing plaster (SAM 3.1, unverified)"


def delta_e(rgb: np.ndarray, area: np.ndarray, around: np.ndarray) -> float:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return float(np.linalg.norm(np.median(lab[area], axis=0) - np.median(lab[around], axis=0)))


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
    if crack is not None and crack.any():
        cand &= ~ndimage.binary_dilation(crack, iterations=max(1, round(0.006 * side)))
    if exclude is not None:
        cand &= ~exclude
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
        around = ndimage.binary_dilation(a, iterations=max(2, round(RING_SHARE * side))) & ~cand
        if crack is not None:
            around &= ~crack
        if exclude is not None:
            around &= ~exclude
        if around.sum() < 50:
            continue
        de = delta_e(rgb, a, around)
        if de >= MIN_DELTA_E:
            out.append({"mask": a, "score": float(best[a].max()), "delta_e": round(de, 1), "share": round(float(share), 4)})
    return sorted(out, key=lambda d: -d["share"])


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
                    "score": round(f["score"], 2), "delta_e": f["delta_e"]})
    return out
