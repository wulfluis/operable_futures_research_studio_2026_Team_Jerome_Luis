"""Millimetres without a bank card (D116): a door in photo 2 gives the scale, carried to photo 1 through the link of
the two photos (D66, D75). Spec: spec/features/scale_without_card.md.

An estimate, never a measurement: the door's height is assumed, and the door and the crack may lie at different
depths. Its error is shown with every width. A level read from it can never be Category 1 (rule 3, src/app/core.py).
A person in the photo is not used: photos must not show people (rule 9). All numbers are assumptions."""

from __future__ import annotations

import math

import numpy as np

DOOR_MM = 2100.0  # assumption: the usual height of a door opening; it varies from about 2000 to 2200 mm
DOOR_REL = 0.05  # assumption: ± 105 mm of door heights
DEPTH_REL = 0.15  # assumption: the door and the crack at different depths or angles to the camera
MIN_SCORE = 0.5  # SAM 3.1's score from which a door counts (illustrative, as parts.MIN_CONF)
ASPECT = (1.6, float("inf"))  # taller than wide; an open door seen edge-on is very narrow, only its height is used
EDGE = 0.01  # a door cut by the photo's top or bottom edge is not used: its height is not seen


def door_in(things: list, size_px: tuple[int, int]) -> dict | None:
    """The best whole door among SAM's masks [(part, mask, score)], in photo 2's own pixels (size_px = w, h).
    None when no door is seen whole and upright."""
    w, h = size_px
    best = None
    for part, mask, score in things or []:
        if part != "door" or score < MIN_SCORE or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        mh, mw = mask.shape
        top, bottom = ys.min() / mh, (ys.max() + 1) / mh
        if top <= EDGE or bottom >= 1 - EDGE:
            continue
        hp, wp = (bottom - top) * h, (xs.max() + 1 - xs.min()) / mw * w
        if wp <= 0 or not ASPECT[0] <= hp / wp <= ASPECT[1]:
            continue
        if best is None or score > best["score"]:
            best = {"height_px": float(hp), "width_px": float(wp), "score": round(float(score), 2),
                    "box": [float(xs.min() / mw), float(top), float((xs.max() + 1) / mw), float(bottom)]}
    return best


def scale_from_door(door: dict | None, photo1_to_photo2: float | None) -> dict | None:
    """mm per pixel of photo 1 from the door (photo 2) and the link's local scale (photo 2 pixels per photo 1 pixel at
    the crack). None when either is missing."""
    if door is None or not photo1_to_photo2 or photo1_to_photo2 <= 0:
        return None
    mm_per_px2 = DOOR_MM / door["height_px"]
    rel = math.hypot(DOOR_REL, DEPTH_REL)
    return {"source": "door", "name": "a door in photo 2", "assumed_mm": DOOR_MM,
            "mm_per_px": mm_per_px2 * photo1_to_photo2, "mm_per_px_photo2": mm_per_px2, "rel": round(rel, 3),
            "door": door,
            "basis": f"door assumed {DOOR_MM / 1000:.2f} m high (assumption), ± {rel:.0%} for the door's height and its "
                     "distance from the crack; an estimate, not a measurement"}


def local_scale(carry, line_px: np.ndarray | None, link_k: float, photo2_k: float = 1.0) -> float | None:
    """Photo 2 pixels per photo 1 (full) pixel at the crack, from the link: the crack line's two ends carried across.
    carry: Location.carry (link_grey pixels -> photo 2's own pixels); line_px in link_grey pixels."""
    if carry is None or line_px is None or len(line_px) < 2:
        return None
    c = np.asarray(line_px, float)
    a = carry(c[[0, -1]])
    if a is None:
        return None
    d1 = float(np.linalg.norm(c[-1] - c[0]))
    if d1 < 1e-6:
        return None
    return float(np.linalg.norm(a[1] - a[0])) * photo2_k / d1 * link_k


def widths(scale: dict, width_max_px: float | None, width_median_px: float | None, length_px: float | None,
           unc_px: float = 1.0) -> dict | None:
    """The crack in mm from an estimated scale: widest, median, length and the widest's error (scale's share plus one
    pixel, the measurement's own error)."""
    if scale is None or width_max_px is None:
        return None
    k = scale["mm_per_px"]
    wmax = width_max_px * k
    return {"width_max_mm": round(wmax, 3), "width_median_mm": None if width_median_px is None else
            round(width_median_px * k, 3), "length_mm": None if length_px is None else round(length_px * k, 1),
            "uncertainty_mm": round(wmax * scale["rel"] + unc_px * k, 3), "rel": scale["rel"],
            "source": scale["source"], "basis": scale["basis"]}
