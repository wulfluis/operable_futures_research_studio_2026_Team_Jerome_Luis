"""Other defects, separate from the cracks (D89): written rules on the autoencoder's unusual pixels.
Spec: spec/features/anomaly_ae.md. Every value is illustrative until calibrated; every label says "possible"."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

MIN_AREA_SHARE = 0.0015  # of the photo: smaller unusual areas are specks
CRACK_MARGIN_SHARE = 0.008  # of the long side: unusual pixels this close to a traced crack belong to the crack
RING_SHARE = 0.016  # of the long side: the surface around an area, its reference colour
RUST_HUE_DEG = (5.0, 45.0)  # orange to brown
RUST_SAT_MORE = 0.12  # saturation above the surface around it (0 to 1)
DARK_MORE = 0.12  # brightness below the surface around it (0 to 1)
LABELS = {"rust": "Possible rust stain", "dark": "Dark patch (damp or dirt?)", "other": "Other unusual area"}


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    return hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0  # hue in degrees, saturation, value


def label_area(rgb: np.ndarray, area: np.ndarray, around: np.ndarray) -> str:
    """rust, dark or other, from the area's colour compared with the surface around it."""
    hue, sat, val = _hsv(rgb)
    s_in, s_out = float(np.median(sat[area])), float(np.median(sat[around]))
    v_in, v_out = float(np.median(val[area])), float(np.median(val[around]))
    coloured = area & (sat > 0.1)
    h_in = float(np.median(hue[coloured])) if coloured.sum() >= max(5, 0.2 * area.sum()) else None
    if h_in is not None and RUST_HUE_DEG[0] <= h_in <= RUST_HUE_DEG[1] and s_in - s_out >= RUST_SAT_MORE:
        return "rust"
    if v_out - v_in >= DARK_MORE:
        return "dark"
    return "other"


def find(rgb: np.ndarray, unusual: np.ndarray, crack: np.ndarray | None = None,
         exclude: np.ndarray | None = None) -> list[dict]:
    """The other defects in the photo, largest first: each with its label, its outline and centre in the photo's
    pixels (x, y), and its share of the photo. Unusual pixels near a traced crack, or on the card, are left out."""
    h, w = unusual.shape
    side = max(h, w)
    m = unusual.copy()
    if crack is not None and crack.any():
        m &= ~ndimage.binary_dilation(crack, iterations=max(1, round(CRACK_MARGIN_SHARE * side)))
    if exclude is not None:
        m &= ~exclude
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    labels, n = ndimage.label(m)
    if n == 0:
        return []
    sizes = ndimage.sum(m, labels, range(1, n + 1))
    blocked = unusual if crack is None else unusual | crack
    ring_px = max(2, round(RING_SHARE * side))
    out = []
    for i in np.argsort(sizes)[::-1]:
        if sizes[i] < MIN_AREA_SHARE * h * w:
            break
        area = labels == i + 1
        around = ndimage.binary_dilation(area, iterations=ring_px) & ~blocked
        if exclude is not None:
            around &= ~exclude
        if around.sum() < 20:
            around = ~blocked
        kind = label_area(rgb, area, around)
        cs, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        ys, xs = np.nonzero(area)
        out.append({"kind": kind, "label": LABELS[kind], "share": round(float(sizes[i]) / (h * w), 5),
                    "centre": [float(xs.mean()), float(ys.mean())], "outline": c.astype(float).tolist()})
    return out
