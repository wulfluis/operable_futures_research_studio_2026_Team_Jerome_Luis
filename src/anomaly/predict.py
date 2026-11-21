"""The autoencoder on one photo (D89): the heatmap, the photo's score and the other defects, for the crack viewer.
Spec: spec/features/anomaly_ae.md. Shown only: nothing here changes a level."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from anomaly import model as ae
from anomaly import regions

HEAT_FROM = 0.5  # the heatmap starts at half the pixel threshold (below: transparent)
HEAT_FULL = 3.0  # and is at full strength at three times it
# heat colours: deep red to orange to pale yellow; never green, which reads as "safe" (rule 1)
STOPS = np.array([[120, 20, 60], [220, 38, 38], [251, 146, 60], [254, 240, 138]], dtype=float)


def heat_png(heat: np.ndarray, pixel_threshold: float) -> bytes:
    t = np.clip((heat / pixel_threshold - HEAT_FROM) / (HEAT_FULL - HEAT_FROM), 0, 1)
    pos = t * (len(STOPS) - 1)
    i = np.minimum(pos.astype(int), len(STOPS) - 2)
    f = (pos - i)[..., None]
    rgb = STOPS[i] * (1 - f) + STOPS[i + 1] * f
    alpha = np.where(t > 0, 70 + 170 * t, 0)
    im = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _small(mask: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 0


def analyse_photo(photo: Image.Image, crack_mask: np.ndarray | None = None,
                  exclude: np.ndarray | None = None) -> dict | None:
    """crack_mask and exclude: full-photo masks (the traced cracks, the bank card). None when the model is missing.
    Coordinates are returned as shares of the photo's width and height (0 to 1)."""
    loaded = ae.load()
    if loaded is None:
        return None
    m, thr, dev = loaded
    heat, tiles = ae.heatmap(m, photo, dev)
    h, w = heat.shape
    rgb = np.asarray(ae.work_image(photo))
    found = regions.find(rgb, ae.unusual(heat, thr), _small(crack_mask, (w, h)), _small(exclude, (w, h)))
    for f in found:
        f["centre"] = [round(f["centre"][0] / w, 5), round(f["centre"][1] / h, 5)]
        f["outline"] = [[round(x / w, 5), round(y / h, 5)] for x, y in f["outline"]]
    return {"score": round(float(tiles.mean()) / thr["tile_threshold"], 2), "defects": found,
            "heat_png": heat_png(heat, thr["pixel_floor"]), "version": thr.get("version")}
