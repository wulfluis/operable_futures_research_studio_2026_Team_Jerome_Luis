"""The photo read in steps (D122): the original, then the crack laid over it, then the crack alone on black, and the
autoencoder's map alone on black when it ran. Spec: spec/features/crack_view.md ("Steps"). Shown only."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

SIDE = 900  # px, long side of each step
CRACK = (244, 114, 182)  # the crack's colour in the steps (as the viewer's first crack); never green


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def crack_mask(result, size: tuple[int, int]) -> np.ndarray | None:
    """The crack model's decided area (the photo at working size, core.run_measured) at `size` (w, h)."""
    d = getattr(result, "decision", None)
    m = None if d is None else d.mask
    if m is None or not m.any():
        return None
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def steps(result) -> list[tuple[str, bytes]]:
    """[(caption, JPEG)] in reading order; [] without the photo or a crack."""
    photo = getattr(result, "photo", None)
    if photo is None:
        return []
    im = photo.convert("RGB").copy()
    im.thumbnail((SIDE, SIDE))
    out = [("1 · The photo", _jpeg(im))]
    m = crack_mask(result, im.size)
    if m is None:
        return out
    rgb = np.asarray(im).astype(float)
    over = rgb.copy()
    over[m] = 0.35 * over[m] + 0.65 * np.array(CRACK)
    out.append(("2 · The crack found, laid over the photo", _jpeg(Image.fromarray(over.astype(np.uint8)))))
    alone = np.zeros_like(rgb)
    alone[m] = CRACK
    out.append(("3 · The crack alone", _jpeg(Image.fromarray(alone.astype(np.uint8)))))
    heat = (getattr(result, "anomaly", None) or {}).get("heat_png")
    if heat:
        h = Image.open(io.BytesIO(heat)).convert("RGBA").resize(im.size)
        black = Image.new("RGBA", im.size, (0, 0, 0, 255))
        black.alpha_composite(h)
        out.append(("4 · What looks unlike sound concrete (autoencoder)", _jpeg(black)))
    return out


def _from_uri(uri: str | None) -> Image.Image | None:
    import base64
    if not uri or "," not in uri:
        return None
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGBA")


def steps_from_entry(zoom: bytes | None, marks: dict | None) -> list[tuple[str, bytes]]:
    """The same steps for a result on the engineer's wall, from what the wall keeps: the viewer's photo and its
    detected-area and heatmap layers (crack_view.view_data)."""
    if not zoom:
        return []
    im = Image.open(io.BytesIO(zoom)).convert("RGB")
    im.thumbnail((SIDE, SIDE))
    out = [("1 · The photo", _jpeg(im))]
    area = _from_uri((marks or {}).get("area"))
    if area is not None:
        area = area.resize(im.size)
        a = (np.asarray(area)[..., 3] > 40)
        rgb = np.asarray(im).astype(float)
        over = rgb.copy()
        over[a] = 0.35 * over[a] + 0.65 * np.array(CRACK)
        out.append(("2 · The crack found, laid over the photo", _jpeg(Image.fromarray(over.astype(np.uint8)))))
        alone = np.zeros_like(rgb)
        alone[a] = CRACK
        out.append(("3 · The crack alone", _jpeg(Image.fromarray(alone.astype(np.uint8)))))
    heat = _from_uri((marks or {}).get("heat"))
    if heat is not None:
        black = Image.new("RGBA", im.size, (0, 0, 0, 255))
        black.alpha_composite(heat.resize(im.size))
        out.append(("4 · Unlike sound concrete (autoencoder)", _jpeg(black)))
    return out
