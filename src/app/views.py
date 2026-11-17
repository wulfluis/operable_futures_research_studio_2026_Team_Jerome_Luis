"""Three more ways to look at photo 1 (D81), shown with the result:
- width heatmap: the crack coloured by its measured width, point by point, on a dimmed photo, with the risk matrix's
  width bands marked on the colour bar (bank card only: the width is in mm);
- confidence heatmap: where the crack model sees crack, and how sure it is, from 0 to 100 %;
- enhanced view: the photo with its local contrast boosted and inverted, so hairline continuations stand out.
  It is not an X-ray: a photo cannot see inside the concrete, the bars or a stress field, and the view says so.
None of them changes the level. Spec: spec/features/app_measure.md, "Views"."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps, font_manager
from PIL import Image, ImageDraw, ImageFont

XRAY_NOTE = "Enhanced view: contrast boosted and inverted. Not an X-ray: it cannot see inside the concrete."
WIDTH_MAP, CONF_MAP = "plasma", "magma"


def _font(px):
    f = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(f, max(10, int(px)))


def _dim(photo: Image.Image, k: float = 0.45) -> np.ndarray:
    g = np.asarray(photo.convert("L"), float)
    return np.repeat((255 - (255 - g) * k)[..., None] * 0.9, 3, axis=2)  # light and grey, so colour reads on it


def _bar(img: Image.Image, cmap: str, lo: float, hi: float, unit: str, ticks=(), title: str = "") -> Image.Image:
    """A colour bar under the picture, with its end values, optional marked values, and a title."""
    W = img.width
    f = _font(W / 46)
    h = int(f.size * 4.2)
    out = Image.new("RGB", (W, img.height + h), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    x0, x1, y0 = int(W * 0.06), int(W * 0.94), img.height + int(f.size * 1.6)
    ramp = (colormaps[cmap](np.linspace(0, 1, x1 - x0))[:, :3] * 255).astype(np.uint8)
    out.paste(Image.fromarray(np.repeat(ramp[None], int(f.size * 0.9), axis=0)), (x0, y0))
    d.text((x0, img.height + 4), title, fill=(31, 35, 40), font=f)
    by = y0 + int(f.size * 0.9) + 2
    d.text((x0, by), f"{lo:g}", fill=(31, 35, 40), font=f)
    d.text((x1, by), f"{hi:g} {unit}", fill=(31, 35, 40), font=f, anchor="ra")
    for v, name in ticks:
        if lo < v < hi:
            x = x0 + (v - lo) / (hi - lo) * (x1 - x0)
            d.line([(x, y0 - 3), (x, y0 + f.size)], fill=(31, 35, 40), width=2)
            d.text((x, by), name, fill=(31, 35, 40), font=f, anchor="ma")
    return out


def width_heatmap(work: Image.Image, scale: float, crack) -> Image.Image | None:
    """The crack's pixels coloured by the width measured nearest to them (mm), on the dimmed photo."""
    if crack is None or crack.profile_px is None or crack.mask is None or len(crack.profile_px) < 3:
        return None
    from scipy.spatial import cKDTree

    from triage.matrix import _DATA as MATRIX  # the width bands, for the marks on the bar
    W, H = work.size
    m = np.asarray(Image.fromarray(crack.mask.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 40
    from scipy import ndimage
    m = ndimage.binary_dilation(m, iterations=max(1, W // 300))  # drawn thicker than life, so its colour shows
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    _, k = cKDTree(np.asarray(crack.profile_px, float) * scale).query(np.column_stack([xs, ys]))
    w = np.asarray(crack.profile_mm, float)[k]
    lo, hi = 0.0, max(0.2, float(np.ceil(np.max(crack.profile_mm) * 12)) / 10)
    rgb = _dim(work)
    rgb[ys, xs] = colormaps[WIDTH_MAP](np.clip((w - lo) / (hi - lo), 0, 1))[:, :3] * 255
    ticks = [(b["upper_mm"], f"{b['upper_mm']:g}") for b in MATRIX["width_bands"] if b.get("upper_mm")]
    return _bar(Image.fromarray(rgb.astype(np.uint8)), WIDTH_MAP, lo, hi, "mm", ticks,
                "WIDTH ALONG THE CRACK (bank card; line drawn thicker) · marks: matrix width bands")


def confidence_heatmap(work: Image.Image, prob: np.ndarray) -> Image.Image:
    """The crack model's probability, 0 to 100 %, over the dimmed photo (strong where it is sure)."""
    W, H = work.size
    p = np.clip(np.asarray(prob, float)[:H, :W], 0, 1)
    col = colormaps[CONF_MAP](p)[..., :3] * 255
    a = np.clip(p * 1.4, 0, 1)[..., None]
    rgb = _dim(work) * (1 - a) + col * a
    return _bar(Image.fromarray(rgb.astype(np.uint8)), CONF_MAP, 0, 100, "%", (),
                "WHERE THE MODEL SEES CRACK · how sure it is")


def enhanced_view(work: Image.Image) -> Image.Image:
    """Local contrast boosted (CLAHE) and inverted, in a cool tone, with the note that it is not an X-ray."""
    from skimage import exposure
    g = np.asarray(work.convert("L"), float) / 255
    e = 1 - exposure.equalize_adapthist(g, clip_limit=0.03)
    rgb = np.stack([e * 0.80, e * 0.92, e], -1) * 255
    img = Image.fromarray(rgb.astype(np.uint8))
    f = _font(img.width / 46)
    out = Image.new("RGB", (img.width, img.height + int(f.size * 4.2)), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    lines, cur = [], ""
    for word in XRAY_NOTE.split():  # wrapped to the picture's width
        if cur and d.textlength(cur + " " + word, font=f) > img.width * 0.92:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    lines.append(cur)
    d.multiline_text((int(img.width * 0.04), img.height + int(f.size * 0.5)), "\n".join(lines), fill=(31, 35, 40),
                     font=f, spacing=4)
    return out


def make(work: Image.Image, prob: np.ndarray | None, scale: float, crack, measured: bool) -> dict[str, Image.Image]:
    """The views for one photo 1, by name (shown in this order)."""
    out = {}
    if measured:
        w = width_heatmap(work, scale, crack)
        if w is not None:
            out["Width heatmap"] = w
    if prob is not None:
        out["Confidence heatmap"] = confidence_heatmap(work, prob)
    out["Enhanced view (not an X-ray)"] = enhanced_view(work)
    return out
