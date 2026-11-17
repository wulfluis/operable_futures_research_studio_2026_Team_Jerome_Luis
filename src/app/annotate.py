"""The crack's measurements drawn on the photo itself, as instruments rather than text (D80):
- a ruler that runs beside the crack and follows it, a tick every 10 mm (5 mm on a short crack), a number every 50 mm,
  the total at its end: the length read as on a tape;
- a loupe on the widest point, magnified from the full-resolution photo, with a calliper on the crack's two edges and
  reference lines of known width beside it at the same magnification (as an engineer's crack-width card);
- a protractor at the crack's start: the vertical, dashed, and the arc to the crack's direction;
- a map-style scale bar, black and white.
Only when the crack was measured with a bank card: without one nothing on the photo is in mm. A photo shows no depth,
so none is drawn. Spec: spec/features/app_measure.md, "On the photo"."""

from __future__ import annotations

import math

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

INK, PAPER, RED = (31, 35, 40), (255, 255, 255), (209, 36, 47)
SCALE_STEPS_MM = (5, 10, 20, 50, 100, 200)  # the scale bar takes the longest of these that fits a quarter of the photo
GAUGE_MM = (0.2, 0.5, 1.0, 2.0)  # reference lines in the loupe, as on a crack-width card (illustrative choice)
LOUPE_SHARE = 0.34  # loupe diameter, share of the photo's width
LOUPE_ZOOM_PX = 12  # the loupe shows about this many crack widths across (at least 40 px of the full photo)


def _font(px: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    f = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(f, max(9, int(px)))


def labels(crack) -> dict[str, str]:
    """The few numbers the instruments carry (kept short: the drawing does the explaining)."""
    return {"width": f"{crack.width_max_mm:.2f} mm", "length": f"{crack.length_main_mm:.0f} mm",
            "angle": f"{crack.angle_to_vertical_deg:.0f}°" if crack.angle_to_vertical_deg is not None else ""}


def scale_bar_mm(mm_per_px: float, width_px: int) -> int | None:
    """The longest round length (mm) whose bar fits a quarter of the photo's width; None if none fits."""
    fits = [s for s in SCALE_STEPS_MM if s / mm_per_px <= width_px / 4]
    return fits[-1] if fits else None


def ruler_steps(length_mm: float) -> tuple[float, float]:
    """(minor, major) tick spacing in mm: 10 and 50, or 5 and 25 for a crack under 100 mm, 50 and 250 over 1.5 m."""
    if length_mm < 100:
        return 5.0, 25.0
    if length_mm > 1500:
        return 50.0, 250.0
    return 10.0, 50.0


def _smooth(path: np.ndarray, k: int) -> np.ndarray:
    if len(path) < 3 or k < 2:
        return path
    pad = np.vstack([np.repeat(path[:1], k, 0), path, np.repeat(path[-1:], k, 0)])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(2)], 1)


def _outlined(d, pts, width, fill=PAPER, edge=INK):
    d.line(pts, fill=edge, width=width + 3, joint="curve")
    d.line(pts, fill=fill, width=width, joint="curve")


def _text(d, xy, s, font, anchor="mm"):
    d.text(xy, s, fill=INK, font=font, anchor=anchor, stroke_width=max(2, font.size // 5), stroke_fill=PAPER)


def draw_ruler(d, path: np.ndarray, length_mm: float, size, offset: float, lw: int, font) -> None:
    """A ruler beside the crack, following its line, ticks by real length along it."""
    p = _smooth(path, max(2, len(path) // 25))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 10:
        return
    tang = np.gradient(p, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    W, H = size
    centre = np.array([W / 2, H / 2])  # the ruler on the side facing the photo's middle: more room for it
    if np.mean(np.sum((centre - p) * normal, 1)) < 0:
        normal = -normal
    edge = p + normal * offset
    _outlined(d, [tuple(q) for q in edge], max(3, lw * 3))
    mm_per_px = length_mm / s[-1]
    minor, major = ruler_steps(length_mm)
    for mm in np.arange(0, length_mm + 1e-6, minor):
        i = int(np.searchsorted(s, mm / mm_per_px).clip(0, len(p) - 1))
        big = abs(mm / major - round(mm / major)) < 1e-6
        a = edge[i] - normal[i] * (lw * 1.5)
        b = edge[i] + normal[i] * (lw * (5 if big else 3))
        d.line([tuple(a), tuple(b)], fill=INK, width=max(1, lw if big else lw - 1))
        if big and mm > 0 and length_mm - mm > minor * 2:
            _text(d, tuple(edge[i] + normal[i] * (lw * 5 + font.size * 0.8)), f"{mm:g}", font)
    end = edge[-1] + normal[-1] * (lw * 5 + font.size)
    _text(d, tuple(end), f"{length_mm:.0f} mm", _font(font.size * 1.35))


def draw_protractor(d, start: np.ndarray, angle_deg: float, toward: np.ndarray, r: float, lw: int, font) -> None:
    """The vertical (dashed) and the arc from it to the crack's direction, at the crack's start."""
    up = np.array([0.0, -1.0]) if toward[1] < start[1] else np.array([0.0, 1.0])  # the vertical on the crack's side
    for k in range(0, int(r * 1.3), 8):
        d.line([tuple(start + up * k), tuple(start + up * min(k + 4, r * 1.3))], fill=INK, width=lw + 2)
        d.line([tuple(start + up * k), tuple(start + up * min(k + 4, r * 1.3))], fill=PAPER, width=lw)
    a0 = math.degrees(math.atan2(up[1], up[0]))
    v = toward - start
    a1 = math.degrees(math.atan2(v[1], v[0]))
    da = (a1 - a0 + 180) % 360 - 180
    lo, hi = (a0, a0 + da) if da >= 0 else (a0 + da, a0)
    box = [start[0] - r, start[1] - r, start[0] + r, start[1] + r]
    d.arc(box, lo, hi, fill=INK, width=lw + 2)
    d.arc(box, lo, hi, fill=PAPER, width=lw)
    mid = math.radians(a0 + da / 2)
    _text(d, (start[0] + (r + font.size) * math.cos(mid), start[1] + (r + font.size) * math.sin(mid)),
          f"{angle_deg:.0f}°", font)


def draw_loupe(img: Image.Image, full: Image.Image, scale: float, crack, avoid: np.ndarray, font) -> None:
    """A magnifier on the widest point, from the full-resolution photo: a calliper on the crack's edges and reference
    lines of known width beside it, at the same magnification."""
    W, H = img.size
    D = int(W * LOUPE_SHARE)
    cx, cy = (float(v) for v in crack.widest_point_px)  # full-photo pixels
    side = max(40.0, LOUPE_ZOOM_PX * (crack.px_across or 4))  # full px shown across the loupe
    zoom = D / side
    crop = full.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    lens = crop.resize((D, D), Image.LANCZOS).convert("RGB")
    ld = ImageDraw.Draw(lens)
    lw = max(2, D // 90)
    if crack.widest_edges_px is not None:  # the calliper: a jaw on each edge, joined across the crack
        e = (np.asarray(crack.widest_edges_px, float) - [cx - side / 2, cy - side / 2]) * zoom
        u = e[1] - e[0]
        u = u / max(np.linalg.norm(u), 1e-9)
        n = np.array([-u[1], u[0]])
        jaw = D * 0.16
        for q in e:
            _outlined(ld, [tuple(q - n * jaw), tuple(q + n * jaw)], lw)
        a, b = e[0] - u * D * 0.12, e[1] + u * D * 0.12
        _outlined(ld, [tuple(a), tuple(e[0])], lw)
        _outlined(ld, [tuple(e[1]), tuple(b)], lw)
    mm_per_lens_px = (crack.mm_per_px or 0) / zoom  # reference lines at the loupe's own scale
    if mm_per_lens_px > 0:  # a row of reference lines in the lower middle, inside the circle
        small = _font(font.size * 0.8)
        gauges = [(g, g / mm_per_lens_px) for g in GAUGE_MM if g / mm_per_lens_px <= D * 0.12]
        step = [max(w, small.size * 1.8) + D * 0.04 for _, w in gauges]
        x = D / 2 - (sum(step) - D * 0.04) / 2
        for (g, w), st in zip(gauges, step):
            ld.rectangle([x, D * 0.64, x + max(1, w), D * 0.80], fill=INK)
            _text(ld, (x + w / 2, D * 0.86), f"{g:g}", small)
            x += st
        if gauges:
            _text(ld, (D / 2, D * 0.94), "mm", small)
    mask = Image.new("L", (D, D), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, D - 1, D - 1], fill=255)
    # the loupe goes to the corner furthest from the crack, so it never hides it
    corners = [(8, 8), (W - D - 8, 8), (8, H - D - 8), (W - D - 8, H - D - 8)]
    pts = avoid if len(avoid) else np.array([[W / 2, H / 2]])
    ox, oy = max(corners, key=lambda c: np.min(np.linalg.norm(pts - [c[0] + D / 2, c[1] + D / 2], axis=1)))
    d = ImageDraw.Draw(img)
    wx, wy = cx * scale, cy * scale
    centre = np.array([ox + D / 2, oy + D / 2])
    v = np.array([wx, wy]) - centre
    rim = centre + v / max(np.linalg.norm(v), 1e-9) * D / 2
    _outlined(d, [(wx, wy), tuple(rim)], max(2, W // 300))
    img.paste(lens, (int(ox), int(oy)), mask)
    d.ellipse([ox - 2, oy - 2, ox + D + 1, oy + D + 1], outline=INK, width=4)
    d.ellipse([ox, oy, ox + D - 1, oy + D - 1], outline=PAPER, width=2)
    _text(d, (ox + D / 2, oy + D * 0.14), labels(crack)["width"], _font(font.size * 1.35))


def draw_scale_bar(d, mm_per_px: float, size, lw: int, font) -> None:
    """A map-style scale bar, alternating black and white, bottom left."""
    W, H = size
    s = scale_bar_mm(mm_per_px, W)
    if s is None:
        return
    n = s / mm_per_px
    x0, y0, h = W * 0.04, H - H * 0.045, max(6, lw * 3)
    parts = 5
    d.rectangle([x0 - 2, y0 - 2, x0 + n + 2, y0 + h + 2], fill=INK)
    for i in range(parts):
        d.rectangle([x0 + n * i / parts, y0, x0 + n * (i + 1) / parts, y0 + h], fill=INK if i % 2 else PAPER)
    _text(d, (x0, y0 - font.size * 0.8), "0", font)
    _text(d, (x0 + n, y0 - font.size * 0.8), f"{s} mm", font)


def draw_measurements(img: Image.Image, scale: float, crack, card_corners_px, full: Image.Image | None = None
                      ) -> Image.Image:
    """The photo part of the overlay (full-photo pixels times `scale`) with the instruments drawn on it.
    full: the full-resolution photo, for the loupe (none without it)."""
    out = img.copy()
    if crack is None or not crack.found or crack.width_max_mm is None:
        return out
    d = ImageDraw.Draw(out)
    W, H = out.size
    lw = max(2, W // 300)
    font = _font(W / 48)
    path = None
    if crack.main_path_px is not None and len(crack.main_path_px) >= 2:
        path = np.asarray(crack.main_path_px, float) * scale
        draw_ruler(d, path, crack.length_main_mm, out.size, offset=max(12, W / 45), lw=lw, font=font)
        if crack.angle_to_vertical_deg is not None:
            j = min(len(path) - 1, max(3, len(path) // 6))
            draw_protractor(d, path[0], crack.angle_to_vertical_deg, path[j], r=max(18, W / 18), lw=lw, font=font)
    if crack.mm_per_px:
        draw_scale_bar(d, crack.mm_per_px / scale, out.size, lw, font)
    if full is not None and crack.widest_point_px is not None:
        draw_loupe(out, full, scale, crack, path if path is not None else np.empty((0, 2)), font)
    return out
