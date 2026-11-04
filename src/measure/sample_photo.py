"""A realistic-looking sample photo with a known answer: concrete wall, a sample bank card (no real number),
an irregular crack with a varying width and a side branch, seen by a slightly tilted phone, saved as JPEG.

Usage, from the repository root:
    python src/measure/sample_photo.py            # writes data/samples/sample_crack_card.jpg and its truth JSON
It is still synthetic: it shows what the app does, it does not prove it works on real walls.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure.reference import PRESETS
from measure.synthetic import camera_homography

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "samples" / "sample_crack_card.jpg"
CARD_MM = PRESETS["card"][1]
PLANE_MM = (320.0, 255.0)  # wall area rendered, larger than the view so the tilted camera sees no edge
RES = 16  # plane pixels per mm
CARD_CENTRE_MM, CARD_ANGLE_DEG = (100.0, 137.0), 4.0  # card stuck on the wall, slightly turned
CORNER_R_MM = 3.18  # ISO/IEC 7810 ID-1 corner radius


def _fractal_line(a, b, rng, depth=7, rough=0.18):
    pts = [np.array(a, float), np.array(b, float)]
    for d in range(depth):
        new = [pts[0]]
        for p, q in zip(pts[:-1], pts[1:]):
            m = (p + q) / 2
            n = np.array([-(q - p)[1], (q - p)[0]])
            new += [m + n * rng.normal(0, rough * 0.6 ** d), q]
        pts = new
    return np.array(pts)


def _draw_crack(plane_dark, line, widths_mm):
    """Distance field per segment, only inside each segment's box: crack depth 1 at the core, soft edges."""
    H, W = plane_dark.shape
    for (p, q), (wa, wb) in zip(zip(line[:-1] * RES, line[1:] * RES), zip(widths_mm[:-1], widths_mm[1:])):
        r = max(wa, wb) * RES / 2 + 3
        x0, x1 = int(max(0, min(p[0], q[0]) - r)), int(min(W, max(p[0], q[0]) + r + 1))
        y0, y1 = int(max(0, min(p[1], q[1]) - r)), int(min(H, max(p[1], q[1]) + r + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
        v = q - p
        t = np.clip(((xs - p[0]) * v[0] + (ys - p[1]) * v[1]) / max(v @ v, 1e-9), 0, 1)
        d = np.hypot(xs - p[0] - t * v[0], ys - p[1] - t * v[1])
        half = (wa + (wb - wa) * t) * RES / 2
        depth = np.clip(half + 0.5 - d, 0, 1)  # anti-aliased edge, one plane pixel wide
        plane_dark[y0:y1, x0:x1] = np.maximum(plane_dark[y0:y1, x0:x1], depth)


def _card_texture():
    w, h = round(CARD_MM[0] * RES), round(CARD_MM[1] * RES)
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    base = np.stack([20 + 40 * xx, 70 + 60 * yy, 140 + 70 * xx], axis=-1)  # blue to teal
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    c = lambda mm: round(mm * RES)  # noqa: E731
    draw.rounded_rectangle([c(9), c(18), c(21), c(27)], radius=c(1.5), fill=(212, 175, 95), outline=(150, 120, 60))
    try:
        big, small = ImageFont.load_default(size=c(5)), ImageFont.load_default(size=c(3.2))
    except TypeError:  # old Pillow
        big = small = ImageFont.load_default()
    draw.text((c(8), c(6)), "SAMPLE BANK", fill=(240, 240, 245), font=big)
    draw.text((c(8), c(35)), "NO REAL CARD NUMBER", fill=(225, 230, 240), font=small)
    draw.text((c(8), c(42)), "SECOND LOOK · TEST CARD", fill=(225, 230, 240), font=small)
    alpha = Image.new("L", img.size, 0)
    ImageDraw.Draw(alpha).rounded_rectangle([0, 0, w - 1, h - 1], radius=c(CORNER_R_MM), fill=255)
    return img, alpha


def make_sample(seed: int = 7, size=(3000, 2250), px_per_mm: float = 11.0, tilt_lr=8.0, tilt_ud=-5.0, roll=2.5):
    rng = np.random.default_rng(seed)
    W, H = round(PLANE_MM[0] * RES), round(PLANE_MM[1] * RES)
    # wall: beige-grey concrete with grain, pores and a faint stain
    tone = np.array([168.0, 160.0, 148.0])
    grain = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 1.5) * 14
    blotch = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 60) * 260
    pores = ndimage.gaussian_filter((rng.random((H, W)) > 0.9993).astype(float), 2.0) * -900
    lum = grain + blotch + np.clip(pores, -70, 0)
    plane = (tone[None, None, :] + lum[..., None] * np.array([1.0, 0.97, 0.93])).astype(np.float32)
    # crack: main path with a branch, width from 0.35 to 1.4 mm and back
    main = _fractal_line((210, 43), (230, 213), rng)
    s = np.linspace(0, 1, len(main))
    w_main = 0.35 + 1.05 * np.exp(-((s - 0.45) / 0.22) ** 2)
    k = len(main) // 2
    branch = _fractal_line(main[k], main[k] + (38, 30), rng, depth=6)
    w_branch = np.linspace(0.6, 0.2, len(branch))
    dark = np.zeros((H, W))
    _draw_crack(dark, main, w_main)
    _draw_crack(dark, branch, w_branch)
    lip = ndimage.gaussian_filter(dark, 4) * 0.25  # slightly darker crack lips
    inside = np.array([48.0, 44.0, 40.0])
    plane = plane * (1 - np.clip(lip, 0, 0.3))[..., None]
    plane = plane * (1 - dark[..., None]) + inside * dark[..., None]
    # card with a soft shadow
    card, alpha = _card_texture()
    card_r = card.rotate(-CARD_ANGLE_DEG, resample=Image.BICUBIC, expand=True)
    alpha_r = alpha.rotate(-CARD_ANGLE_DEG, resample=Image.BICUBIC, expand=True)
    ox = round(CARD_CENTRE_MM[0] * RES - card_r.width / 2)
    oy = round(CARD_CENTRE_MM[1] * RES - card_r.height / 2)
    a = np.zeros((H, W))
    a[oy:oy + card_r.height, ox:ox + card_r.width] = np.asarray(alpha_r) / 255.0
    shadow = ndimage.shift(ndimage.gaussian_filter(a, 8), (10, 6), order=1) * 0.3
    plane = plane * (1 - shadow)[..., None]
    rgb_card = np.zeros((H, W, 3))
    rgb_card[oy:oy + card_r.height, ox:ox + card_r.width] = np.asarray(card_r, dtype=float)
    plane = plane * (1 - a[..., None]) + rgb_card * a[..., None]

    # photograph it
    Hm, tilt = camera_homography(size, tilt_lr, tilt_ud, px_per_mm, centre_mm=(PLANE_MM[0] / 2, PLANE_MM[1] / 2),
                                 roll_deg=roll)
    Hinv = np.linalg.inv(Hm)
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    q = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ Hinv.T.astype(np.float32)
    mx, my = q[..., 0] / q[..., 2] * RES, q[..., 1] / q[..., 2] * RES
    img = np.stack([ndimage.map_coordinates(plane[..., c], [my, mx], order=1, mode="nearest") for c in range(3)],
                   axis=-1)
    light = 1.06 - 0.16 * ((xs / w - 0.35) ** 2 + (ys / h - 0.3) ** 2)  # light from the top left, vignetting
    img = ndimage.gaussian_filter(img * light[..., None], (0.8, 0.8, 0))
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    def to_px(p_mm):
        p = np.c_[np.asarray(p_mm, float), np.ones(len(p_mm))] @ Hm.T
        return p[:, :2] / p[:, 2:3]

    c, sn = math.cos(math.radians(CARD_ANGLE_DEG)), math.sin(math.radians(CARD_ANGLE_DEG))
    hw, hh = CARD_MM[0] / 2, CARD_MM[1] / 2
    corners = [(CARD_CENTRE_MM[0] + x * c - y * sn, CARD_CENTRE_MM[1] + x * sn + y * c)
               for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    main_len = float(np.linalg.norm(np.diff(main, axis=0), axis=1).sum())
    truth = {"width_max_mm": round(float(w_main.max()), 3), "length_main_mm": round(main_len, 1),
             "length_branch_mm": round(float(np.linalg.norm(np.diff(branch, axis=0), axis=1).sum()), 1),
             "tilt_deg": round(tilt, 1), "card_corners_px": to_px(corners).round(2).tolist(),
             "note": "Synthetic sample photo; the widest point of the main crack is 1.4 mm, the branch is thinner."}
    return img, truth


def ensure_sample(path: Path = OUT) -> tuple[Path, dict]:
    """Create the sample photo once; later calls reuse it."""
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        img, truth = make_sample()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path, "JPEG", quality=90)
        meta.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return path, json.loads(meta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    p, t = ensure_sample()
    print(f"written {p}\ntruth: width {t['width_max_mm']} mm, length {t['length_main_mm']} mm, tilt {t['tilt_deg']}°")
