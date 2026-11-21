"""Showcase cases for the class demo (D90): drawn photos with a bank card that show what the crack viewer can do:
several independent cracks each in its own colour, a crack with a gap that stays one crack, a network of cracks, side
branches, a hairline next to a wide crack (D88), and the autoencoder's heatmap on each (D89). Every photo is drawn here:
no real home and no resident's photo (rule 9). Spec: spec/features/class_demo.md (Simulated residents).

Rust stains and damp patches are not drawn: the autoencoder does not see soft stains, and a drawn stain would only show
what the drawing was made to show (docs/experiments/2026-11-21_anomaly_ae.md)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from measure.synthetic import CARD_MM, CARD_ORIGIN_MM, _dist_to_polyline, camera_homography

PLASTER = np.array([186.0, 181.0, 172.0])  # a light, warm painted plaster
CARD = np.array([226.0, 226.0, 224.0])
CRACK = np.array([52.0, 48.0, 45.0])


def wander(points, seed: int, amp_mm: float = 1.6, step_mm: float = 4.0) -> list:
    """A crack never runs straight (a straight line is read as an edge, D51): the line between the given points with
    small sideways steps, as real cracks wander."""
    rng = np.random.default_rng(seed)
    p = np.asarray(points, float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step_mm))
        d = (b - a) / np.linalg.norm(b - a)
        side = np.array([-d[1], d[0]])
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t + side * (rng.normal(0, amp_mm) if t < 1 else 0.0))
    return [tuple(q) for q in out]


# (name, element of photo 2, PHI-Net task 6 test index for photo 2, tilt left-right, up-down, cracks)
# cracks: (centre line in the wall plane in mm, y down; width in mm). The view covers about 0 to 200 mm across and
# 0 to 150 mm down; the card sits at x 15 to 101, y 60 to 114 (measure/synthetic.py).
CASES = [
    ("two independent cracks", "wall", 400, 6, -3,
     [([(115, 10), (140, 48), (170, 92), (186, 118)], 0.9), ([(30, 8), (34, 30), (38, 55)], 0.4)]),
    ("three cracks, three widths", "column", 98, 5, 4,
     [([(122, 10), (124, 75), (128, 140)], 0.6), ([(160, 20), (181, 52)], 1.3), ([(40, 125), (100, 132)], 0.35)]),
    ("one crack with a gap", "column", 84, 6, 2,
     [([(130, 10), (136, 40), (134, 62)], 0.7), ([(135, 67), (140, 100), (137, 140)], 0.7)]),
    ("a network of cracks (map cracking)", "wall", 459, 3, -2,
     [([(110, 15), (150, 18), (185, 14)], 0.5), ([(110, 15), (112, 55), (108, 95)], 0.5),
      ([(150, 18), (147, 58), (152, 96)], 0.5), ([(185, 14), (188, 56), (183, 98)], 0.5),
      ([(112, 55), (147, 58), (188, 56)], 0.5), ([(108, 95), (152, 96), (183, 98)], 0.5)]),
    ("a crack with side branches", "beam", 35, 7, -4,
     [([(110, 20), (135, 45), (150, 80), (175, 120)], 1.0), ([(135, 45), (160, 35), (185, 38)], 0.5),
      ([(150, 80), (125, 100), (112, 125)], 0.4)]),
    ("a hairline next to a wide crack", "beam", 42, 5, -5,
     [([(120, 10), (125, 70), (121, 140)], 1.8), ([(160, 15), (166, 70), (170, 135)], 0.35)]),
]


def draw(cracks, size=(1200, 900), seed: int = 0, tilt_lr_deg: float = 5.0, tilt_ud_deg: float = -3.0,
         px_per_mm: float = 6.0, sub: int = 2) -> np.ndarray:
    """A colour photo (H x W x 3, uint8) of plaster with a bank card and the cracks, seen by a slightly tilted phone,
    as measure/synthetic.py draws its grey crack."""
    w, h = size
    to_px, _ = camera_homography(size, tilt_lr_deg, tilt_ud_deg, px_per_mm)
    to_mm = np.linalg.inv(to_px)
    rng = np.random.default_rng(seed)
    lines = [(np.asarray(wander(line, seed + i), float), width) for i, (line, width) in enumerate(cracks)]
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    acc = np.zeros((h, w, 3))
    for oy in range(sub):
        for ox in range(sub):
            q = np.stack([xs + (ox + 0.5) / sub - 0.5, ys + (oy + 0.5) / sub - 0.5, np.ones_like(xs)], -1) @ to_mm.T
            mx, my = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
            img = np.broadcast_to(PLASTER, (h, w, 3)).copy()
            card = ((mx >= CARD_ORIGIN_MM[0]) & (mx <= CARD_ORIGIN_MM[0] + CARD_MM[0]) &
                    (my >= CARD_ORIGIN_MM[1]) & (my <= CARD_ORIGIN_MM[1] + CARD_MM[1]))
            img[card] = CARD
            for line, width in lines:
                img[_dist_to_polyline(mx, my, line) <= width / 2] = CRACK
            acc += img
    img = acc / sub ** 2
    on_wall = np.abs(img - PLASTER).sum(-1) < 30  # texture on the plaster only, not on the card or the cracks
    grain = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 1.2) * 6
    cloud = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 14) * 40
    img = img + ((grain + cloud) * on_wall)[..., None]
    img = ndimage.gaussian_filter(img, (0.6, 0.6, 0)) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def cases():
    """(photo 1, element, PHI-Net index for photo 2, name) for every showcase case."""
    for k, (name, element, index, lr, ud, cracks) in enumerate(CASES):
        yield draw(cracks, seed=100 + k, tilt_lr_deg=lr, tilt_ud_deg=ud), element, index, name
