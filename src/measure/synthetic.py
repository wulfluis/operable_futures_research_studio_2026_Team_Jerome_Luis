"""Synthetic test photo with a known answer: a wall, a bank card and a crack of known width and length, seen by a
pinhole camera that can be tilted. Used by the tests, `measure_crack.py --demo` and the app's demo button.
Not a real photo: it checks the geometry and the code, not how well cracks are found on real walls.
"""

import math

import numpy as np
from scipy import ndimage

from measure.card import FOCAL_PRIOR
from measure.reference import PRESETS

CARD_MM = PRESETS["card"][1]
CARD_ORIGIN_MM = (15.0, 60.0)
CRACK_MM = np.array([(125.0, 15.0), (140.0, 55.0), (133.0, 95.0), (160.0, 140.0)])  # centre line in the wall plane
PLANE_CENTRE_MM = (100.0, 75.0)


def _dist_to_polyline(x, y, line):
    d = np.full(x.shape, np.inf)
    for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):
        vx, vy = bx - ax, by - ay
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.minimum(d, np.hypot(x - ax - t * vx, y - ay - t * vy))
    return d


def camera_homography(size, tilt_lr_deg=0.0, tilt_ud_deg=0.0, px_per_mm=6.0, centre_mm=PLANE_CENTRE_MM,
                      roll_deg=0.0):
    """Wall plane (mm) -> image (px) for a camera looking at centre_mm, turned by the angles (roll: about its axis)."""
    w, h = size
    f = FOCAL_PRIOR * max(w, h)
    a, b, r = math.radians(tilt_lr_deg), math.radians(tilt_ud_deg), math.radians(roll_deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    z = f / px_per_mm
    cx, cy = centre_mm
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    # plane point (X, Y) -> camera: R @ (X - cx, Y - cy, 0) + (0, 0, z)
    M = np.column_stack([R[:, 0], R[:, 1], np.array([0, 0, z]) - R[:, 0] * cx - R[:, 1] * cy])
    H = K @ M
    tilt = math.degrees(math.acos(abs(R[2, 2])))
    return H / H[2, 2], tilt


def make_demo(width_mm: float = 1.0, size=(1200, 900), seed: int = 0, sub: int = 2, tilt_lr_deg: float = 12.0,
              tilt_ud_deg: float = -6.0, px_per_mm: float = 6.0, texture_amp: float = 1.0):
    """Return (grey image as float 0-255, truth). About px_per_mm pixels per mm at the centre of the wall."""
    w, h = size
    H_mm_to_px, tilt = camera_homography(size, tilt_lr_deg, tilt_ud_deg, px_per_mm)
    H_px_to_mm = np.linalg.inv(H_mm_to_px)
    acc = np.zeros((h, w))
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    for oy in range(sub):
        for ox in range(sub):
            px = xs + (ox + 0.5) / sub - 0.5
            py = ys + (oy + 0.5) / sub - 0.5
            q = np.stack([px, py, np.ones_like(px)], axis=-1) @ H_px_to_mm.T
            mx, my = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
            val = np.full((h, w), 150.0)
            card = ((mx >= CARD_ORIGIN_MM[0]) & (mx <= CARD_ORIGIN_MM[0] + CARD_MM[0]) &
                    (my >= CARD_ORIGIN_MM[1]) & (my <= CARD_ORIGIN_MM[1] + CARD_MM[1]))
            val[card] = 225.0
            val[_dist_to_polyline(mx, my, CRACK_MM) <= width_mm / 2] = 55.0
            acc += val
    rng = np.random.default_rng(seed)
    wall = acc / sub ** 2
    # concrete-like texture on the wall only (pores and grain), so the photo is not unnaturally smooth (D12)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 1.2) * 9 * texture_amp
    texture += ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 12) * 60 * texture_amp
    on_wall = np.abs(wall - 150.0) < 1
    grey = ndimage.gaussian_filter(wall + texture * on_wall, 0.6)
    grey += rng.normal(0, 4, grey.shape)
    grey = np.clip(grey, 0, 255)

    def to_px(p):
        q = np.c_[np.asarray(p, dtype=float), np.ones(len(p))] @ H_mm_to_px.T
        return q[:, :2] / q[:, 2:3]

    cx, cy = CARD_ORIGIN_MM
    card_px = to_px([(cx, cy), (cx + CARD_MM[0], cy), (cx + CARD_MM[0], cy + CARD_MM[1]), (cx, cy + CARD_MM[1])])
    crack_px = to_px(CRACK_MM)
    roi = (*(crack_px.min(axis=0) - 25), *(crack_px.max(axis=0) + 25))
    truth = {"width_mm": width_mm, "length_mm": float(np.linalg.norm(np.diff(CRACK_MM, axis=0), axis=1).sum()),
             "card_corners_px": card_px.tolist(), "roi_px": [float(v) for v in roi], "tilt_deg": tilt}
    return grey, truth
