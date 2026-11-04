"""Bank card as the reference: find it in the photo and check whether the photo was taken straight.

Spec: spec/features/measurement.md, decision D42.
1. Find four-sided shapes (OpenCV contours on edges and on brightness and colour), on a small copy of the photo.
2. Keep the shapes that a card (ISO/IEC 7810 ID-1, 85.60 x 53.98 mm) could make, seen at most MAX_TILT_DEG off
   straight: the camera pose is solved (planar PnP) and the card's corners must be reprojected within
   MAX_REPROJECTION of the card size. The largest such shape is taken.
3. Place each side precisely on the full photo: the strongest edge across the side, sampled over its middle 70 %
   (the card's corners are rounded), one line fitted per side; the corners are the crossings of the lines.
4. Tilt = angle between the card's normal and the camera axis, from the pose.
Assumptions: the photo is not cropped (the camera centre is the image centre) and the focal length is about
FOCAL_PRIOR x the long side (phone main camera, about 26 mm equivalent). Zoom or a crop makes the tilt less exact.
All thresholds are illustrative until tested on the pilot photos.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from measure.reference import PRESETS

CARD_MM = PRESETS["card"][1]  # (85.60, 53.98)
DETECT_LONG_SIDE = 1000  # px; the search for the card runs on a copy this size
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.002, 0.8
MAX_TILT_DEG = 35.0  # beyond this, a shape is not accepted as a card
MAX_REPROJECTION = 0.02  # corner error / card diagonal, for a shape to count as a card
TILT_OK_DEG = 10.0  # illustrative
TILT_RETAKE_DEG = 20.0  # illustrative
FOCAL_PRIOR = 0.75  # x long side of the photo; assumption


@dataclass(frozen=True)
class CardDetection:
    corners_px: np.ndarray  # (4, 2), in order around the card
    long_first: bool  # side corner 0 -> 1 is the long side
    tilt_deg: float
    tilt_left_right_deg: float  # turn about the vertical axis
    tilt_up_down_deg: float  # turn about the horizontal axis
    reprojection: float
    status: str  # straight, tilted or retake
    message: str

    def to_dict(self) -> dict:
        return {"corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners_px],
                "tilt_deg": round(self.tilt_deg, 1), "tilt_left_right_deg": round(self.tilt_left_right_deg, 1),
                "tilt_up_down_deg": round(self.tilt_up_down_deg, 1), "status": self.status,
                "message": self.message, "fit_error": round(self.reprojection, 4)}


def camera_matrix(shape) -> np.ndarray:
    h, w = shape[:2]
    f = FOCAL_PRIOR * max(h, w)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def order_corners(q) -> np.ndarray:
    """Clockwise on screen, starting with the corner nearest the top left."""
    q = np.asarray(q, dtype=float).reshape(4, 2)
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def card_pose(corners_px, shape):
    """Camera pose for a card with these corners: (reprojection error / diagonal, tilt, left-right, up-down,
    long_first). Both ways of laying the card on the corners are tried; the better fit wins."""
    K = camera_matrix(shape)
    q = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    diag = float(np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2
    L, S = max(CARD_MM), min(CARD_MM)
    best = None
    for long_first in (True, False):
        w, h = (L, S) if long_first else (S, L)
        obj = np.array([(0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0)], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(4, 2)
        err = float(np.linalg.norm(proj - q, axis=1).max()) / max(diag, 1e-9)
        n = cv2.Rodrigues(rvec)[0][:, 2]
        n = n if n[2] > 0 else -n
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        lr = math.degrees(math.atan2(float(n[0]), float(n[2])))
        ud = math.degrees(math.atan2(float(n[1]), float(n[2])))
        if best is None or err < best[0]:
            best = (err, tilt, lr, ud, long_first)
    return best


def _quads(rgb_small) -> list[np.ndarray]:
    grey = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)[..., 1]
    maps = []
    for ch in (grey, sat):
        b = cv2.GaussianBlur(ch, (5, 5), 0)
        for lo, hi in ((20, 60), (40, 120), (80, 200)):
            maps.append(cv2.dilate(cv2.Canny(b, lo, hi), np.ones((3, 3), np.uint8)))
        _, t = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        maps += [t, 255 - t]
    area_img = grey.shape[0] * grey.shape[1]
    quads = []
    for m in maps:
        for c in cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if not MIN_AREA_FRAC * area_img <= cv2.contourArea(c) <= MAX_AREA_FRAC * area_img:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.015, 0.03, 0.05):
                a = cv2.approxPolyDP(hull, eps * peri, True)
                if len(a) == 4 and cv2.isContourConvex(a):
                    if cv2.contourArea(a) >= 0.85 * cv2.contourArea(hull):
                        quads.append(order_corners(a))
                    break
    return quads


def refine_corners(grey: np.ndarray, quad: np.ndarray, search_px: float) -> np.ndarray:
    """Sub-pixel corners: the strongest edge across each side, a line per side, corners where lines cross."""
    g = ndimage.gaussian_filter(grey.astype(float), 1.0)
    lines = []
    offs = np.arange(-search_px, search_px + 1e-9, 0.5)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        base = a + np.linspace(0.15, 0.85, 60)[:, None] * (b - a)
        P = base[:, None, :] + offs[None, :, None] * n
        prof = ndimage.map_coordinates(g, [P[..., 1], P[..., 0]], order=1, mode="nearest")
        grad = np.abs(np.gradient(prof, axis=1))
        k = np.clip(np.argmax(grad, axis=1), 1, len(offs) - 2)
        r = np.arange(len(k))
        y0, y1, y2 = grad[r, k - 1], grad[r, k], grad[r, k + 1]
        den = y0 - 2 * y1 + y2
        sub = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        pts = base + ((offs[k] + np.clip(sub, -0.5, 0.5) * 0.5)[:, None]) * n
        strong = y1 >= 0.3 * np.median(y1)  # drop samples with no clear edge (a finger, a shadow)
        vx, vy, x0, y0_ = cv2.fitLine(pts[strong].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0_]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    return out if np.linalg.norm(out - quad, axis=1).max() <= 2 * search_px + 2 else quad


def _status(tilt):
    if tilt <= TILT_OK_DEG:
        return "straight", f"Photo taken straight enough (tilt {tilt:.0f} degrees)."
    if tilt <= TILT_RETAKE_DEG:
        return "tilted", (f"Photo tilted by {tilt:.0f} degrees: corrected using the card, but hold the phone "
                          "parallel to the wall next time.")
    return "retake", f"Photo tilted by {tilt:.0f} degrees: retake it with the phone parallel to the wall."


def from_corners(corners_px, shape) -> CardDetection | None:
    """Pose and tilt for card corners given by hand (or after detection)."""
    q = order_corners(corners_px)
    pose = card_pose(q, shape)
    if pose is None:
        return None
    err, tilt, lr, ud, long_first = pose
    status, msg = _status(tilt)
    return CardDetection(q, long_first, tilt, lr, ud, err, status, msg)


def detect_card(rgb: np.ndarray) -> CardDetection | None:
    """Find the bank card in an RGB photo; None if no card-shaped rectangle is found."""
    h, w = rgb.shape[:2]
    f = min(1.0, DETECT_LONG_SIDE / max(h, w))
    small = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else rgb
    small = np.ascontiguousarray(small, dtype=np.uint8)
    accepted = []
    for q in _quads(small):
        full = q / f
        pose = card_pose(full, rgb.shape)
        if pose and pose[0] <= MAX_REPROJECTION and pose[1] <= MAX_TILT_DEG:
            accepted.append((cv2.contourArea(full.astype(np.float32)), full))
    if not accepted:
        return None
    coarse = max(accepted, key=lambda a: a[0])[1]
    grey = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    fine = refine_corners(grey, coarse, search_px=max(4.0, 3.0 / f + 2))
    det = from_corners(fine, rgb.shape)
    if det is None or det.reprojection > MAX_REPROJECTION:
        det = from_corners(coarse, rgb.shape)
    return det
