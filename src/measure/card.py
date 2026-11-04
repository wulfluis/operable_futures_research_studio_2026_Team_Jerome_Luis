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
# Fallback search from straight edge pieces (illustrative): low Canny limits for a light card on a light wall,
# the longest LINE_MAX lines, each side at least LINE_MIN_SIDE on edges (a finger may hide part), mean LINE_MIN_MEAN
LINE_CANNY = ((8, 24), (15, 45), (30, 90))
LINE_MAX = 30
MIN_EDGE_SCORE, EDGE_SCORE_SLACK = 0.6, 0.25  # illustrative
MIN_SIDE_CONTRAST = 1.0  # grey levels (illustrative)
BORDER_FRAC = 0.005  # share of the long side: a candidate closer to the photo's edge is rejected
LINE_MIN_SIDE, LINE_MIN_MEAN = 0.45, 0.7


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


def _intersect(l1, l2):
    """Crossing of two lines given as (unit normal angle theta, distance rho): x cos t + y sin t = rho."""
    (t1, r1), (t2, r2) = l1, l2
    A = np.array([[math.cos(t1), math.sin(t1)], [math.cos(t2), math.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([r1, r2]))


def _side_support(edges: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 60) -> float:
    """Share of points along the side a-b (middle 90 %) that lie on an edge pixel."""
    t = np.linspace(0.05, 0.95, n)[:, None]
    p = np.round(a + t * (b - a)).astype(int)
    h, w = edges.shape
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if not ok.all():
        return 0.0
    return float((edges[p[:, 1], p[:, 0]] > 0).mean())


def _line_quads(rgb_small) -> list[tuple[float, np.ndarray]]:
    """Fallback when no closed outline is found (low contrast, a finger over an edge, gaps): straight edge pieces
    are joined into lines, and four lines in two near-parallel pairs make a candidate. Returns (support, corners);
    each side must lie mostly on edges (a finger may hide part of one side)."""
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    h, w = grey.shape
    diag = math.hypot(h, w)
    edges = np.zeros_like(grey)
    for lo, hi in LINE_CANNY:
        edges |= cv2.Canny(grey, lo, hi)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=int(0.035 * diag),
                           maxLineGap=int(0.012 * diag))
    if segs is None:
        return []
    lines = []  # [theta, rho, total length]
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        length = math.hypot(x2 - x1, y2 - y1)
        theta = math.atan2(x2 - x1, -(y2 - y1)) % math.pi  # normal direction
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        for ln in lines:  # join collinear pieces
            dt = abs(ln[0] - theta)
            if min(dt, math.pi - dt) < math.radians(3):
                r = rho if dt < math.pi / 2 else -rho
                if abs(ln[1] - r) < 0.008 * diag:
                    ln[2] += length
                    break
        else:
            lines.append([theta, rho, length])
    lines = sorted(lines, key=lambda ln: -ln[2])[:LINE_MAX]
    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            dt = abs(lines[i][0] - lines[j][0])
            if min(dt, math.pi - dt) < math.radians(15):
                r2 = lines[j][1] if dt < math.pi / 2 else -lines[j][1]
                if abs(lines[i][1] - r2) > 0.04 * diag:
                    pairs.append((lines[i], lines[j]))
    near = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    out = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (a1, a2), (b1, b2) = pairs[i], pairs[j]
            dt = abs(a1[0] - b1[0]) % math.pi
            if min(dt, math.pi - dt) < math.radians(45):
                continue
            corners = [_intersect(a1[:2], b1[:2]), _intersect(b1[:2], a2[:2]), _intersect(a2[:2], b2[:2]),
                       _intersect(b2[:2], a1[:2])]
            if any(c is None for c in corners):
                continue
            q = order_corners(np.array(corners, dtype=np.float32).reshape(4, 1, 2))
            area = cv2.contourArea(q.astype(np.float32))
            if not MIN_AREA_FRAC * h * w <= area <= MAX_AREA_FRAC * h * w:
                continue
            if not cv2.isContourConvex(q.astype(np.float32).reshape(4, 1, 2)):
                continue
            support = [_side_support(near, q[k], q[(k + 1) % 4]) for k in range(4)]
            if min(support) < LINE_MIN_SIDE or sum(support) / 4 < LINE_MIN_MEAN:
                continue
            out.append((sum(support) / 4, q))
    return out


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


def tilt_hint(corners_px) -> str:
    """Which way to straighten the phone, from the card's look: the side that looks larger was closer to the camera.
    Corners in order top-left, top-right, bottom-right, bottom-left (order_corners)."""
    q = np.asarray(corners_px, dtype=float).reshape(4, 2)
    top, bottom = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    ud, lr = max(top, bottom) / max(min(top, bottom), 1e-9), max(left, right) / max(min(left, right), 1e-9)
    if max(ud, lr) < 1.02:
        return ""
    if ud >= lr:
        near, fix = ("lower", "top") if bottom > top else ("upper", "bottom")
    else:
        near, fix = ("left", "right") if left > right else ("right", "left")
    return (f" The {near} edge of the phone was closer to the wall (the card looks larger there): bring the {fix} "
            "of the phone toward the wall.")


def _status(tilt, hint: str = ""):
    if tilt <= TILT_OK_DEG:
        return "straight", f"Photo taken straight enough (tilt {tilt:.0f} degrees)."
    if tilt <= TILT_RETAKE_DEG:
        return "tilted", (f"Photo tilted by {tilt:.0f} degrees: corrected using the card, but hold the phone "
                          "parallel to the wall next time." + hint)
    return "retake", f"Photo tilted by {tilt:.0f} degrees: retake it with the phone parallel to the wall." + hint


def from_corners(corners_px, shape) -> CardDetection | None:
    """Pose and tilt for card corners given by hand (or after detection)."""
    q = order_corners(corners_px)
    pose = card_pose(q, shape)
    if pose is None:
        return None
    err, tilt, lr, ud, long_first = pose
    status, msg = _status(tilt, tilt_hint(q))
    return CardDetection(q, long_first, tilt, lr, ud, err, status, msg)


def _edge_map(rgb_small) -> np.ndarray:
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = np.zeros_like(grey)
    for lo, hi in LINE_CANNY:
        edges |= cv2.Canny(grey, lo, hi)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8))  # a side may lie 2 px off its edge


def _touches_border(q, shape) -> bool:
    """A card must lie wholly inside the photo; a shape running to the edge is a shadow or a wall joined to it."""
    h, w = shape[:2]
    m = BORDER_FRAC * max(h, w)
    return bool((q[:, 0] < m).any() or (q[:, 1] < m).any() or (q[:, 0] > w - 1 - m).any()
                or (q[:, 1] > h - 1 - m).any())


def _inside(inner, outer) -> bool:
    poly = np.asarray(outer, np.float32).reshape(-1, 1, 2)
    tol = 0.01 * float(np.sqrt(abs(cv2.contourArea(poly))))  # a corner may sit just over the outer edge
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -tol for x, y in np.asarray(inner))


def _side_contrast(grey_small, q, d: float = 4.0) -> float:
    """A card is one object: along all four sides the inside differs from the outside the same way (lighter or
    darker). Returns the weakest side's contrast in grey levels when all four agree, else 0. A shape over print or
    over a crack has the same surface on both sides of at least one side."""
    c = q.mean(axis=0)
    diffs = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        p = a + np.linspace(0.1, 0.9, 40)[:, None] * (b - a)
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-u[1], u[0]])
        n = n if np.dot(c - (a + b) / 2, n) > 0 else -n
        inside = ndimage.map_coordinates(grey_small, [(p + n * d)[:, 1], (p + n * d)[:, 0]], order=1, mode="nearest")
        outside = ndimage.map_coordinates(grey_small, [(p - n * d)[:, 1], (p - n * d)[:, 0]], order=1, mode="nearest")
        diffs.append(float(np.median(inside - outside)))
    if min(diffs) > 0:
        return min(diffs)
    return -max(diffs) if max(diffs) < 0 else 0.0


def _edge_score(edges, q) -> float:
    """How well the four sides lie on edges: half the weakest side, half the mean (a finger may hide part of one)."""
    s = [_side_support(edges, q[k], q[(k + 1) % 4]) for k in range(4)]
    return 0.5 * min(s) + 0.5 * sum(s) / 4


def detect_card(rgb: np.ndarray) -> CardDetection | None:
    """Find the bank card in an RGB photo; None if no card-shaped rectangle is found. Of all card-shaped candidates
    the one whose sides lie best on edges wins (not the largest: 4 Nov, a shadow joined to the photo's edge)."""
    h, w = rgb.shape[:2]
    f = min(1.0, DETECT_LONG_SIDE / max(h, w))
    small = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else rgb
    small = np.ascontiguousarray(small, dtype=np.uint8)
    edges = _edge_map(small)
    grey_small = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(float)

    def candidates(quads):
        out = []
        for q in quads:
            if _touches_border(q, small.shape):
                continue
            full = q / f
            pose = card_pose(full, rgb.shape)
            if pose and pose[0] <= MAX_REPROJECTION and pose[1] <= MAX_TILT_DEG:
                area = cv2.contourArea(q.astype(np.float32)) / (small.shape[0] * small.shape[1])
                out.append((_edge_score(edges, q), area, full, _side_contrast(grey_small, q)))
        return out

    # both searches every time: closed outlines, and straight edge pieces (a light card on a light wall, a finger
    # over an edge); a closed outline of some print or stain must not stop the second search (4 Nov)
    accepted = candidates(_quads(small)) + candidates(q for _, q in _line_quads(small))
    if not accepted:
        return None
    # a shape lying wholly inside another card-shaped one is print on the card (a letter, a logo box), not the card
    outer = [a for a in accepted if not any(b is not a and b[1] > 1.5 * a[1] and _inside(a[2], b[2]) for b in accepted)]
    distinct = [a for a in outer if a[3] >= MIN_SIDE_CONTRAST]  # inside differs from outside on all four sides
    outer = distinct or outer
    best = max(a[0] for a in outer)  # of the shapes that lie well on edges, the largest
    good = [a for a in outer if a[0] >= max(MIN_EDGE_SCORE, best - EDGE_SCORE_SLACK)]
    coarse = max(good, key=lambda a: (a[1], a[0]))[2] if good else max(outer, key=lambda a: a[0])[2]
    grey = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    fine = refine_corners(grey, coarse, search_px=max(4.0, 3.0 / f + 2))
    det = from_corners(fine, rgb.shape)
    if det is None or det.reprojection > MAX_REPROJECTION:
        det = from_corners(coarse, rgb.shape)
    return det
