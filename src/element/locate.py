"""Where on the element the crack is: the crack close-up found inside the element photo. Spec:
spec/features/element_position.md (D58).

The close-up is linked to the element photo through the bank card, when it lies in both photos (D59): its four
corners in each give the wall plane's mapping exactly, however thin the crack looks from afar. Without a card, or when
the card is not found far away, SIFT key points and a homography (OpenCV) are tried. The column's or beam's long
edges in the element photo give its axis and depth; the crack's distance to the element's seen ends gives end zone or
middle. Anything short of a clear answer is "unknown", which the risk matrix counts as an end zone (rule 2).
Every limit here is illustrative until checked on real photo pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

LINK_SIDE = 1600  # px, long side of both photos for the link
RATIO = 0.75  # Lowe's ratio test
MIN_INLIERS = 20  # matching points that fit the homography (illustrative)
MAX_AREA_SHARE = 0.8  # the close-up must land as a smaller patch of the element photo (illustrative)
EDGE_MIN_SHARE = 0.35  # an element edge is at least this share of the photo's long side (illustrative)
AXIS_TOL_DEG = 15.0  # element edges within this of the photo's vertical (column) or horizontal (beam) (illustrative)
SAME_LINE_SHARE = 0.015  # pieces closer than this share of the long side across the axis are one edge
BORDER_SHARE = 0.02  # an end this close to the photo border is cut by the photo, so not seen
END_ZONE_DEPTHS = 1.0  # end zone: about one element depth from the end (risk_matrix_v3.json, assumption)
EDGE_SIDE_PX = 8  # px, width of the band either side of an edge whose brightness is compared
EDGE_CONTRAST = 10.0  # grey levels: an element edge steps at least this much (illustrative)
EDGE_WINDOW_PX = 31  # px along the edge over which the step is judged
EDGE_KEEP = 0.6  # share of that window that must step (illustrative)
AXIS_ELEMENTS = {"column": 0.0, "beam": 90.0}  # axis angle to the photo's vertical
CARD_PATCH_PX = (86, 54)  # the card straightened to 1 px per mm, long side across, to compare its look (D59)
MIN_LOOK = 0.5  # how alike the card must look in both photos (correlation, illustrative)
LOOK_MARGIN = 0.1  # the best candidate must beat the next by this, or the card is not sure (illustrative)
PLAIN_STD = 4.0  # grey levels: a card this plain has no look to compare; then it is not used as the bridge
LOOK_SIDE = 2000  # px: the search for the card's look runs on a copy of the element photo this size
LOOK_MIN_SHARE = 0.012  # the far card's long side, smallest share of the photo's long side searched (about 5 m)
LOOK_SCALES = 20
LOOK_TRIES = 8  # places, best look first, checked for a card's outline
CARD_MARGIN = 0.3  # share of the card's size of wall kept round it in its look: a dark card with little print is
# almost uniform inside and matches any dark patch; with the wall round it, it is a card on a wall (D59, field test)
TURN_MARGIN = 0.1  # the card counts as turned half round only when that look is clearly better (photos upright)


@dataclass
class Location:
    linked: bool
    position: str  # "end_zone", "middle" or "unknown" (keys of app.core.POSITIONS)
    reason: str
    inliers: int = 0
    axis_to_vertical_deg: float | None = None  # the element's axis in the element photo
    crack_to_axis_deg: float | None = None  # 0 = along the element, 90 = across it
    depth_px: float | None = None
    via: str | None = None  # "bank card" or "key points"
    crack_to_vertical_deg: float | None = None  # against the room's true vertical (D61); 0 = vertical, 90 = level
    direction: str | None = None  # "vertical", "diagonal", "horizontal" or "in between"
    diag: dict = field(default_factory=dict)  # what each step found, shown for testing
    preview: Image.Image | None = field(default=None, repr=False)  # element photo with the link drawn; never shared
    crack_px: np.ndarray | None = field(default=None, repr=False)  # the crack's line in the element photo's pixels

    def angle_for_rules(self, element: str) -> float | None:
        """The angle the type rules read as 'to vertical' (D43), taken against the element's own axis."""
        if self.crack_to_axis_deg is None or element not in AXIS_ELEMENTS:
            return None
        return self.crack_to_axis_deg if element == "column" else 90.0 - self.crack_to_axis_deg

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 1)  # noqa: E731
        return {"linked": self.linked, "position": self.position, "reason": self.reason, "inliers": self.inliers,
                "axis_to_vertical_deg": r(self.axis_to_vertical_deg), "crack_to_axis_deg": r(self.crack_to_axis_deg),
                "depth_px": r(self.depth_px), "via": self.via, "crack_to_vertical_deg": r(self.crack_to_vertical_deg),
                "direction": self.direction, "diag": self.diag,
                "source": "the two photos linked (D58, D59)"}


def to_link_size(grey: np.ndarray) -> tuple[np.ndarray, float]:
    """Grey uint8 with long side at most LINK_SIDE, and the factor from the original."""
    k = min(1.0, LINK_SIDE / max(grey.shape))
    g = grey if k == 1.0 else cv2.resize(grey, (round(grey.shape[1] * k), round(grey.shape[0] * k)),
                                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g, dtype=np.uint8), k


def link(close: np.ndarray, element: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Homography from the close-up to the element photo (both grey uint8, link size), and its inlier count."""
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(close, None)
    k2, d2 = sift.detectAndCompute(element, None)
    if d1 is None or d2 is None or len(k1) < MIN_INLIERS or len(k2) < MIN_INLIERS:
        return None, 0
    good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2)
            if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < MIN_INLIERS:
        return None, n
    h, w = close.shape
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    area = cv2.contourArea(quad)
    if not cv2.isContourConvex(quad.astype(np.float32)) or not 0 < area < MAX_AREA_SHARE * element.size:
        return None, n
    return H, n


VERTICAL_MIN_SHARE = 0.15  # a room line (frame, corner, ceiling) is at least this share of the long side (D61)
VERTICAL_TOL_DEG = 15.0  # within this of the photo's vertical or horizontal (illustrative)
VERTICAL_MIN_TOTAL = 0.3  # the lines together at least this share of the long side, else no true vertical


def true_vertical(grey: np.ndarray, at: np.ndarray) -> tuple[np.ndarray | None, int]:
    """The room's true vertical near `at` (D61): long straight lines near the photo's vertical (frames, corners) or
    horizontal (ceiling, cornice, lintel; by their perpendicular), each counted by its length and more when near
    `at`. Returns (unit vector pointing down the photo, number of lines), or (None, n) when there are too few."""
    side = max(grey.shape)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=int(VERTICAL_MIN_SHARE * side),
                           maxLineGap=int(0.01 * side))
    if segs is None:
        return None, 0
    tilts, weights, total = [], [], 0.0
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        d = np.array([x2 - x1, y2 - y1])
        length = float(np.linalg.norm(d))
        a = float(np.degrees(np.arctan2(d[0], d[1])))  # 0 = straight down the photo
        a = (a + 90.0) % 180.0 - 90.0  # -90..90 off vertical
        if abs(a) <= VERTICAL_TOL_DEG:
            tilt = a
        elif abs(abs(a) - 90.0) <= VERTICAL_TOL_DEG:
            tilt = a - 90.0 if a > 0 else a + 90.0  # a horizontal line's tilt off level, as a vertical's
        else:
            continue
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        tilts.append(tilt)
        weights.append(length / (1.0 + float(np.linalg.norm(mid - at)) / side))
        total += length
    if len(tilts) < 2 or total < VERTICAL_MIN_TOTAL * side:
        return None, len(tilts)
    order = np.argsort(tilts)
    cum = np.cumsum(np.asarray(weights)[order])
    tilt = float(np.asarray(tilts)[order][np.searchsorted(cum, cum[-1] / 2)])  # weighted median
    r = np.radians(tilt)
    return np.array([np.sin(r), np.cos(r)]), len(tilts)


def direction_word(angle_to_vertical: float) -> str:
    """The bands of the type rules (D43): vertical up to 25, diagonal 35 to 55, horizontal from 65 degrees."""
    if angle_to_vertical <= 25:
        return "vertical"
    if 35 <= angle_to_vertical <= 55:
        return "diagonal"
    if angle_to_vertical >= 65:
        return "horizontal"
    return "in between"


def _direction(pts: np.ndarray) -> np.ndarray:
    q = pts - pts.mean(axis=0)
    d = np.linalg.eigh(q.T @ q)[1][:, -1] if len(pts) >= 3 else pts[-1] - pts[0]
    return d / max(np.linalg.norm(d), 1e-12)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """0 to 90 degrees between two lines (no arrow)."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def element_edges(grey: np.ndarray, element: str, centre: np.ndarray):
    """The nearest long straight edge on each side of `centre` along the element's axis, as
    (axis unit vector, edge offsets across the axis, seen extent along the axis), or None."""
    side = max(grey.shape)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=int(0.1 * side),
                           maxLineGap=int(0.02 * side))
    if segs is None:
        return None
    nominal = np.array([0.0, 1.0]) if element == "column" else np.array([1.0, 0.0])
    normal = np.array([-nominal[1], nominal[0]])
    pieces = []  # offset across the axis, start and end along it, direction
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        d = np.array([x2 - x1, y2 - y1])
        length = np.linalg.norm(d)
        d /= max(length, 1e-12)
        if _angle_between(d, nominal) > AXIS_TOL_DEG:
            continue
        a, b = np.array([x1, y1]) - centre, np.array([x2, y2]) - centre
        pieces.append(((a @ normal + b @ normal) / 2, *sorted((a @ nominal, b @ nominal)),
                       d if d @ nominal >= 0 else -d))
    if not pieces:
        return None
    pieces.sort(key=lambda p: p[0])
    lines, tol = [], SAME_LINE_SHARE * side  # pieces on one line joined: offset, start, end, length, direction
    for off, s, e, d in pieces:
        if lines and abs(off - lines[-1][0]) <= tol:
            o, s0, e0, n, dd = lines[-1]
            lines[-1] = [(o * n + off * (e - s)) / (n + e - s), min(s0, s), max(e0, e), n + e - s, dd + d * (e - s)]
        else:
            lines.append([off, s, e, e - s, d * (e - s)])
    smooth = cv2.GaussianBlur(grey, (5, 5), 0).astype(np.float32)
    kept = []  # offset, direction, seen start and end along the axis (from the crack's level)
    for off, s, e, _, dsum in lines:
        d = dsum / max(np.linalg.norm(dsum), 1e-12)
        run = _edge_run(smooth, centre + off * normal, d, s, e)
        if run is not None and run[1] - run[0] >= EDGE_MIN_SHARE * side:
            kept.append((off, d, *run))
    left = [k for k in kept if k[0] < 0]
    right = [k for k in kept if k[0] > 0]
    if not left or not right:
        return None
    a, b = max(left, key=lambda k: k[0]), min(right, key=lambda k: k[0])
    axis = a[1] + b[1]
    axis /= max(np.linalg.norm(axis), 1e-12)
    return axis, (a[0], b[0]), (max(a[2], b[2]), min(a[3], b[3]))  # the nearer end of the two edges


def _edge_run(smooth: np.ndarray, p0: np.ndarray, d: np.ndarray, s: float, e: float) -> tuple[float, float] | None:
    """Where along the line through p0 (direction d) the brightness steps the same way across it, as one unbroken
    run through the crack's level (t = 0): (start, end) in px along d. Random texture lines and lines that run past
    the element's end stop stepping, so the run ends where the element's edge ends. None: not an edge at the crack."""
    h, w = smooth.shape
    n = np.array([-d[1], d[0]])
    t = np.arange(-np.hypot(h, w), np.hypot(h, w), 1.0)
    pts = p0 + t[:, None] * d
    m = EDGE_SIDE_PX + 2
    inside = (pts[:, 0] >= m) & (pts[:, 0] < w - 1 - m) & (pts[:, 1] >= m) & (pts[:, 1] < h - 1 - m)
    t, pts = t[inside], pts[inside]
    if len(t) < 10 or not t[0] <= 0 <= t[-1]:
        return None
    def band(sign):  # mean brightness of a band EDGE_SIDE_PX wide beside the line, 2 px away from it
        vals = [ndimage.map_coordinates(smooth, [(pts + sign * k * n)[:, 1], (pts + sign * k * n)[:, 0]], order=1)
                for k in range(2, 2 + EDGE_SIDE_PX)]
        return np.mean(vals, axis=0)

    diff = band(1) - band(-1)
    on_piece = (t >= s) & (t <= e)
    sign = np.sign(np.median(diff[on_piece])) if on_piece.any() else 0.0
    if sign == 0:
        return None
    good = ndimage.uniform_filter1d((sign * diff >= EDGE_CONTRAST).astype(float), EDGE_WINDOW_PX) >= EDGE_KEEP
    i = int(np.argmin(np.abs(t)))
    if not good[i]:
        return None
    lo, hi = i, i
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < len(t) - 1 and good[hi + 1]:
        hi += 1
    return float(t[lo]), float(t[hi])


def _cut_by_border(point: np.ndarray, shape) -> bool:
    m = BORDER_SHARE * max(shape)
    return point[0] < m or point[1] < m or point[0] > shape[1] - 1 - m or point[1] > shape[0] - 1 - m


def card_patch(rgb: np.ndarray, corners) -> tuple[np.ndarray, np.ndarray]:
    """The card straightened (1 px per mm, long side across) and its corners in that order: clockwise from the top
    left on screen, starting on a long side. The same rule in both photos keeps the corners matched while both are
    taken roughly upright; the look then also tells a card turned half round."""
    from measure.card import order_corners
    q = order_corners(corners).astype(np.float32)
    if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1]):
        q = np.roll(q, -1, axis=0)
    w, h = CARD_PATCH_PX
    m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(np.ascontiguousarray(rgb[..., :3]), m, (w, h), flags=cv2.INTER_AREA), q


def card_view(rgb: np.ndarray, q: np.ndarray, margin: float = CARD_MARGIN) -> np.ndarray:
    """The card straightened as card_patch, with a band of the wall round it (corners q in card_patch's order)."""
    w, h = CARD_PATCH_PX
    mx, my = margin * w, margin * h
    dst = np.float32([[mx, my], [mx + w, my], [mx + w, my + h], [mx, my + h]])
    m = cv2.getPerspectiveTransform(np.float32(q), dst)
    return cv2.warpPerspective(np.ascontiguousarray(rgb[..., :3]), m, (round(w + 2 * mx), round(h + 2 * my)),
                               flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _look(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two straightened cards, colour by colour, on their inner part, blurred (the far one is small)."""
    def norm(p):
        p = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 1.5)[5:-5, 5:-5]
        p = p - p.mean(axis=(0, 1))
        return p / max(float(p.std()), 1e-6)
    return float(np.mean(norm(a) * norm(b)))


def turn_in_photo(corners: np.ndarray) -> int:
    """Quarter turns (np.rot90) that bring the straightened card back to how it lay in the photo: a card lying
    upright on the wall (long side vertical) must be searched for upright."""
    v = np.asarray(corners[1], float) - np.asarray(corners[0], float)
    return -int(round(np.degrees(np.arctan2(v[1], v[0])) / 90.0)) % 4


def find_by_look(element_rgb: np.ndarray, patch_c: np.ndarray, close_size: float, turn: int = 0,
                 min_share: float = LOOK_MIN_SHARE):
    """Where the card from the close-up may be in the element photo, by its look, over sizes from a few metres away
    to the close-up's own. Up to LOOK_TRIES places, best first: (score, box x0, y0, x1, y1 in full pixels, the best
    score of another place at about the same size)."""
    grey = cv2.cvtColor(np.ascontiguousarray(element_rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    k = min(1.0, LOOK_SIDE / max(grey.shape))
    img = cv2.resize(grey, (round(grey.shape[1] * k), round(grey.shape[0] * k)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    tmpl = np.ascontiguousarray(np.rot90(cv2.cvtColor(patch_c, cv2.COLOR_RGB2GRAY).astype(np.float32), turn))
    side = max(img.shape)
    hits = []  # score, x, y, w, h, turned; at each size the best place and the best place apart from it
    for share in np.geomspace(min_share, min(0.6, close_size), LOOK_SCALES):
        long_ = share * side
        w = int(round(long_ * tmpl.shape[1] / max(tmpl.shape)))
        h = int(round(long_ * tmpl.shape[0] / max(tmpl.shape)))
        if min(w, h) < 8 or w >= img.shape[1] or h >= img.shape[0]:
            continue
        t0 = cv2.GaussianBlur(cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
        for turned, t in ((False, t0), (True, np.ascontiguousarray(np.rot90(t0, 2)))):
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            hits.append((float(score), x, y, w, h, turned))
            res[max(0, y - h):y + h + 1, max(0, x - w):x + w + 1] = -1  # the next best place at this size
            _, s2, _, (x2, y2) = cv2.minMaxLoc(res)
            hits.append((float(s2), x2, y2, w, h, turned))
    hits.sort(key=lambda s: -s[0])
    out = []
    for s, x, y, w, h, _ in hits:
        if len(out) == LOOK_TRIES:
            break
        if any(abs(x + w / 2 - (b[0] + b[2]) * k / 2) < w and abs(y + h / 2 - (b[1] + b[3]) * k / 2) < h
               for _, b, _ in out):
            continue  # the same place at another size
        out.append((s, (x / k, y / k, (x + w) / k, (y + h) / k), _rival(hits, x, y, w, h)))
    return out


def _rival(hits, x, y, w, h) -> float:
    """The best score of another place at about the same size as the box (x, y, w, h)."""
    def apart(o):  # another place at about the same size: a second card-like thing (a tiny size matches any stripe)
        cx, cy = o[1] + o[3] / 2, o[2] + o[4] / 2
        same_size = 1 / 1.3 <= o[3] / w <= 1.3
        return same_size and not (x - w / 2 <= cx <= x + 1.5 * w and y - h / 2 <= cy <= y + 1.5 * h)
    return max((o[0] for o in hits if apart(o)), default=-1.0)


def card_link(close_card: tuple[np.ndarray, np.ndarray], close_size: float, element_rgb: np.ndarray, kc: float,
              ke: float, diag: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Homography from the close-up (link size) to the element photo (link size) through the bank card (D59), the
    card's corners in the element photo (full pixels), and why not when there is none. close_card: (corners in
    close-up pixels, as card_patch ordered them; the straightened card); close_size: the card's long side / the
    close-up's long side. The card is found by its look, then its corners by its outline near there."""
    from measure.card import find_card_near
    corners_c, patch_c = close_card[0], close_card[1]
    view_c = close_card[2] if len(close_card) > 2 else patch_c
    diag = {} if diag is None else diag
    diag["card_plainness"] = round(float(cv2.cvtColor(patch_c, cv2.COLOR_RGB2GRAY).std()), 1)
    if diag["card_plainness"] < PLAIN_STD:
        return None, None, "the card is plain, so it cannot be told apart from other card-shaped things"
    why = "the bank card was not found in the element photo"
    turn = turn_in_photo(corners_c)
    diag["card_upright_in_photo"] = turn % 2 == 1
    grow = 1 + 2 * CARD_MARGIN if view_c is not patch_c else 1.0
    places = []
    for score, (x0, y0, x1, y1), rival in find_by_look(element_rgb, view_c, close_size * grow, turn,
                                                         LOOK_MIN_SHARE * grow):
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / grow / 2, (y1 - y0) / grow / 2
        places.append((score, (cx - hw, cy - hh, cx + hw, cy + hh), rival))  # the card itself, without the wall
    diag["look_scores"] = [round(s, 2) for s, _, _ in places]
    diag["look_boxes"] = [[round(v) for v in b] for _, b, _ in places]
    diag["outline_found"], diag["outline_look"] = [], []
    for score, box, rival in places:
        if score < MIN_LOOK:
            break
        q = find_card_near(element_rgb, box)
        diag["outline_found"].append(q is not None)
        if q is None:
            continue  # looks a little like the card, but has no card's outline: try the next place
        inside, qo = card_patch(element_rgb, q)
        view = inside if view_c is patch_c else card_view(element_rgb, qo)  # compare like with like
        upright_look, turned_look = _look(view_c, view), _look(view_c, np.rot90(view, 2))
        half = turned_look > upright_look + TURN_MARGIN
        look = turned_look if half else upright_look
        diag["outline_look"].append(round(look, 2))
        if look < MIN_LOOK:
            continue
        if score - rival < LOOK_MARGIN:
            return None, None, "more than one place in the element photo looks like the card"
        qo = np.roll(qo, -2, axis=0) if half else qo
        H = cv2.getPerspectiveTransform(np.float32(corners_c * kc), np.float32(qo * ke))
        return H, qo, ""
    return None, None, why


def locate(close_grey: np.ndarray, crack_px: np.ndarray | None, element_rgb: np.ndarray, element: str | None,
           close_card: tuple[np.ndarray, np.ndarray] | None = None, close_card_size: float | None = None,
           ) -> Location:
    """close_grey: the crack photo in grey (any size); crack_px: its crack line, (n, 2) x, y in close_grey pixels;
    element_rgb: the element photo; element: the matrix element the model read (column, beam, ...) or None;
    close_card: the bank card in the close-up (corners in close_grey pixels, ordered by card_patch, and the
    straightened card), when one was found there; close_card_size: its long side / close_grey's long side."""
    eg, ke = to_link_size(cv2.cvtColor(np.ascontiguousarray(element_rgb[..., :3]), cv2.COLOR_RGB2GRAY))
    cg, kc = to_link_size(close_grey.astype(np.uint8))
    H, n, via, card_q, why_not = None, 0, None, None, ""
    diag: dict = {"card_in_close_up": close_card is not None, "crack_line_in_close_up": crack_px is not None,
                  "element_photo_px": list(element_rgb.shape[1::-1])}
    if close_card is not None:
        H, card_q, why_not = card_link(close_card, close_card_size or 1.0, element_rgb, kc, ke, diag)
        via = "bank card" if H is not None else None
        diag["card_link"] = "linked" if H is not None else why_not
    if H is None:
        H, n = link(cg, eg)
        via = "key points" if H is not None else None
        diag["key_points"] = n
    if H is None:
        why = (why_not + ", and " if why_not else "") + "too few matching points"
        boxes = [np.array(b, float) * ke for b in diag.get("look_boxes", [])[:1]]
        return Location(False, "unknown", f"The two photos could not be linked ({why}): position not known.", n,
                        diag=diag, preview=_preview(element_rgb, ke, look_boxes=boxes))
    card_e = None if card_q is None else card_q * ke
    if crack_px is None or len(crack_px) < 2:
        return Location(True, "unknown", "No crack line to place on the element: position not known.", n,
                        via=via, diag=diag, preview=_preview(element_rgb, ke, H, cg.shape, card=card_e))
    crack = cv2.perspectiveTransform(np.float32(crack_px * kc)[None], H)[0].astype(float)
    centre = crack.mean(axis=0)
    down, n_lines = true_vertical(eg, centre)  # the room's vertical where the crack is (D61)
    diag["room_lines"] = n_lines
    to_vertical = None if down is None else _angle_between(_direction(crack), down)
    diag["true_vertical_found"] = down is not None

    def with_direction(loc: Location) -> Location:
        loc.crack_px = crack / ke  # for the part it lies on (D62); in the element photo's own pixels
        if to_vertical is not None:
            loc.crack_to_vertical_deg, loc.direction = to_vertical, direction_word(to_vertical)
        return loc
    if element not in AXIS_ELEMENTS:
        why = "a wall has no ends to read" if element and "wall" in element else "the element is not clear"
        return with_direction(Location(True, "unknown", f"Photos linked through the {via}, but {why}: position "
                                                        "not known.", n, via=via, diag=diag,
                                       preview=_preview(element_rgb, ke, H, cg.shape, crack, card=card_e)))
    found = element_edges(eg, element, centre)
    diag["element_edges"] = found is not None
    if found is None:
        return with_direction(Location(True, "unknown", f"Photos linked through the {via}, but the {element}'s two "
                                                        "long edges were not found: position not known.", n, via=via,
                                       diag=diag, preview=_preview(element_rgb, ke, H, cg.shape, crack, card=card_e)))
    axis, (o1, o2), (s, e) = found
    normal = np.array([-axis[1], axis[0]])
    depth = abs(o2 - o1)
    along = (crack - centre) @ axis
    to_axis = _angle_between(_direction(crack), axis)
    ends = {"start": centre + s * axis, "end": centre + e * axis}
    seen = {k: not _cut_by_border(p, eg.shape) for k, p in ends.items()}
    gap_start, gap_end = along.min() - s, e - along.max()  # crack's nearest point to each end
    zone = END_ZONE_DEPTHS * depth
    near = (seen["start"] and gap_start <= zone) or (seen["end"] and gap_end <= zone)
    if near:
        position, why = "end_zone", f"part of the crack lies within one {element} depth of its end"
    elif all(seen.values()):
        position, why = "middle", f"both ends of the {element} are seen and the crack is more than one depth from each"
    else:
        position, why = "unknown", f"an end of the {element} is cut by the photo"
    how = "the bank card" if via == "bank card" else f"{n} matching points"
    loc = Location(True, position, f"Photos linked through {how}: {why}.", n,
                   float(np.degrees(np.arccos(min(1.0, abs(axis[1]))))), to_axis, depth, via, diag=diag)
    loc.preview = _preview(element_rgb, ke, H, cg.shape, crack, (centre, axis, normal, (o1, o2), (s, e)), card_e)
    return with_direction(loc)


def _preview(rgb, k, H=None, close_shape=None, crack=None, edges=None, card=None, look_boxes=()) -> Image.Image:
    """The element photo, small, with the close-up's outline (white), the card (cyan), the crack (red, with a ring
    so a thin crack can be found) and the two edges (yellow)."""
    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3])).convert("RGB")
    img = img.resize((round(img.width * k), round(img.height * k)))
    draw = ImageDraw.Draw(img)
    lw = max(2, img.width // 250)
    if H is not None:
        h, w = close_shape
        q = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
        draw.line([tuple(p) for p in q] + [tuple(q[0])], fill=(255, 255, 255), width=lw)
    if edges is not None:
        c, axis, normal, offs, (s, e) = edges
        for o in offs:
            draw.line([tuple(c + o * normal + s * axis), tuple(c + o * normal + e * axis)], fill=(255, 210, 0),
                      width=lw)
    for b in look_boxes:  # not linked: where the card's look matched best, for testing
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 140, 0), width=lw)
    if card is not None:
        draw.line([tuple(p) for p in card] + [tuple(card[0])], fill=(0, 220, 255), width=lw)
    if crack is not None:
        draw.line([tuple(p) for p in crack], fill=(230, 30, 30), width=lw + 1)
        c = crack.mean(axis=0)
        r = max(12.0, 0.6 * float(np.linalg.norm(crack.max(axis=0) - crack.min(axis=0))))
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=(230, 30, 30), width=lw)
    return img
