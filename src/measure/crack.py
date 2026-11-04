"""Crack geometry: crack pixels, centre line, width and length in millimetres. Spec: spec/features/measurement.md.

Method, after [46] ch. 5.2 and the tools it uses [47][48]: a binary crack mask, its centre line (skeleton), and the
opening measured perpendicular to the centre line at every centre-line point. The width at a point is read from the
grey-level profile across the crack (edges at half the depth between wall and crack, sub-pixel); where the profile has
too little contrast it falls back to the mask edges. Uncertainty: +-1 pixel at the local pixel size [46] (adopted
from [46]; our own accuracy is still to be checked on the width lines of the scale card), plus the scale error.
Assumes the crack is darker than the wall. Every threshold here is illustrative until tested on the pilot photos.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

MIN_PX_ACROSS = 3.0  # [46]: below 3 pixels across the crack, errors grow quickly
GOOD_PX_ACROSS = 6.0  # [46]: 6 to 8 pixels across recommended (about +-15 %)
WIDTH_UNCERTAINTY_PX = 1.0  # [46] ch. 5.2
BACKGROUND_PX = 41  # illustrative; must be wider than the widest crack in pixels
MIN_COMPONENT_PX = 30  # illustrative
MIN_ELONGATION = 3.0  # illustrative
MIN_CONTRAST = 8.0  # grey levels between wall and crack, illustrative
FAR_FROM_REFERENCE_MM = 150.0  # illustrative: beyond it the pixel size may differ from the reference [46]
DIRECTION_RADIUS_PX = 5
SMOOTH_RADIUS_PX = 3
STEP_PX = 0.25
EIGHT = np.ones((3, 3), dtype=bool)


# ---------- crack pixels ----------

def otsu(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0 or v.max() == v.min():
        return float(v.max()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = ((hist * centres).sum() - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def keep_crack_components(mask: np.ndarray) -> np.ndarray:
    """Keep pieces that are large enough and thin: long (elongated) or spread out (a network of cracks)."""
    labels, n = ndimage.label(mask, structure=EIGHT)
    keep = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        ys, xs = np.nonzero(labels[sl] == i)
        if len(ys) < MIN_COMPONENT_PX:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.c_[xs, ys].T) + 0.25 * np.eye(2))
        elongation = float(np.sqrt(ev[-1] / ev[0]))
        fill = len(ys) / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
        keep[i] = elongation >= MIN_ELONGATION or fill < 0.3
    return keep[labels]


def segment(grey: np.ndarray, roi=None, exclude: np.ndarray | None = None, background_px: int = BACKGROUND_PX,
            threshold: float | None = None) -> np.ndarray:
    """Classical crack mask (until the U-Net exists): pixels darker than the wall around them (black-hat),
    thresholded with Otsu inside the search box `roi` = (x0, y0, x1, y1). `exclude` marks pixels never taken
    (the reference object and its edge)."""
    h, w = grey.shape
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else (int(round(v)) for v in roi)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    out = np.zeros((h, w), dtype=bool)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return out
    p = background_px
    xa, ya, xb, yb = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
    g = ndimage.gaussian_filter(grey[ya:yb, xa:xb].astype(float), 1.0)
    blackhat = ndimage.grey_closing(g, size=(p, p)) - g
    inner = blackhat[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    valid = np.ones_like(inner, dtype=bool) if exclude is None else ~exclude[y0:y1, x0:x1]
    if not valid.any():
        return out
    t = otsu(inner[valid]) if threshold is None else threshold
    m = (inner > t) & valid
    m = ndimage.binary_closing(m, structure=EIGHT) & valid
    out[y0:y1, x0:x1] = keep_crack_components(m)
    return out


# ---------- centre line ----------

def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a one-pixel centre line."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            c = img[1:-1, 1:-1]
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(r.astype(np.int16) for r in ring[:8])
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if remove.any():
                c[remove] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _graph(skel: np.ndarray):
    """Centre-line pixels as (x, y) and their neighbour lists. A diagonal link is dropped when a 4-neighbour
    already joins the two pixels, so corners do not form false junctions."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    idx = np.full((skel.shape[0] + 2, skel.shape[1] + 2), -1, dtype=np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    a_list, b_list = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        nb = idx[ys + 1 + dy, xs + 1 + dx]
        ok = nb >= 0
        if dy and dx:
            ok &= (idx[ys + 1, xs + 1 + dx] < 0) & (idx[ys + 1 + dy, xs + 1] < 0)
        a_list.append(np.arange(n)[ok])
        b_list.append(nb[ok])
    a, b = np.concatenate(a_list), np.concatenate(b_list)
    nbrs = [[] for _ in range(n)]
    for i, j in zip(a.tolist(), b.tolist()):
        nbrs[i].append(j)
        nbrs[j].append(i)
    return np.c_[xs, ys].astype(float), nbrs, a, b


def trace_branches(nbrs) -> list[list[int]]:
    """Split the centre line into branches running between end points and junctions (closed loops included)."""
    deg = [len(x) for x in nbrs]
    seen = set()
    branches = []

    def walk(start, nxt):
        path, prev, cur = [start, nxt], start, nxt
        seen.add((min(start, nxt), max(start, nxt)))
        while deg[cur] == 2:
            n2 = nbrs[cur][0] if nbrs[cur][1] == prev else nbrs[cur][1]
            e = (min(cur, n2), max(cur, n2))
            if e in seen:
                break
            seen.add(e)
            path.append(n2)
            prev, cur = cur, n2
        return path

    for s in range(len(nbrs)):
        if deg[s] != 2:
            for t in nbrs[s]:
                if (min(s, t), max(s, t)) not in seen:
                    branches.append(walk(s, t))
    for s in range(len(nbrs)):
        if deg[s] == 2 and (min(s, nbrs[s][0]), max(s, nbrs[s][0])) not in seen:
            branches.append(walk(s, nbrs[s][0]))
    return branches


def prune(skel: np.ndarray, spur_px: int) -> np.ndarray:
    """Remove short side branches (spurs) that run from a junction to a free end."""
    skel = skel.copy()
    for _ in range(3):
        pts, nbrs, _, _ = _graph(skel)
        deg = [len(x) for x in nbrs]
        removed = False
        for br in trace_branches(nbrs):
            ends = (deg[br[0]], deg[br[-1]])
            if 1 in ends and max(ends) >= 3 and len(br) <= spur_px:
                body = br[1:] if deg[br[0]] >= 3 else br[:-1]
                xy = pts[body].astype(int)
                skel[xy[:, 1], xy[:, 0]] = False
                removed = True
        if not removed:
            break
    return skel


def path_length_mm(path_xy: np.ndarray, to_mm, window: int = 5) -> float:
    """Length of a pixel chain, smoothed so that pixel staircases do not add length."""
    p = np.asarray(path_xy, dtype=float)
    if len(p) < 2:
        return 0.0
    if len(p) > window:
        k = window // 2
        kernel = np.ones(window) / window
        sm = np.column_stack([np.convolve(np.pad(p[:, j], k, mode="edge"), kernel, mode="valid") for j in (0, 1)])
        sm[0], sm[-1] = p[0], p[-1]
        p = sm
    q = to_mm(p)
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


# ---------- width ----------

def _crossing(t, v, start, step, level, rising):
    """Walk from index start by step until v crosses level; the crossing position in t, linearly interpolated."""
    j = start
    while 0 <= j + step < len(v):
        a, b = v[j], v[j + step]
        if (rising and a < level <= b) or (not rising and a >= level > b):
            return t[j] + (level - a) / (b - a) * (t[j + step] - t[j])
        j += step
    return None


def profile_width(grey_s, maskf, x, y, nx, ny, half):
    """Width in pixels across the crack at (x, y) along the normal (nx, ny): (left t, right t, source)."""
    span = 2 * half + 6
    t = np.arange(-span, span + 1e-9, STEP_PX)
    cx, cy = x + t * nx, y + t * ny
    m = ndimage.map_coordinates(maskf, [cy, cx], order=1, mode="constant", cval=0.0)
    c0 = len(t) // 2
    if m[c0] < 0.5:
        return None
    ml, mr = _crossing(t, m, c0, -1, 0.5, False), _crossing(t, m, c0, 1, 0.5, False)
    if ml is None or mr is None:
        return None
    mask_w = mr - ml
    g = ndimage.map_coordinates(grey_s, [cy, cx], order=1, mode="nearest")
    core = np.flatnonzero(np.abs(t) <= half + 1)
    ic = core[np.argmin(g[core])]
    left = g[(t >= -(half + 6)) & (t <= -(half + 2))]
    right = g[(t >= half + 2) & (t <= half + 6)]
    if len(left) and len(right):
        bl, br, c = float(np.median(left)), float(np.median(right)), float(g[ic])
        if min(bl, br) - c >= MIN_CONTRAST:
            gl = _crossing(t, g, ic, -1, (bl + c) / 2, True)
            gr = _crossing(t, g, ic, 1, (br + c) / 2, True)
            if gl is not None and gr is not None and 0.5 * mask_w - 1 <= gr - gl <= 1.5 * mask_w + 2:
                return gl, gr, "grey"
    return ml, mr, "mask"


# ---------- result ----------

@dataclass
class CrackMeasurement:
    found: bool
    width_max_mm: float | None = None  # widest point, smoothed over about 3 px of centre line
    width_uncertainty_mm: float | None = None  # +-1 px at the widest point plus the scale error
    width_median_mm: float | None = None
    px_across: float | None = None  # pixels across the crack at the widest point
    mm_per_px: float | None = None  # median pixel size across the crack
    length_main_mm: float | None = None  # longest path along the centre line
    length_total_mm: float | None = None  # all branches
    n_branches: int = 0
    n_pieces: int = 0
    widest_point_px: tuple | None = None
    width_source_grey: float | None = None  # share of widths read from the grey profile (rest: mask edges)
    # shape, for the pattern rules (src/triage/pattern_rules.py); angles assume the phone was held upright
    angle_to_vertical_deg: float | None = None  # main crack direction: 0 = vertical, 90 = horizontal, on the wall
    straightness: float | None = None  # end-to-end distance / length of the main path (1 = straight)
    coherence: float | None = None  # 1 = all of the centre line runs one way, 0 = no preferred direction
    n_loops: int = 0  # closed cells in the centre line (a network of cracks has many)
    branch_ratio: float | None = None  # length of all branches / main length
    flags: list[str] = field(default_factory=list)
    # for drawing, full-image pixel coordinates; not written to JSON
    mask: np.ndarray | None = field(default=None, repr=False)
    skeleton_px: np.ndarray | None = field(default=None, repr=False)
    main_path_px: np.ndarray | None = field(default=None, repr=False)
    widest_edges_px: np.ndarray | None = field(default=None, repr=False)

    def matrix_input(self) -> dict | None:
        """Inputs for triage.matrix.assess(); None when the width cannot be trusted (then Expert needed, D3)."""
        if not self.found or self.width_max_mm is None or self.px_across is None or self.px_across < MIN_PX_ACROSS:
            return None
        return {"width_mm": round(self.width_max_mm, 3), "width_uncertainty_mm": round(self.width_uncertainty_mm, 3)}

    def to_dict(self) -> dict:
        r = lambda v, d=3: None if v is None else round(float(v), d)  # noqa: E731
        return {"found": self.found, "width_max_mm": r(self.width_max_mm),
                "width_uncertainty_mm": r(self.width_uncertainty_mm), "width_median_mm": r(self.width_median_mm),
                "px_across_at_widest": r(self.px_across, 2), "mm_per_px": r(self.mm_per_px, 4),
                "length_main_mm": r(self.length_main_mm, 1), "length_total_mm": r(self.length_total_mm, 1),
                "n_branches": self.n_branches, "n_pieces": self.n_pieces,
                "widest_point_px": None if self.widest_point_px is None else [r(v, 1) for v in self.widest_point_px],
                "width_from_grey_profile": r(self.width_source_grey, 2),
                "shape": {"angle_to_vertical_deg": r(self.angle_to_vertical_deg, 1),
                          "straightness": r(self.straightness, 2), "coherence": r(self.coherence, 2),
                          "n_loops": self.n_loops, "branch_ratio": r(self.branch_ratio, 2)},
                "flags": self.flags, "matrix_input": self.matrix_input()}


def shape_features(pts, main, tm, length_main, length_total, n_edges, n_pieces) -> dict:
    """Direction and form of the crack, for the pattern rules. `pts` and `main` in crop pixels, `tm` to mm.
    'Vertical' is the image's up-down direction carried onto the wall: it assumes the phone was held upright."""
    pm = tm(main)
    c = main.mean(axis=0)
    up = tm(np.vstack([c, c + (0.0, 1.0)]))
    vdir = (up[1] - up[0]) / np.linalg.norm(up[1] - up[0])
    q = pm - pm.mean(axis=0)
    d = np.linalg.eigh(q.T @ q)[1][:, -1] if len(pm) >= 3 else (pm[-1] - pm[0])
    d = d / max(np.linalg.norm(d), 1e-12)
    angle = float(np.degrees(np.arccos(min(1.0, abs(float(d @ vdir))))))
    chord = float(np.linalg.norm(pm[-1] - pm[0]))
    # local directions along the whole centre line, as doubled angles (a line has no arrow)
    step = max(1, len(pts) // 2000)
    sub = pts[::step]
    tree = cKDTree(pts)
    z = []
    for nb in tree.query_ball_point(sub, r=DIRECTION_RADIUS_PX):
        if len(nb) >= 3:
            qq = pts[nb] - pts[nb].mean(axis=0)
            t = np.linalg.eigh(qq.T @ qq)[1][:, -1]
            z.append(np.exp(2j * np.arctan2(t[1], t[0])))
    return {"angle_to_vertical_deg": angle, "straightness": chord / length_main if length_main > 0 else None,
            "coherence": float(abs(np.mean(z))) if z else None, "n_loops": max(0, int(n_edges - len(pts) + n_pieces)),
            "branch_ratio": length_total / length_main if length_main > 0 else None}


def measure(grey: np.ndarray, mask: np.ndarray, to_mm, rel_scale_uncertainty: float = 0.0,
            reference_centre_mm=None) -> CrackMeasurement:
    """Width and length of the crack in `mask` (full image, True = crack). `to_mm` maps (n, 2) pixel (x, y) to mm."""
    if not mask.any():
        return CrackMeasurement(False, flags=["no crack found in the search box: width could not be measured"])
    ys, xs = np.nonzero(mask)
    margin = 20
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    off = np.array([x0, y0], dtype=float)
    mk = mask[y0:y1, x0:x1]
    g = grey[y0:y1, x0:x1].astype(float)

    def tm(p):
        return to_mm(np.asarray(p, dtype=float) + off)

    edt = ndimage.distance_transform_edt(mk)
    skel = skeletonize(mk)
    half90 = float(np.percentile(edt[skel], 90)) if skel.any() else 1.0
    skel = prune(skel, max(5, int(2 * half90) + 2))
    pts, nbrs, ea, eb = _graph(skel)
    flags = []
    if len(pts) < 3:
        return CrackMeasurement(False, flags=["crack too small to measure"], mask=mask)

    # length
    branches = trace_branches(nbrs)
    total = sum(path_length_mm(pts[br], tm) for br in branches)
    pts_mm = tm(pts)
    w = np.linalg.norm(pts_mm[ea] - pts_mm[eb], axis=1)
    G = sparse.coo_matrix((w, (ea, eb)), shape=(len(pts), len(pts))).tocsr()
    n_pieces, comp = csgraph.connected_components(G, directed=False)
    biggest = np.flatnonzero(comp == np.bincount(comp).argmax())
    d = csgraph.dijkstra(G, directed=False, indices=biggest[0])
    u = biggest[np.argmax(d[biggest])]
    d, pred = csgraph.dijkstra(G, directed=False, indices=u, return_predecessors=True)
    v = biggest[np.argmax(d[biggest])]
    path = [v]
    while path[-1] != u and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    main = pts[path[::-1]]
    length_main = path_length_mm(main, tm)
    if n_pieces > 1:
        flags.append(f"crack found in {n_pieces} pieces: main length is the longest piece only; check the overlay")

    # width at every centre-line point away from ends and junctions
    deg = np.array([len(x) for x in nbrs])
    tree = cKDTree(pts)
    valid = deg == 2
    for kind, radius in ((deg >= 3, 2 * half90 + 2), (deg <= 1, half90 + 2)):
        if kind.any():
            dist, _ = cKDTree(pts[kind]).query(pts)
            valid &= dist > radius
    gs = ndimage.gaussian_filter(g, 0.7)
    mkf = mk.astype(float)
    samples = []  # (index, width px, width mm, edge a, edge b, source)
    for i, nb in zip(np.flatnonzero(valid), tree.query_ball_point(pts[valid], r=DIRECTION_RADIUS_PX)):
        if len(nb) < 3:
            continue
        q = pts[nb] - pts[nb].mean(axis=0)
        tangent = np.linalg.eigh(q.T @ q)[1][:, -1]
        n = np.array([-tangent[1], tangent[0]])
        x, y = pts[i]
        prof = profile_width(gs, mkf, x, y, n[0], n[1], max(float(edt[int(y), int(x)]), 0.5))
        if prof is None:
            continue
        tl, tr, src = prof
        a, b = pts[i] + tl * n, pts[i] + tr * n
        ab = tm(np.vstack([a, b]))
        samples.append((i, tr - tl, float(np.linalg.norm(ab[1] - ab[0])), a, b, src,
                        float(np.linalg.norm(tm(pts[i] + n) - tm(pts[i])))))
    result = CrackMeasurement(True, length_main_mm=length_main, length_total_mm=total, n_branches=len(branches),
                              n_pieces=int(n_pieces), mask=mask, skeleton_px=pts + off, main_path_px=main + off,
                              **shape_features(pts, main, tm, length_main, total, len(ea), n_pieces))
    if len(samples) < 3:
        result.flags = flags + ["too few centre-line points to measure the width: width could not be measured"]
        return result

    sp = np.array([pts[s[0]] for s in samples])
    wpx = np.array([s[1] for s in samples])
    wmm = np.array([s[2] for s in samples])
    near = cKDTree(sp).query_ball_point(sp, r=SMOOTH_RADIUS_PX)
    wmm_s = np.array([np.median(wmm[k]) for k in near])
    wpx_s = np.array([np.median(wpx[k]) for k in near])
    k = int(np.argmax(wmm_s))
    i, _, _, a, b, _, mm_px = samples[k]
    result.width_max_mm = float(wmm_s[k])
    result.width_median_mm = float(np.median(wmm))
    result.px_across = float(wpx_s[k])
    result.mm_per_px = float(np.median([s[6] for s in samples]))
    result.width_uncertainty_mm = WIDTH_UNCERTAINTY_PX * mm_px + result.width_max_mm * rel_scale_uncertainty
    result.widest_point_px = tuple(pts[i] + off)
    result.widest_edges_px = np.vstack([a, b]) + off
    result.width_source_grey = float(np.mean([s[5] == "grey" for s in samples]))

    if result.px_across < MIN_PX_ACROSS:
        flags.append(f"only {result.px_across:.1f} pixels across the crack (at least {MIN_PX_ACROSS:g} needed [46]): "
                     "retake closer; the width is not trusted")
    elif result.px_across < GOOD_PX_ACROSS:
        flags.append(f"{result.px_across:.1f} pixels across the crack: about +-30 % [46]; 6 to 8 recommended")
    if result.width_source_grey < 0.5:
        flags.append("low contrast: most widths come from the mask edges, not the grey profile")
    if reference_centre_mm is not None:
        dist = float(np.linalg.norm(tm(result.widest_point_px - off)[0] - np.asarray(reference_centre_mm)))
        if dist > FAR_FROM_REFERENCE_MM:
            flags.append(f"crack is {dist:.0f} mm from the reference: put the reference next to the crack [46]")
    result.flags = flags
    return result
