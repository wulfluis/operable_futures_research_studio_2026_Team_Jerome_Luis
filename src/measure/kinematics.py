"""Crack kinematics from one photo: opening and sliding (D99). Spec: spec/features/crack_kinematics.md.

The principle is the one of Pantoja-Rosero et al. 2022 [48], used in E. Corres Sojo's thesis [46] (ch. 5.2): the two
faces of a crack were one surface before it opened, so the bumps of one face match the hollows of the other. The
translation that moves one face back onto the other is the crack's displacement at that point: across the crack it is
the opening, along the crack the sliding. Own implementation (the EPFL code is GPL-3.0, this repository MIT): at
stations along the crack, each face read as a profile along the crack in a window; the shift along the crack that
lines one face's bumps up with the other's hollows is the sliding, the gap left between them the opening
(translation only, an assumption over a short window). Shown only: nothing here changes a level (rule 5). Thresholds illustrative."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

STEP_PX = 12.0  # a station every this many pixels along the crack
WINDOW_WIDTHS = 6.0  # the window's half-length, in local crack widths ...
MIN_WINDOW_PX = 30.0  # ... and at least this many pixels
MIN_WIDTH_PX = 3.0  # thinner: the faces are not measurable
MIN_ROUGHNESS_PX = 0.4  # a face smoother than this (std of its profile) has no bumps to match: sliding not readable
MIN_CORRELATION = 0.6  # the two faces must match at least this well for the sliding to be read (illustrative)
MAX_SLIDE_WIDTHS = 2.0  # sliding is searched up to this many crack widths
MOSTLY_OPENING = 0.3  # sliding / opening under this: "mostly opening" (illustrative)
MOSTLY_SLIDING = 1.0  # over this: "mostly sliding" (illustrative)


@dataclass
class Station:
    point_px: tuple  # (x, y) on the crack's centre line
    normal: tuple  # unit vector across the crack
    tangent: tuple  # unit vector along the crack
    vector_px: tuple  # the displacement of one face onto the other, photo pixels
    opening: float  # across the crack, in the unit
    sliding: float | None  # along the crack (signed, + along the tangent), None when not readable
    error_px: float  # the match's remaining mean distance


@dataclass
class Kinematics:
    unit: str = "px"
    stations: list = field(default_factory=list)
    opening_median: float | None = None
    opening_max: float | None = None
    sliding_median: float | None = None  # of the absolute sliding, readable stations only
    sliding_max: float | None = None
    sliding_share: float | None = None  # median |sliding| / median opening
    readable_share: float = 0.0  # share of stations where sliding could be read
    reading: str = "not read"
    reason: str = ""

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 3)  # noqa: E731
        return {"unit": self.unit, "opening_median": r(self.opening_median), "opening_max": r(self.opening_max),
                "sliding_median": r(self.sliding_median), "sliding_max": r(self.sliding_max),
                "sliding_share": r(self.sliding_share), "readable_share": round(self.readable_share, 2),
                "reading": self.reading, "reason": self.reason, "n_stations": len(self.stations),
                "method": "two crack faces matched (Pantoja-Rosero et al. 2022 [48], as in Corres Sojo 2024 [46]); "
                          "one photo: the displacement since the crack formed; shown only"}


def face_profiles(mask: np.ndarray, p: np.ndarray, t: np.ndarray, half: float, reach: float):
    """The two faces around point p, as profiles along the crack: for each step u along t (1 px), the crack is crossed
    along the normal n from -reach to +reach; of the stretches inside the crack, the one nearest the line through p
    gives the face on the + side (a, its far end) and on the - side (b, minus its near end). NaN where none is found."""
    n = np.array([-t[1], t[0]])
    u = np.arange(-half, half + 1.0)
    v = np.arange(-reach, reach + 0.01, 0.25)
    pts = p[None, None, :] + u[:, None, None] * t + v[None, :, None] * n
    inside = ndimage.map_coordinates(mask.astype(np.float32), [pts[..., 1], pts[..., 0]], order=1,
                                     mode="constant") > 0.5
    a = np.full(len(u), np.nan)
    b = np.full(len(u), np.nan)
    for i, row in enumerate(inside):
        if not row.any() or row[0] or row[-1]:  # no crack, or it runs past the reach: no face found
            continue
        edges_ = np.flatnonzero(np.diff(row.astype(np.int8)))
        starts, ends = edges_[::2] + 1, edges_[1::2]
        mid = (v[starts] + v[ends]) / 2
        j = int(np.abs(mid).argmin())
        a[i], b[i] = v[ends[j]] + 0.125, -(v[starts[j]] - 0.125)
    return u, a, b


def shift_between(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float | None, float]:
    """The shift s (px, along the crack) that best lines face a up with face b (a(u) ~ o - b(u - s)): the bumps of one
    face are the hollows of the other, so a and -b are compared. Returns (s, correlation); s None when no clear peak."""
    best, scores = None, []
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            x, y = a[s:], -b[:len(b) - s]
        else:
            x, y = a[:s], -b[-s:]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 0.6 * len(a) or np.std(x[ok]) < 1e-6 or np.std(y[ok]) < 1e-6:
            scores.append(-1.0)
            continue
        scores.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    sc = np.array(scores)
    i = int(sc.argmax())
    if sc[i] < MIN_CORRELATION:
        return None, float(sc[i])
    frac = 0.0
    if 0 < i < len(sc) - 1:  # sub-pixel: a parabola through the peak
        den = sc[i - 1] - 2 * sc[i] + sc[i + 1]
        frac = 0.5 * (sc[i - 1] - sc[i + 1]) / den if den < 0 else 0.0
    return float(i - max_shift + frac), float(sc[i])


def edge_profiles(edges_px: np.ndarray, p: np.ndarray, t: np.ndarray, half: float, reach: float):
    """The two faces around p from the crack's measured edges (src/measure/crack.py: for each centre-line point the
    two edge points found in the photo's grey values, sub-pixel), binned 1 px along the crack. As face_profiles."""
    n = np.array([-t[1], t[0]])
    u = np.arange(-half, half + 1.0)
    a = np.full(len(u), np.nan)
    b = np.full(len(u), np.nan)
    e = np.asarray(edges_px, float).reshape(-1, 2)
    rel = e - p
    along, across = rel @ t, rel @ n
    ok = (np.abs(along) <= half + 0.5) & (np.abs(across) <= reach)
    idx = np.round(along[ok] + half).astype(int)
    for side, arr in ((across[ok] > 0, a), (across[ok] < 0, b)):
        vals = np.abs(across[ok][side])
        for j in np.unique(idx[side]):
            if 0 <= j < len(u):
                arr[j] = float(np.median(vals[idx[side] == j]))
    return u, a, b


def station(mask: np.ndarray | None, p: np.ndarray, t: np.ndarray, width_px: float,
            edges_px: np.ndarray | None = None) -> Station | None:
    """The displacement at one point p of the centre line, with t the unit direction of the crack there. The faces
    come from the measured edges when given (the app), else from the mask (the published toy cracks)."""
    n = np.array([-t[1], t[0]])
    half = max(MIN_WINDOW_PX, WINDOW_WIDTHS * width_px)
    reach = max(4.0 * width_px, 14.0)
    u, a, b = (edge_profiles(edges_px, p, t, half, reach) if edges_px is not None
               else face_profiles(mask, p, t, half, reach))
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < 0.6 * len(u):
        return None
    if np.nanstd(a) < MIN_ROUGHNESS_PX or np.nanstd(b) < MIN_ROUGHNESS_PX:  # smooth faces: nothing to match
        s, corr = None, 0.0
    else:
        s, corr = shift_between(a, b, int(min(half / 2, MAX_SLIDE_WIDTHS * width_px + 2)))
    if s is None:
        opening = float(np.nanmedian(a + b))
        slide = None
    else:
        k = int(round(s))
        aa, bb = (a[k:], b[:len(b) - k]) if k >= 0 else (a[:k], b[-k:])
        opening = float(np.nanmedian(aa + bb))
        slide = -s  # face a moved by -s along t relative to face b (a's bump at u came from b's at u - s ... reversed)
    d = opening * n + (slide or 0.0) * t
    return Station(tuple(p), tuple(n), tuple(t), tuple(d), opening, slide, 1.0 - corr)


def from_path(mask: np.ndarray | None, path_px: np.ndarray, widths_px: np.ndarray | float | None = None,
              mm_per_px: float | None = None, edges_px: np.ndarray | None = None) -> Kinematics:
    """Opening and sliding along one traced crack. path_px: its centre line (x, y); widths_px: the width at each
    path point (or one value); mm_per_px: the scale from the bank card, None for pixels; edges_px: the measured
    edges, (n, 2, 2), used instead of the mask when given."""
    out = Kinematics(unit="mm" if mm_per_px else "px")
    path = np.asarray(path_px, float)
    if (edges_px is None and (mask is None or not mask.any())) or len(path) < 3:
        out.reason = "No crack line to read."
        return out
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2 * STEP_PX:
        out.reason = "The crack is too short to read its faces."
        return out
    if widths_px is None:
        dist = ndimage.distance_transform_edt(mask)
        widths_px = 2 * dist[np.clip(path[:, 1].round().astype(int), 0, mask.shape[0] - 1),
                             np.clip(path[:, 0].round().astype(int), 0, mask.shape[1] - 1)]
    w = np.broadcast_to(np.asarray(widths_px, float), (len(path),))
    k = mm_per_px or 1.0
    for si in np.arange(STEP_PX / 2, s[-1] - STEP_PX / 2, STEP_PX):
        i = int(np.searchsorted(s, si))
        lo, hi = int(np.searchsorted(s, si - STEP_PX)), min(len(path) - 1, int(np.searchsorted(s, si + STEP_PX)))
        t = path[hi] - path[max(0, lo)]
        if np.hypot(*t) < 1e-6 or w[i] < MIN_WIDTH_PX:
            continue
        st = station(mask, path[i], t / np.hypot(*t), float(w[i]), edges_px)
        if st is None or st.opening <= 0:
            continue
        st.opening *= k
        st.sliding = None if st.sliding is None else st.sliding * k
        out.stations.append(st)
    if not out.stations:
        out.reason = ("The crack is too thin or its faces too smooth to match: opening and sliding not read "
                      f"(faces need about {MIN_WIDTH_PX:.0f} px across).")
        return out
    op = np.array([x.opening for x in out.stations])
    sl = np.array([abs(x.sliding) for x in out.stations if x.sliding is not None])
    out.opening_median, out.opening_max = float(np.median(op)), float(op.max())
    out.readable_share = len(sl) / len(out.stations)
    if len(sl) >= max(2, 0.3 * len(out.stations)):
        out.sliding_median, out.sliding_max = float(np.median(sl)), float(sl.max())
        out.sliding_share = out.sliding_median / max(out.opening_median, 1e-9)
        out.reading = ("mostly opening" if out.sliding_share < MOSTLY_OPENING else
                       "opening and sliding" if out.sliding_share <= MOSTLY_SLIDING else "mostly sliding")
        out.reason = (f"Read at {len(out.stations)} points along the crack: opening {out.opening_median:.2f} "
                      f"{out.unit}, sliding {out.sliding_median:.2f} {out.unit} (median): {out.reading}.")
    else:
        out.reading = "opening only (sliding not readable)"
        out.reason = (f"Opening {out.opening_median:.2f} {out.unit} (median of {len(out.stations)} points); the faces "
                      "are too smooth to read the sliding.")
    return out


def for_crack(c, unit_mm: bool) -> Kinematics | None:
    """Opening and sliding of one measured crack (a CrackMeasurement): its main line, its measured edges and widths.
    In mm when the width was measured with a bank card (unit_mm), else in the photo's pixels."""
    path = getattr(c, "main_path_px", None)
    edges = getattr(c, "profile_edges_px", None)
    prof = getattr(c, "profile_px", None)
    if path is None or edges is None or prof is None or len(prof) < 3:
        return None
    from scipy.spatial import cKDTree
    edges = np.asarray(edges, float)
    wpx = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    _, j = cKDTree(np.asarray(prof, float)).query(np.asarray(path, float))
    k = from_path(None, path, wpx[j], float(c.mm_per_px) if unit_mm and c.mm_per_px else None, edges.reshape(-1, 2))
    return k
