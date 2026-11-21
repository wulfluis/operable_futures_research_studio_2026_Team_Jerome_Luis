"""Every crack in the photo, not only the longest (D88). Spec: spec/features/all_cracks.md.

The crack model's mask is split into pieces; pieces that nearly touch are one crack (the model often leaves a small gap
in a faint crack), pieces further apart are independent cracks. Each crack is measured on its own; the photo's answer
takes the width at the widest point over ALL cracks (the level must never miss a wider piece, AGENTS.md rule 2) and the
length and shape of the main crack, the one with the longest main line."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from measure.crack import CrackMeasurement

GAP_SHARE = 0.02  # of the photo's long side: closer pieces are one crack (illustrative until calibrated)
MIN_EXTENT_SHARE = 0.02  # of the photo's long side: smaller pieces are specks, left out (illustrative)
MAX_POINTS_PER_PIECE = 4000  # pixels per piece used to measure the distance between pieces
EIGHT = np.ones((3, 3), dtype=bool)
WIDTH_FIELDS = ("width_max_mm", "width_uncertainty_mm", "px_across", "widest_point_px", "widest_edges_px")


def limits(shape: tuple[int, int]) -> tuple[float, float]:
    """(gap, smallest extent) in pixels for a mask of this shape (height, width)."""
    side = float(max(shape))
    return GAP_SHARE * side, MIN_EXTENT_SHARE * side


def group_pieces(mask: np.ndarray, gap_px: float, min_extent_px: float) -> list[np.ndarray]:
    """The cracks in `mask` (True = crack), each as its own mask, the largest extent first. Pieces closer than
    `gap_px` are one crack; pieces with an extent under `min_extent_px` are left out, unless nothing else is there."""
    labels, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return []
    boxes = ndimage.find_objects(labels)
    extent = np.array([np.hypot(b[0].stop - b[0].start, b[1].stop - b[1].start) for b in boxes])
    keep = [i for i in range(n) if extent[i] >= min_extent_px] or [int(np.argmax(extent))]
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    pts = {}
    for i in keep:
        sel = np.flatnonzero(lab == i + 1)
        if len(sel) > MAX_POINTS_PER_PIECE:
            sel = sel[np.linspace(0, len(sel) - 1, MAX_POINTS_PER_PIECE).round().astype(int)]
        pts[i] = np.column_stack([xs[sel], ys[sel]]).astype(float)
    parent = {i: i for i in keep}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_i, a in enumerate(keep):
        ba, tree = boxes[a], None
        for b in keep[a_i + 1:]:
            bb = boxes[b]
            dy = max(0, ba[0].start - bb[0].stop, bb[0].start - ba[0].stop)
            dx = max(0, ba[1].start - bb[1].stop, bb[1].start - ba[1].stop)
            if np.hypot(dx, dy) > gap_px or root(a) == root(b):
                continue  # the boxes are already too far apart, or already one crack
            if tree is None:
                tree = cKDTree(pts[a])
            if float(tree.query(pts[b], distance_upper_bound=gap_px + 1)[0].min()) <= gap_px:
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for i in keep:
        groups.setdefault(root(i), []).append(i)
    masks = [np.isin(labels, [i + 1 for i in members]) for members in groups.values()]
    size = [np.hypot(*(np.ptp(np.nonzero(m), axis=1) + 1)) for m in masks]
    return [masks[k] for k in np.argsort(size)[::-1]]


def combine(parts: list[CrackMeasurement]) -> CrackMeasurement:
    """One answer for the photo from each crack's own measurement: the widest point over all cracks (with that
    crack's uncertainty and pixels across), the main crack's length and shape, every crack kept for drawing."""
    found = [p for p in parts if p.found]
    if not found:
        return parts[0] if parts else CrackMeasurement(False, flags=["no crack found: width could not be measured"])
    if len(found) == 1:
        return replace(found[0], n_cracks=1, cracks=[found[0]])
    main = max(found, key=lambda p: p.length_main_mm or 0.0)
    out = replace(main, n_cracks=len(found), cracks=found)
    out.length_total_mm = float(sum(p.length_total_mm or 0.0 for p in found))
    out.n_branches = int(sum(p.n_branches for p in found))
    out.n_pieces = int(sum(p.n_pieces for p in found))
    out.skeleton_px = np.vstack([p.skeleton_px for p in found if p.skeleton_px is not None])
    out.branch_paths_px = [b for p in found for b in (p.branch_paths_px or [])]
    measured = [p for p in found if p.width_max_mm is not None and p.profile_px is not None]
    flags = list(main.flags)
    if measured:
        wide = max(measured, key=lambda p: p.width_max_mm)
        for k in WIDTH_FIELDS:
            setattr(out, k, getattr(wide, k))
        out.profile_px = np.vstack([p.profile_px for p in measured])
        out.profile_mm = np.concatenate([p.profile_mm for p in measured])
        out.profile_edges_px = (np.concatenate([p.profile_edges_px for p in measured])
                                if all(p.profile_edges_px is not None for p in measured) else None)
        out.width_median_mm = float(np.median(out.profile_mm))
        out.mm_per_px = float(np.median([p.mm_per_px for p in measured if p.mm_per_px]))
        out.width_source_grey = float(np.mean([p.width_source_grey or 0.0 for p in measured]))
        flags += [f for f in wide.flags if f not in flags]  # the flags of the width the level uses
    out.flags = flags
    return out


def measure_all(mask: np.ndarray, measure_one: Callable[[np.ndarray], CrackMeasurement],
                gap_px: float | None = None, min_extent_px: float | None = None) -> CrackMeasurement:
    """Group `mask` into cracks, measure each with `measure_one(crack_mask)`, and combine them."""
    g, e = limits(mask.shape)
    groups = group_pieces(mask, g if gap_px is None else gap_px, e if min_extent_px is None else min_extent_px)
    if not groups:
        return measure_one(mask)
    return combine([measure_one(m) for m in groups])
