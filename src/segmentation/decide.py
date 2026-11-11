"""Turn the U-Net's crack-probability map into a statement. Written rule, not learned. Spec: spec/features/segmentation.md."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

PIXEL_THRESHOLD = 0.5  # a pixel is crack above this probability (illustrative)
CRACK_PIXELS = 150  # largest crack group at least this big -> crack found (at 416 px; illustrative until calibrated)
NO_CRACK_PIXELS = 50  # largest group below this -> no crack found; in between -> Expert needed (illustrative)

# A straight line across the photo is a corner or an edge, not a crack (D51; all illustrative until calibrated)
EDGE_BORDER = 0.03  # share of the long side: a group this close to a side of the photo reaches it
EDGE_MIN_LENGTH = 0.4  # share of the short side: shorter straight parts are never ignored
EDGE_BAND = 0.015  # share of the straight part's length: pixels this close to its line belong to it ...
EDGE_BAND_PX = 4  # ... and never less than this many pixels (the line has a width)
EDGE_SHARE = 0.5  # share of the group's pixels on its straight part: less, and the group is a crack with a straight bit
EDGE_REST_STRAIGHT = 0.85  # a leftover piece with this share on its own straight line is part of the corner

# A crack is a line: a thick, round patch is not counted as one (D54; illustrative until calibrated)
BLOB_MAX_ELONGATION = 3.0  # length / width at or under this: a patch, not a line (a round patch is about 1)

CRACK, NO_CRACK, UNSURE = "Crack found", "No crack found", "Expert needed"
REASONS = {
    CRACK: "The model marks a continuous crack line in the photo.",
    NO_CRACK: "The model marks no crack line; this is not a safety check.",
    UNSURE: "The model marks only a short, uncertain line, so an expert should look.",
}
BLOB_REASON = ("The model marks only a dark patch, not a line: it could be a stain, an object or spalling, so an "
               "expert should look.")
BLOB_NOTE = "Dark patches that are not line-shaped were not counted as cracks."
EDGE_NOTE = ("A long straight line running across the photo was read as a corner or edge and ignored; a crack "
             "running exactly along an edge is not checked.")


@dataclass
class Decision:
    statement: str
    confidence: float  # mean probability of the largest group (0 when there is none)
    largest_group: int  # pixels
    mask: np.ndarray  # bool, the largest group and every group above NO_CRACK_PIXELS (ignored edges left out)
    edges_ignored: int = 0  # groups read as a corner or edge (D51)
    patches: int = 0  # groups too thick and round to be a crack (D54)
    patch_only: bool = False  # Expert needed because only a large patch is marked (it could be spalling)

    @property
    def reason(self) -> str:
        main = BLOB_REASON if self.patch_only else REASONS[self.statement]
        notes = [EDGE_NOTE] if self.edges_ignored else []
        if self.patches and not self.patch_only:
            notes.append(BLOB_NOTE)
        return " ".join([main] + notes)


def elongation(group: np.ndarray) -> float:
    """Length / width of one group of pixels (a crop holding only it). Width from the distance transform (95th
    percentile, so one wide spot does not make a long crack a patch); length = area / width. A line of 200 x 4 px gives
    about 50, a round patch about 1.3."""
    edt = ndimage.distance_transform_edt(np.pad(group, 1))[1:-1, 1:-1][group]
    width = max(1.0, 2 * float(np.percentile(edt, 95)))
    return float(group.sum()) / width / width


def straight_part(ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The pixels on the main straight line through these pixels (True per pixel) and that line's length in pixels.
    First the direction and place with the most pixels in one narrow band (a Hough transform, 1 degree steps), then
    fit a line to those pixels, keep the pixels near it, fit again: a bend or a branch does not pull it off. None: no line."""
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    if len(pts) < 3:
        return None
    width = 2 * EDGE_BAND_PX  # one bin of the search
    best, inside = -1, None
    for a in np.deg2rad(np.arange(180)):
        d = pts[:, 1] * np.cos(a) - pts[:, 0] * np.sin(a)  # distance across a line at angle a
        bins = np.floor(d / width).astype(int) - int(np.floor(d.min() / width))
        counts = np.append(np.bincount(bins), 0)
        k = int(counts.argmax())
        below = counts[k - 1] if k else 0
        pair = counts[k] + max(below, counts[k + 1])  # the fullest bin and its fuller neighbour: a line on a
        if pair > best:                                # border between two bins is not split
            best = pair
            border = k + 1 if counts[k + 1] >= below else k  # the border between the two bins, in bins
            inside = np.abs(d - (border + np.floor(d.min() / width)) * width) <= width
    for _ in range(10):  # until the line stops moving
        if inside.sum() < 3:
            return None
        centre = pts[inside].mean(axis=0)
        direction = np.linalg.svd(pts[inside] - centre, full_matrices=False)[2]  # rows: along the line, across it
        along, across = (pts - centre) @ direction[0], (pts - centre) @ direction[1]
        new = np.abs(across) <= max(EDGE_BAND_PX, EDGE_BAND * float(np.ptp(along[inside])))
        if np.array_equal(new, inside):
            break
        inside = new
    if inside.sum() < 3:
        return None
    return inside, float(np.ptp(along[inside]))


def reaches_side(ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    m = EDGE_BORDER * max(h, w)
    return bool(ys.min() <= m or ys.max() >= h - 1 - m or xs.min() <= m or xs.max() >= w - 1 - m)


def edge_pixels(ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Of one group of crack pixels: True for those that are a corner or an edge (D51); None when it holds none.
    An edge is a long straight line, most of the group, reaching a side of the photo. What is left of the group is
    split into pieces: a straight piece (the short corner where two walls meet) goes with the edge; a piece that
    wanders (a crack leaving the corner) stays a crack."""
    fit = straight_part(ys, xs)
    if fit is None:
        return None
    inside, length = fit
    if (inside.mean() < EDGE_SHARE or length < EDGE_MIN_LENGTH * min(shape)
            or not reaches_side(ys[inside], xs[inside], shape)):
        return None
    out = inside.copy()
    rest = np.zeros(shape, bool)
    rest[ys[~inside], xs[~inside]] = True
    pieces, n = ndimage.label(rest, structure=np.ones((3, 3)))
    piece_of = pieces[ys, xs]
    for k in range(1, n + 1):
        sel = piece_of == k
        f = straight_part(ys[sel], xs[sel])
        if f is None or f[0].mean() >= EDGE_REST_STRAIGHT:  # too small to fit, or straight: part of the corner
            out |= sel
    return out


def decide(prob: np.ndarray, scale: float = 1.0) -> Decision:
    """scale = long side of the map / 416: a crack line's pixel count grows with its length in pixels."""
    crack_pixels, no_crack_pixels = CRACK_PIXELS * scale, NO_CRACK_PIXELS * scale
    crack = prob > PIXEL_THRESHOLD
    labels, n = ndimage.label(crack, structure=np.ones((3, 3)))
    if n == 0:
        return Decision(NO_CRACK, 0.0, 0, np.zeros(prob.shape, bool))
    sizes = ndimage.sum_labels(np.ones(prob.shape), labels, index=np.arange(1, n + 1))
    edges = 0
    for i in np.flatnonzero(sizes >= no_crack_pixels):  # only groups that could count are checked
        ys, xs = np.nonzero(labels == i + 1)
        edge = edge_pixels(ys, xs, prob.shape)
        if edge is not None:
            crack[ys[edge], xs[edge]] = False  # ignored: not counted, not drawn, not measured
            edges += 1
    if edges:  # what is left is judged as before
        labels, n = ndimage.label(crack, structure=np.ones((3, 3)))
        if n == 0:
            return Decision(NO_CRACK, 0.0, 0, np.zeros(prob.shape, bool), edges)
        sizes = ndimage.sum_labels(np.ones(prob.shape), labels, index=np.arange(1, n + 1))
    boxes = ndimage.find_objects(labels)
    patches, big_patch = [], None
    for i in np.flatnonzero(sizes >= no_crack_pixels):
        if elongation(labels[boxes[i]] == i + 1) <= BLOB_MAX_ELONGATION:
            patches.append(i)
            if sizes[i] >= crack_pixels and (big_patch is None or sizes[i] > sizes[big_patch]):
                big_patch = i
    if patches:
        patch_sizes = sizes.copy()
        sizes[patches] = 0  # not counted as a crack
        if sizes.max() < no_crack_pixels and big_patch is not None:  # only a large patch: never "No crack found"
            m = labels == big_patch + 1
            return Decision(UNSURE, float(prob[m].mean()), int(patch_sizes[big_patch]), m, edges, len(patches), True)
        if sizes.max() == 0:
            return Decision(NO_CRACK, 0.0, 0, np.zeros(prob.shape, bool), edges, len(patches))
    largest = int(sizes.max())
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= min(largest, no_crack_pixels)))
    confidence = float(prob[labels == 1 + int(sizes.argmax())].mean())
    if largest >= crack_pixels:
        statement = CRACK
    elif largest < no_crack_pixels:
        statement = NO_CRACK
    else:
        statement = UNSURE
    return Decision(statement, confidence, largest, keep, edges, len(patches))


def pixel_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Counts for the crack class; add them up over photos, then call rates()."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    return dict(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()), fn=int((~pred & truth).sum()))


def rates(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 0.0)
