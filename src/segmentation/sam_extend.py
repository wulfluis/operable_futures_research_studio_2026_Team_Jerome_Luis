"""SAM 3.1 extends the crack the crack model found (D91). Spec: spec/features/sam_vision.md.

The crack model (U-Net v4) still decides whether there is a crack; SAM 3.1, asked for "crack", traces the faint
branches and continuations the model misses, and only SAM lines connected to the model's crack are added. Alone, SAM
drew cracks on 7 to 20 times more crack-free tiles than the model (src/experiments/sam_vision_eval.py), so it never
decides. Its lines are thicker than the crack: they say where the crack runs; the width is still read from the photo
(src/measure/crack.py)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

ENABLED = True  # switch off to go back to the crack model alone
SAM_SIDE = 1024  # px, the photo's long side for SAM (it reads 644 px inside)
CRACK_CONF = 0.4  # SAM's score from which a crack line counts (illustrative, chosen on the validation tiles)
JOIN_SHARE = 0.02  # a SAM line this close to the model's crack (share of the long side) is connected
CONNECTED_ONLY = False  # D92: once the model found a crack, every SAM crack line counts, not only the connected ones
# (branches of a network often meet at a spalled patch, not at the line the model found). Test tiles: 85.2 % of the
# crack traced against 84.9 % connected-only and 82.0 % for the model; lines off the crack 5.2 % against 2.6 % and
# 1.4 %; missed cracks and false alarms unchanged, since the model still decides.
MIN_SHARE = 0.002  # smaller SAM masks are specks


def extend(model_mask: np.ndarray, sam: np.ndarray, join_share: float = JOIN_SHARE) -> np.ndarray:
    """The model's crack plus SAM's crack pixels in pieces that touch it (the model's crack grown by join_share)."""
    if not model_mask.any() or not sam.any():
        return model_mask
    near = ndimage.binary_dilation(model_mask, iterations=max(1, round(join_share * max(model_mask.shape))))
    lab, _ = ndimage.label(sam, structure=np.ones((3, 3)))
    keep = np.unique(lab[near & sam])
    keep = keep[keep > 0]
    return model_mask | np.isin(lab, keep)


def sam_cracks(photo: Image.Image) -> np.ndarray | None:
    """SAM's crack lines on the photo (full size, True = crack), or None when SAM 3.1 is not on this laptop."""
    from element.parts import sam_segmenter, shared_predictor
    p = shared_predictor()
    if p is None:
        return None
    im = photo.convert("RGB")
    k = SAM_SIDE / max(im.size)
    small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR) if k < 1 else im
    out = sam_segmenter(p, ["crack"])(np.asarray(small))
    m = np.zeros((small.height, small.width), bool)
    for _, mask, score in out:
        if score >= CRACK_CONF and mask.sum() >= MIN_SHARE * mask.size:
            m |= mask
    m = without_edges(m)
    if small.size != im.size:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(im.size, Image.NEAREST)) > 0
    return m


def without_edges(m: np.ndarray, side: int = 512) -> np.ndarray:
    """SAM's lines without the long straight edges the crack model's rule already ignores (D51): a door frame or the
    line between a wall and a ceiling. Checked on a copy reduced to `side` px, as the rule was tuned."""
    if not m.any():
        return m
    from segmentation.decide import edge_pixels
    k = side / max(m.shape)
    small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (max(1, round(m.shape[1] * k)), max(1, round(m.shape[0] * k))), Image.NEAREST)) > 0
    labels, n = ndimage.label(small, structure=np.ones((3, 3)))
    drop = np.zeros_like(small)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        e = edge_pixels(ys, xs, small.shape)
        if e is not None:
            drop[ys[e], xs[e]] = True
    if not drop.any():
        return m
    drop = np.asarray(Image.fromarray(drop.astype(np.uint8) * 255).resize((m.shape[1], m.shape[0]), Image.NEAREST)) > 0
    return m & ~ndimage.binary_dilation(drop, iterations=2)


def extended(photo: Image.Image, model_mask: np.ndarray) -> np.ndarray:
    """The model's crack (full size) extended by SAM, or unchanged when SAM is off, missing or fails."""
    if not ENABLED or not model_mask.any():
        return model_mask
    try:
        sam = sam_cracks(photo)
    except Exception:  # SAM never stops the crack model's answer
        return model_mask
    if sam is None:
        return model_mask
    return extend(model_mask, sam) if CONNECTED_ONLY else model_mask | sam
