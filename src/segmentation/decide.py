"""Turn the U-Net's crack-probability map into a statement. Written rule, not learned. Spec: spec/features/segmentation.md."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

PIXEL_THRESHOLD = 0.5  # a pixel is crack above this probability (illustrative)
CRACK_PIXELS = 150  # largest crack group at least this big -> crack found (at 416 px; illustrative until calibrated)
NO_CRACK_PIXELS = 50  # largest group below this -> no crack found; in between -> Expert needed (illustrative)

CRACK, NO_CRACK, UNSURE = "Crack found", "No crack found", "Expert needed"
REASONS = {
    CRACK: "The model marks a continuous crack line in the photo.",
    NO_CRACK: "The model marks no crack line; this is not a safety check.",
    UNSURE: "The model marks only a short, uncertain line, so an expert should look.",
}


@dataclass
class Decision:
    statement: str
    confidence: float  # mean probability of the largest group (0 when there is none)
    largest_group: int  # pixels
    mask: np.ndarray  # bool, the largest group and every group above NO_CRACK_PIXELS

    @property
    def reason(self) -> str:
        return REASONS[self.statement]


def decide(prob: np.ndarray, scale: float = 1.0) -> Decision:
    """scale = long side of the map / 416: a crack line's pixel count grows with its length in pixels."""
    crack_pixels, no_crack_pixels = CRACK_PIXELS * scale, NO_CRACK_PIXELS * scale
    labels, n = ndimage.label(prob > PIXEL_THRESHOLD, structure=np.ones((3, 3)))
    if n == 0:
        return Decision(NO_CRACK, 0.0, 0, np.zeros(prob.shape, bool))
    sizes = ndimage.sum_labels(np.ones(prob.shape), labels, index=np.arange(1, n + 1))
    largest = int(sizes.max())
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= min(largest, no_crack_pixels)))
    confidence = float(prob[labels == 1 + int(sizes.argmax())].mean())
    if largest >= crack_pixels:
        statement = CRACK
    elif largest < no_crack_pixels:
        statement = NO_CRACK
    else:
        statement = UNSURE
    return Decision(statement, confidence, largest, keep)


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
