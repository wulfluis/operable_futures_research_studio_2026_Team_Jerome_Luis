"""Quality gate: written rules for whether a photo is usable. Spec: spec/features/quality_gate.md.

Thresholds are illustrative until tuned on our own phone photos.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import blur_effect

# "Blurred" only when both measures agree (D33): each alone fails some sharp walls (D12).
MIN_SHARPNESS = 100.0  # Laplacian variance; alone it fails sharp smooth walls with a thin crack
MAX_BLUR = 0.45  # blur effect [56], 0 sharp to 1 blurred; alone it fails sharp walls with soft, mottled plaster
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0
WORK_SIZE = 512  # long side in px for measuring; sharpness depends on resolution (decision D8)


@dataclass(frozen=True)
class GateResult:
    image_ok: bool
    sharpness: float
    brightness: float
    failed: list[str] = field(default_factory=list)
    blur: float = 0.0


def to_grey(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float64)


def sharpness(grey: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior pixels."""
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1])
    return float(lap.var())


def blur(grey: np.ndarray) -> float:
    """No-reference blur measure [56]: how little the image changes when it is blurred again. It compares the image
    with itself, not with a fixed amount of texture, so a smooth wall with one sharp crack is not called blurred.
    A flat image with no detail at all counts as fully blurred."""
    value = blur_effect(np.clip(grey, 0, 255) / 255.0)
    return float(value) if np.isfinite(value) else 1.0


def to_work_size(grey: np.ndarray) -> np.ndarray:
    """Scale down so the long side is WORK_SIZE; smaller images are returned unchanged."""
    h, w = grey.shape
    if max(h, w) <= WORK_SIZE:
        return grey
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    img.thumbnail((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def check(grey: np.ndarray, scale_visible: bool | None = None) -> GateResult:
    """Every check must pass; a missing scale-card result counts as a fail (D24)."""
    work = to_work_size(grey)
    s, b, fuzz = sharpness(work), float(work.mean()), blur(work)
    failed = []
    if s < MIN_SHARPNESS and fuzz > MAX_BLUR:
        failed.append("blurred")
    if b < MIN_BRIGHTNESS:
        failed.append("too dark")
    if b > MAX_BRIGHTNESS:
        failed.append("too bright")
    if scale_visible is not True:
        failed.append("scale card not confirmed")
    return GateResult(not failed, s, b, failed, fuzz)
