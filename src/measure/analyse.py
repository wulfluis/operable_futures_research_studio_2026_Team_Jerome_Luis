"""One photo in, one measurement out: quality gate, bank card, tilt, crack width and length.

Spec: spec/features/measurement.md (D41, D42). Used by the app (`app.py`) and the command line (`measure_crack.py`).
It returns a measurement and the reasons to retake, never a triage level and never a safety statement: the level
needs the pattern and the element, and comes from src/triage/matrix.py.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from measure import card as card_mod
from measure.crack import BACKGROUND_PX, CrackMeasurement, measure, segment
from measure.reference import from_rectangle
from quality.gate import check
from triage.matrix import BAND_NAMES, width_band
from triage.pattern_rules import PatternSuggestion, suggest

VERSION = "measure-0.3 (illustrative thresholds)"


@dataclass
class Analysis:
    retake: list[str] = field(default_factory=list)  # any entry means: retake the photo
    warnings: list[str] = field(default_factory=list)
    card: card_mod.CardDetection | None = None
    crack: CrackMeasurement | None = None
    roi: tuple = (0, 0, 0, 0)
    sharpness: float | None = None
    brightness: float | None = None
    pattern: PatternSuggestion | None = None

    @property
    def outcome(self) -> str:
        if self.retake:
            return "Retake"
        if self.crack is None or self.crack.matrix_input() is None:
            return "Width not measured"
        return "Measured"

    def matrix_input(self) -> dict | None:
        """Width inputs for triage.matrix.assess(); None unless the photo is usable and the width trusted."""
        return None if self.retake or self.crack is None else self.crack.matrix_input()

    def band(self) -> str | None:
        m = self.matrix_input()
        return None if m is None else BAND_NAMES[width_band(m["width_mm"] + m["width_uncertainty_mm"])]

    def to_dict(self) -> dict:
        return {"version": VERSION, "outcome": self.outcome, "retake_reasons": self.retake,
                "warnings": self.warnings, "card": None if self.card is None else self.card.to_dict(),
                "roi_px": [round(float(v), 1) for v in self.roi],
                "quality_gate": {"sharpness": None if self.sharpness is None else round(self.sharpness, 1),
                                 "brightness": None if self.brightness is None else round(self.brightness, 1)},
                "crack": None if self.crack is None else self.crack.to_dict(),
                "matrix_input": self.matrix_input(), "width_band_of_upper_end": self.band(),
                "pattern_suggestion": None if self.pattern is None else self.pattern.to_dict(),
                "note": "Measurement only, never a safety statement. The level needs the crack pattern and the "
                        "element and comes from the written risk matrix. Thresholds illustrative until calibrated."}


def reference_mask(shape, corners_px, pad_px) -> np.ndarray:
    """The card plus a band around it: its edges must not be taken for a crack."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(map(float, p)) for p in corners_px], fill=1)
    return ndimage.distance_transform_edt(np.asarray(img) == 0) <= pad_px


def longest_piece(mask: np.ndarray) -> np.ndarray:
    """Keep only the piece with the largest extent (the crack, not stains around it)."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n <= 1:
        return mask
    ext = [np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in ndimage.find_objects(labels)]
    return labels == (int(np.argmax(ext)) + 1)


def analyse(rgb: np.ndarray, roi=None, card_corners=None, mask: np.ndarray | None = None, longest_only=True,
            background_px: int = BACKGROUND_PX, threshold: float | None = None, element: str | None = None,
            at_joint: bool | None = None, spalling: bool | None = None, rust_stain: bool = False) -> Analysis:
    """rgb: H x W x 3 uint8. roi: search box (x0, y0, x1, y1), default the whole photo. card_corners: 4 corners
    given by hand instead of the automatic card search. mask: a crack mask instead of the classical finder.
    element, at_joint, spalling, rust_stain: field-worker answers for the pattern suggestion (never the resident's)."""
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    grey = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float)
    h, w = grey.shape
    a = Analysis(roi=tuple(roi) if roi is not None else (0, 0, w, h))

    a.card = card_mod.from_corners(card_corners, rgb.shape) if card_corners is not None else card_mod.detect_card(rgb)
    gate = check(grey, scale_visible=a.card is not None)
    a.sharpness, a.brightness = gate.sharpness, gate.brightness
    for f in gate.failed:
        if f == "scale card not confirmed":
            a.retake.append("Bank card not found: put it flat on the wall next to the crack, all four corners "
                            "visible, nothing covering its edges.")
        else:
            a.retake.append(f"Photo {f} (quality gate; blur limit not yet tuned, D12).")
    if a.card is None:
        return a
    if a.card.status == "retake":
        a.retake.append(a.card.message)
    elif a.card.status == "tilted":
        a.warnings.append(a.card.message)

    scale = from_rectangle(a.card.corners_px, card_mod.CARD_MM, "bank card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm",
                           long_first=a.card.long_first)
    if mask is None:
        mask = segment(grey, a.roi, reference_mask(grey.shape, a.card.corners_px, background_px), background_px,
                       threshold)
    else:
        x0, y0, x1, y1 = (int(round(v)) for v in a.roi)
        keep = np.zeros_like(mask, dtype=bool)
        keep[max(0, y0):y1, max(0, x0):x1] = True
        mask = mask.astype(bool) & keep
    if longest_only:
        mask = longest_piece(mask)
    centre = scale.to_mm(np.mean(a.card.corners_px, axis=0))[0]
    a.crack = measure(grey, mask, scale.to_mm, scale.rel_uncertainty, centre)
    for f in a.crack.flags:
        (a.retake if "retake closer" in f else a.warnings).append(f)
    shape = a.crack.to_dict()["shape"] if a.crack.found else None
    a.pattern = suggest(shape, element, at_joint=at_joint, spalling=spalling, rust_stain=rust_stain)
    return a
