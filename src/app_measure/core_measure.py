"""Copy of the class demo with bank-card measurement: logic without the web page. Spec: spec/features/app_measure.md.

The original app (`src/app/`, `src/segmentation/`) is not changed: its functions are imported and used as they are.
Without a bank card in the photo the result is the original's, pixel for pixel. With a card, the U-Net crack is
also measured in mm (src/measure/, D41, D42) and the camera tilt is reported. A measurement, never a safety statement.
"""

from __future__ import annotations

import hmac
import io
import threading
from collections import Counter, deque
from dataclasses import dataclass, field, replace

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from measure import card as card_mod
from measure.analyse import longest_piece, reference_mask
from measure.crack import BACKGROUND_PX, CrackMeasurement, measure, segment
from measure.reference import from_rectangle
from quality.gate import check
from segmentation.decide import CRACK, NO_CRACK, PIXEL_THRESHOLD, UNSURE, Decision, decide
from segmentation.predict import draw_overlay, probability, working_image
from triage.matrix import BAND_NAMES, width_band
from triage.pattern_rules import PatternSuggestion, suggest

# The same limits and texts as the original (src/app/core.py), copied because that file needs Python 3.13
MAX_BYTES = 10 * 1024 * 1024
WALL_SIZE = 40
WALL_THUMB = 480
WALL_LARGE = 1400  # px, long side of the copy shown on a result's own page (added in the copy)
STATEMENTS = ("Crack found", "No crack found", "Expert needed", "Retake")
NOT_A_SAFETY_CHECK = "This is not a safety check. An engineer confirms every crack that matters."
Image.MAX_IMAGE_PIXELS = 60_000_000

CARD_PAD = 0.015  # share of the photo's long side around the card that is not crack (its edges look like one)
UNET_GROW = 3  # px the U-Net's crack area is widened by before the classical edges are taken inside it
LINE_FIELDS = ("length_main_mm", "length_total_mm", "n_branches", "n_pieces", "angle_to_vertical_deg", "straightness",
               "coherence", "n_loops", "branch_ratio", "skeleton_px", "main_path_px")  # taken from the U-Net line
MEASURE_VERSION = "app-measure-0.2 (illustrative thresholds)"
WHITE, BLACK = (255, 255, 255), (11, 11, 11)


class ConsentMissing(Exception):
    pass


class PhotoRejected(Exception):
    pass


@dataclass
class MeasuredResult:
    statement: str  # Retake, Crack found, No crack found or Expert needed: the original's statements only
    reason: str
    confidence: float | None
    overlay: Image.Image
    decision: Decision | None = None
    card: card_mod.CardDetection | None = None  # None: no bank card found, the result is the original's
    crack: CrackMeasurement | None = None
    measure_note: str | None = None  # why there is no width, when a card is there
    warnings: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.crack is not None and self.measure_note is None and self.crack.matrix_input() is not None

    def band(self) -> str | None:
        if not self.measured:
            return None
        m = self.crack.matrix_input()
        return BAND_NAMES[width_band(m["width_mm"] + m["width_uncertainty_mm"])]

    def pattern(self, element=None, at_joint=None, spalling=None, rust_stain=False) -> PatternSuggestion | None:
        """Draft rules (D43); the answers come from a field worker, never from the resident (D25)."""
        if self.crack is None or not self.crack.found:
            return None
        return suggest(self.crack.to_dict()["shape"], element, at_joint=at_joint, spalling=spalling,
                       rust_stain=rust_stain)

    def to_dict(self) -> dict:
        return {"version": MEASURE_VERSION, "statement": self.statement, "reason": self.reason,
                "confidence": None if self.confidence is None else round(self.confidence, 3),
                "card": None if self.card is None else self.card.to_dict(),
                "measured": self.measured, "measure_note": self.measure_note, "warnings": self.warnings,
                "crack": None if self.crack is None else self.crack.to_dict(),
                "width_band_of_upper_end": self.band(),
                "note": "Measurement only, never a safety statement. Thresholds illustrative until calibrated."}


def _full_mask(prob: np.ndarray, keep: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The U-Net map and the kept groups, scaled from the working size to the full photo."""
    p = np.asarray(Image.fromarray(prob.astype(np.float32), "F").resize(size, Image.BILINEAR))
    k = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 0
    return (p > PIXEL_THRESHOLD) & k


def _draw_card_and_widest(overlay: Image.Image, scale: float, card, crack) -> Image.Image:
    """The card outline in white, and a white ring at the widest point, on the photo part of the overlay."""
    out = overlay.copy()
    draw = ImageDraw.Draw(out)
    lw = max(2, out.width // 300)
    pts = [tuple(map(float, p * scale)) for p in card.corners_px]
    draw.line(pts + pts[:1], fill=BLACK, width=lw + 2, joint="curve")
    draw.line(pts + pts[:1], fill=WHITE, width=lw, joint="curve")
    if crack is not None and crack.widest_point_px is not None:
        x, y = (float(v) * scale for v in crack.widest_point_px)
        r = max(8, out.width // 60)
        draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], outline=BLACK, width=lw + 2)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=WHITE, width=lw)
    return out


def run_measured(photo: Image.Image, model, device: str = "cpu") -> MeasuredResult:
    """The original pipeline (quality gate, U-Net, written rule), plus the bank card when one is in the photo."""
    full = ImageOps.exif_transpose(photo).convert("RGB")
    img, (w, h) = working_image(full)
    work = img.crop((0, 0, w, h))
    gate = check(np.asarray(work.convert("L"), dtype=np.float64))  # as the original: the photo, not the padding
    problems = [f for f in gate.failed if f != "scale card not confirmed"]
    if problems:
        reason = f"The photo is {' and '.join(problems)}; please take it again."
        return MeasuredResult("Retake", reason, None, draw_overlay(work, np.zeros((h, w), bool), "Retake", reason))

    prob = probability(model, img, device)[:h, :w]
    rgb = np.asarray(full)
    card = card_mod.detect_card(rgb)
    k = w / full.width  # full photo -> working size
    if card is not None:  # the card's edges are straight dark lines: never read them as the crack
        prob = np.where(reference_mask((h, w), card.corners_px * k, CARD_PAD * max(w, h)), 0.0, prob)
    d = decide(prob, scale=max(w, h) / 416)
    line = f"{d.reason} Confidence {d.confidence:.0%}." if d.confidence else d.reason

    if card is None:  # exactly the original's answer
        if d.statement != NO_CRACK:
            line += " Width unknown: add the scale card."
        return MeasuredResult(d.statement, line, d.confidence, draw_overlay(work, d.mask, d.statement, line), d)

    r = MeasuredResult(d.statement, line, d.confidence, work, d, card)
    if card.status == "retake":
        r.measure_note = card.message
    elif card.status == "tilted":
        r.warnings.append(card.message)
    if d.statement in (CRACK, UNSURE) and r.measure_note is None:
        grey = np.asarray(full.convert("L"), dtype=float)
        exclude = reference_mask(grey.shape, card.corners_px, CARD_PAD * max(full.size))
        unet = _full_mask(prob, d.mask, full.size) & ~exclude
        scale = from_rectangle(card.corners_px, card_mod.CARD_MM, "bank card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm",
                               long_first=card.long_first)
        centre = scale.to_mm(np.mean(card.corners_px, axis=0))[0]
        # Length and shape: the U-Net's crack line, which stays continuous on a faint crack.
        # Width: the U-Net's area is wider than the crack, so inside it the classical finder (dark line against its
        # background, src/measure/crack.py) gives the crack's edges; it may find a faint crack only in pieces, so
        # every piece along the line counts for the width, none of them for the length.
        line_mask = longest_piece(unet)
        along = measure(grey, line_mask, scale.to_mm, scale.rel_uncertainty, centre)
        edges = segment(grey, None, exclude, BACKGROUND_PX, None) & ndimage.binary_dilation(line_mask,
                                                                                            iterations=UNET_GROW)
        wide = measure(grey, edges, scale.to_mm, scale.rel_uncertainty, centre)
        if along.found and wide.found and wide.width_max_mm is not None:
            r.crack = replace(wide, **{k: getattr(along, k) for k in LINE_FIELDS})
        else:
            r.crack = along
            if along.found:
                r.warnings.append("Width from the U-Net's crack area, which is wider than the crack: the width may "
                                  "be too large.")
        for f in r.crack.flags:
            if "retake closer" in f and r.measure_note is None:
                r.measure_note = f
            elif "retake closer" not in f:
                r.warnings.append(f)
        if r.crack.matrix_input() is None and r.measure_note is None:
            r.measure_note = "The width could not be read: go closer so the crack is at least 3 pixels wide."
    elif d.statement == NO_CRACK:
        r.measure_note = "Bank card found; there is no crack to measure."

    if r.measured:
        c = r.crack
        line = (f"{d.reason} Confidence {d.confidence:.0%}. Widest {c.width_max_mm:.2f} ± "
                f"{c.width_uncertainty_mm:.2f} mm, length {c.length_main_mm:.0f} mm (bank card).")
    elif d.statement != NO_CRACK:
        line += " Width not measured: " + r.measure_note[0].lower() + r.measure_note[1:]
    r.reason = line
    r.overlay = _draw_card_and_widest(draw_overlay(work, d.mask, d.statement, line), k, card,
                                      r.crack if r.measured else None)
    return r


@dataclass
class WallEntry:
    number: int
    statement: str
    thumb: bytes  # JPEG, long side WALL_THUMB
    large: bytes  # JPEG, long side WALL_LARGE, for the result's own page
    stats: dict  # MeasuredResult.to_dict()


@dataclass
class Wall:
    """Shared results for the projector, in memory only; gone when the app stops (as the original)."""
    size: int = WALL_SIZE
    _items: deque = field(default_factory=deque)
    _counts: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next: int = 1

    def add(self, statement: str, image: Image.Image, stats: dict | None = None) -> int:
        """Keeps a small JPEG for the wall, a larger one and the stats for the result's own page (added in the
        copy). Returns the result's number."""
        def jpeg(side: int) -> bytes:
            img = image.convert("RGB")
            img.thumbnail((side, side))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            return buf.getvalue()

        thumb, large = jpeg(WALL_THUMB), jpeg(WALL_LARGE)
        with self._lock:
            n = self._next
            self._next += 1
            self._items.appendleft(WallEntry(n, statement, thumb, large, stats or {}))
            while len(self._items) > self.size:
                self._items.pop()
            self._counts[statement] += 1
        return n

    def items(self) -> list[tuple[str, bytes]]:
        """(statement, small JPEG), newest first: as the original."""
        with self._lock:
            return [(e.statement, e.thumb) for e in self._items]

    def entries(self) -> list[WallEntry]:
        with self._lock:
            return list(self._items)

    def get(self, number: int) -> WallEntry | None:
        with self._lock:
            return next((e for e in self._items if e.number == number), None)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {s: self._counts[s] for s in STATEMENTS}


def analyse(data: bytes, consent: bool, share: bool, wall: Wall, model, device: str = "cpu",
            gpu: threading.Lock | None = None) -> MeasuredResult:
    """One upload, as the original: consent, size and format, then the pipeline. Only pixels are read."""
    if not consent:
        raise ConsentMissing("Tick the consent box first.")
    if len(data) > MAX_BYTES:
        raise PhotoRejected("The photo is larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("JPEG", "PNG", "MPO"):
                raise PhotoRejected("Send a JPEG or PNG photo.")
            img.load()
            photo = img.copy()
    except (OSError, Image.DecompressionBombError) as e:
        raise PhotoRejected("This file could not be read as a photo.") from e
    with gpu or threading.Lock():
        result = run_measured(photo, model, device)
    if share:
        wall.add(result.statement, result.overlay, result.to_dict())
    return result


def wall_key_ok(given: str | None, key: str | None) -> bool:
    return bool(given) and bool(key) and hmac.compare_digest(given, key)


def make_qr(url: str, path=None) -> Image.Image:
    img = qrcode.make(url, box_size=12, border=2).get_image().convert("RGB")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img
