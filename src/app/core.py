"""Class demo logic without the web page: consent, the pipeline with bank-card measurement, the live wall, the tunnel
link, the QR. Specs: spec/features/class_demo.md, spec/features/app_measure.md (Luis's measurement, folded in, D46).

Without a bank card in the photo the result is the crack statement alone. With a card, the U-Net crack is also measured
in mm (src/measure/, D41, D42) and the camera tilt is reported. A measurement, never a safety statement.
"""

from __future__ import annotations

import hmac
import re
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
from quality import scene as scene_mod
from segmentation.decide import CRACK, NO_CRACK, PIXEL_THRESHOLD, UNSURE, Decision, decide
from segmentation.predict import draw_overlay, probability, working_image
from triage.element_rule import ElementAnswer
from triage.matrix import BAND_NAMES, ELEMENTS, PATTERN_NAMES, Result as LevelResult, assess, width_band
from triage.pattern_rules import PatternSuggestion, suggest

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
MEASURE_VERSION = "app-measure-0.3 (illustrative thresholds)"
# Position on the element -> the risk matrix's location (D53); not known from the photos yet, so 'unknown'
POSITIONS = {"unknown": "unknown", "joint": "end_zone", "end_zone": "end_zone", "middle": "middle"}
POSITION_NAMES = {"unknown": "not known (counts as an end zone)", "joint": "at a beam-column joint",
                  "end_zone": "near a support or joint (end zone)", "middle": "middle of the element"}
WAITING = "Category: send the photo of the whole element to get one."
AUTO_SOURCE = "automatic: element model, written type rules and the written risk matrix (D55)"
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
    element: ElementAnswer | None = None  # from the element photo (D53); None: no element photo
    triage: dict | None = None  # the matrix's inputs and level, given automatically after the element photo (D55)
    wall_number: int | None = None  # the result's number on the live wall, when shared
    scene: scene_mod.SceneAnswer | None = None  # close-up check of the crack photo (D54)

    @property
    def measured(self) -> bool:
        return self.crack is not None and self.measure_note is None and self.crack.matrix_input() is not None

    def band(self) -> str | None:
        if not self.measured:
            return None
        m = self.crack.matrix_input()
        return BAND_NAMES[width_band(m["width_mm"] + m["width_uncertainty_mm"])]

    def pattern(self, element=None, at_joint=None, spalling=None, rust_stain=False) -> PatternSuggestion | None:
        """Draft rules (D43); never asked of the resident (D25)."""
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
                "element": None if self.element is None else element_dict(self.element),
                "pattern_suggestion": _suggestion_dict(self.suggested_pattern()),
                "triage": self.triage,
                "scene": None if self.scene is None else self.scene.to_dict(),
                "note": "Measurement only, never a safety statement. Thresholds illustrative until calibrated."}

    def suggested_pattern(self) -> PatternSuggestion | None:
        """The written rules' type for the model's element (D43, D53); a suggestion, never a level by itself."""
        from element.predict import matrix_element
        return self.pattern(matrix_element(self.element))


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


def run_measured(photo: Image.Image, model, device: str = "cpu", scene_model=None,
                 card_placed: bool = False) -> MeasuredResult:
    """The original pipeline (quality gate, U-Net, written rule), plus the bank card when one is in the photo.
    With a scene model, a photo that is not a close-up is a Retake before the crack model looks at it (D54).
    The card is searched for only when the sender says one was placed (D56): never a card the app made up."""
    full = ImageOps.exif_transpose(photo).convert("RGB")
    img, (w, h) = working_image(full)
    work = img.crop((0, 0, w, h))
    gate = check(np.asarray(work.convert("L"), dtype=np.float64))  # as the original: the photo, not the padding
    problems = [f for f in gate.failed if f != "scale card not confirmed"]
    if problems:
        reason = f"The photo is {' and '.join(problems)}; please take it again."
        return MeasuredResult("Retake", reason, None, draw_overlay(work, np.zeros((h, w), bool), "Retake", reason))
    scene = scene_mod.check(scene_model, full, device, "crack") if scene_model is not None else None
    if scene is not None and not scene.ok:
        r = MeasuredResult("Retake", scene.reason, None, draw_overlay(work, np.zeros((h, w), bool), "Retake",
                                                                       scene.reason))
        r.scene = scene
        return r

    prob = probability(model, img, device)[:h, :w]
    rgb = np.asarray(full)
    card = card_mod.detect_card(rgb) if card_placed else None
    k = w / full.width  # full photo -> working size
    if card is not None:  # the card's edges are straight dark lines: never read them as the crack
        prob = np.where(reference_mask((h, w), card.corners_px * k, CARD_PAD * max(w, h)), 0.0, prob)
    d = decide(prob, scale=max(w, h) / 416)
    line = f"{d.reason} Confidence {d.confidence:.0%}." if d.confidence else d.reason

    if card is None:  # exactly the original's answer
        if d.statement != NO_CRACK:
            line += (" Bank card not found: lay it flat next to the crack, all four corners in the photo, and make "
                     "sure it stands out from the wall." if card_placed else
                     " Width unknown: add a bank card (or the scale card) next to the crack.")
        r = MeasuredResult(d.statement, line, d.confidence, draw_overlay(work, d.mask, d.statement, line), d)
        r.scene = scene
        return r

    r = MeasuredResult(d.statement, line, d.confidence, work, d, card)
    r.scene = scene
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

    def update(self, number: int, stats: dict) -> bool:
        """New stats for a result still on the wall (after the element photo); False if it is gone."""
        with self._lock:
            for e in self._items:
                if e.number == number:
                    e.stats = stats
                    return True
        return False

    def get(self, number: int) -> WallEntry | None:
        with self._lock:
            return next((e for e in self._items if e.number == number), None)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {s: self._counts[s] for s in STATEMENTS}


def analyse(data: bytes, consent: bool, share: bool, wall: Wall, model, device: str = "cpu",
            gpu: threading.Lock | None = None, scene_model=None, card_placed: bool = False) -> MeasuredResult:
    """One upload, as the original: consent, size and format, then the pipeline. Only pixels are read."""
    if not consent:
        raise ConsentMissing("Tick the consent box first.")
    photo = _open_photo(data)
    with gpu or threading.Lock():
        result = run_measured(photo, model, device, scene_model, card_placed)
    if share:
        result.wall_number = wall.add(result.statement, result.overlay, result.to_dict())
    return result


def _open_photo(data: bytes) -> Image.Image:
    """Size and format check; only the pixels are kept (rule 9)."""
    if len(data) > MAX_BYTES:
        raise PhotoRejected("The photo is larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("JPEG", "PNG", "MPO"):
                raise PhotoRejected("Send a JPEG or PNG photo.")
            img.load()
            return img.copy()
    except (OSError, Image.DecompressionBombError) as e:
        raise PhotoRejected("This file could not be read as a photo.") from e


def _suggestion_dict(s: PatternSuggestion | None) -> dict | None:
    return None if s is None else s.to_dict()


def element_dict(a: ElementAnswer) -> dict:
    from element.predict import matrix_element
    key = matrix_element(a)
    return {"answer": a.element, "matrix_element": key, "name": ELEMENTS.get(key) if key else None,
            "could_be_column": a.could_be_column, "probability": round(a.probability, 3), "reason": a.reason,
            "source": "element model, a suggestion"}


def analyse_element(data: bytes, result: MeasuredResult, wall: Wall, model, device: str = "cpu",
                    gpu: threading.Lock | None = None, scene_model=None) -> ElementAnswer:
    """The second photo, of the whole element: same checks as the crack photo, only pixels read, never saved (rule 9).
    A close-up cannot show the element: with a scene model it is refused with a hint (D54). The answer is stored on
    the result and, when shared, on its wall entry."""
    from element.predict import read_element
    photo = _open_photo(data)
    with gpu or threading.Lock():
        if scene_model is not None:
            s = scene_mod.check(scene_model, photo, device, "element")
            if not s.ok:
                raise PhotoRejected(s.reason)
        result.element = read_element(model, photo, device)
    auto_triage(result, wall)
    return result.element


def level_for(result: MeasuredResult, element: str | None, position: str, pattern: str | None,
              rust_stain: bool = False, confidence: float | None = None) -> LevelResult:
    """The written risk matrix's level for these inputs (rule 5). No width (no bank card, or too few pixels) gives
    Expert needed in the matrix itself; never a "safe" answer. confidence: the crack model's when not given."""
    c = result.crack.matrix_input() if result.measured else None
    conf = confidence if confidence is not None else result.confidence
    conf = conf if conf is not None else float("nan")
    return assess(pattern or "none", element or "unknown", c["width_mm"] if c else float("nan"), conf,
                  result.statement != "Retake", width_uncertainty_mm=c["width_uncertainty_mm"] if c else 0.0,
                  escalators=("rust_stain",) if rust_stain else (), location=POSITIONS[position])


def auto_triage(result: MeasuredResult, wall: Wall | None = None) -> LevelResult | None:
    """The level, given by itself once the element photo is in (D55): the model's element, the written rules' type
    and the written risk matrix. Any doubt gives Expert needed, never a category: an element the model is unsure of,
    a type the rules call unclear, no width. The position is not known, so it counts as an end zone (never lower).
    Stored on the result and on its wall entry; None before the element photo."""
    from element.predict import matrix_element
    if result.element is None or result.statement not in ("Crack found", "Expert needed"):
        return None
    element = matrix_element(result.element)
    s = result.suggested_pattern()
    pattern = s.pattern if s is not None and s.pattern and s.clear else None
    conf = min(result.confidence if result.confidence is not None else 0.0, result.element.probability)
    level = level_for(result, element, "unknown", pattern, confidence=conf)
    why = [result.element.reason] if element is None else []
    if not result.measured:
        why.append("The crack type and width need a bank card next to the crack.")
    elif pattern is None:
        why.append("The crack type is not clear from the photo.")
    result.triage = {"source": AUTO_SOURCE, "element": element,
                     "element_name": ELEMENTS.get(element) if element else None,
                     "element_probability": round(result.element.probability, 3), "position": "unknown",
                     "position_name": POSITION_NAMES["unknown"], "pattern": pattern,
                     "pattern_name": PATTERN_NAMES.get(pattern) if pattern else None,
                     "pattern_suggested": s.pattern if s is not None else None,
                     "confidence": round(conf, 3), "level": level.level, "reason": level.reason,
                     "why": " ".join(w for w in why if w), "engineer_review": level.engineer_review,
                     "next_step": level.next_step}
    if wall is not None and result.wall_number is not None:
        wall.update(result.wall_number, result.to_dict())
    return level


def wall_key_ok(given: str | None, key: str | None) -> bool:
    return bool(given) and bool(key) and hmac.compare_digest(given, key)


def make_qr(url: str, path=None) -> Image.Image:
    img = qrcode.make(url, box_size=12, border=2).get_image().convert("RGB")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img


TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def tunnel_url(line: str) -> str | None:
    """The quick tunnel prints its random https link once in its log; find it."""
    m = TUNNEL_URL.search(line)
    return m.group(0) if m else None
