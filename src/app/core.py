"""Class demo logic without the web page: consent, the pipeline with bank-card measurement, the live wall, the tunnel
link, the QR. Specs: spec/features/class_demo.md, spec/features/app_measure.md (Luis's measurement, folded in, D46).

Without a bank card in the photo the result is the crack statement alone. With a card, the U-Net crack is also measured
in mm (src/measure/, D41, D42) and the camera tilt is reported. A measurement, never a safety statement.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import io
import threading
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from measure import card as card_mod
from measure import cracks as cracks_mod
from measure.analyse import reference_mask
from measure.crack import BACKGROUND_PX, CrackMeasurement, measure, segment
from measure.reference import from_rectangle
from quality.gate import check
from quality import scene as scene_mod
from segmentation.decide import CRACK, LOW_THRESHOLD, NO_CRACK, PIXEL_THRESHOLD, UNSURE, Decision, decide
from segmentation.predict import draw_overlay, probability, working_image
from triage.element_rule import ElementAnswer
from triage.matrix import BAND_NAMES, ELEMENTS, PATTERN_NAMES, Result as LevelResult, assess, width_band
from triage.pattern_rules import PatternSuggestion, suggest

MAX_BYTES = 10 * 1024 * 1024
WALL_SIZE = 200  # results kept in memory; raised from 40 so new test cases never push out older ones (D79)
WALL_THUMB = 480
WALL_LARGE = 1400  # px, long side of the copy shown on a result's own page (added in the copy)
STATEMENTS = ("Crack found", "No crack found", "Expert needed", "Retake")
NOT_A_SAFETY_CHECK = "This is not a safety check. An engineer confirms every crack that matters."
Image.MAX_IMAGE_PIXELS = 60_000_000

CARD_PAD = 0.015  # share of the photo's long side around the card that is not crack (its edges look like one)
UNET_GROW = 3  # px the U-Net's crack area is widened by before the classical edges are taken inside it
LINE_FIELDS = ("length_main_mm", "length_total_mm", "n_branches", "n_pieces", "angle_to_vertical_deg", "straightness",
               "coherence", "n_loops", "branch_ratio", "skeleton_px", "main_path_px",
               "branch_paths_px")  # taken from the U-Net line
MEASURE_VERSION = "app-measure-0.3 (illustrative thresholds)"
# Position on the element -> the risk matrix's location (D53); not known from the photos yet, so 'unknown'
POSITIONS = {"unknown": "unknown", "joint": "end_zone", "end_zone": "end_zone", "middle": "middle"}
POSITION_NAMES = {"unknown": "not known (counts as an end zone)", "joint": "at a beam-column joint",
                  "end_zone": "near a support or joint (end zone)", "middle": "in the middle of the element"}
WAITING = "Category: send the photo of the whole element to get one."
# The wall sorts by risk, highest first (D78). Expert needed sits by Category 3: both go to an engineer (rule 4).
RISK_TABS = ("Category 3", "Expert needed", "Category 2", "Category 1")
WAITING_SHORT = "Waiting for photo 2"


def risk_group(stats: dict) -> str | None:
    """The wall tab a shared result belongs to (D78): its level from the written risk matrix. A crack without a
    level yet (no photo of the element) or a model unsure is Expert needed: doubt goes to a person (rule 2), never to
    a lower tab. A retake, or a photo without a crack, has no risk level: not on the risk tabs."""
    level = (stats.get("triage") or {}).get("level")
    if level in RISK_TABS:
        return level
    if level is None and stats.get("statement") in ("Crack found", "Expert needed"):
        return "Expert needed"
    return None
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
    geometry: dict | None = None  # the crack on the wall plane in mm, for the 3D model (D57); with a card only
    test: bool = False  # a simulated resident (test inbox, D59): marked TEST everywhere
    on_element: object | None = None  # photo 2: the crack's place and extent on the element (D61)
    spalling: bool | None = None  # photo 2: spalling or exposed bars seen (D61); None: not known
    spalling_p: float | None = None
    estimate: dict | None = None  # TEST cases only: width and length estimated from photo 2 (D62), never measured
    pixel_shape: dict | None = None  # crack shape read in the photo's pixels, without a bank card (D65)
    location: dict | None = None  # where on the element, from the two photos linked (D66); None: no element photo
    location_preview: Image.Image | None = field(default=None, repr=False)  # element photo with the link; not shared
    link_grey: np.ndarray | None = field(default=None, repr=False)  # the crack photo, small and grey, in memory only
    link_line_px: np.ndarray | None = field(default=None, repr=False)  # its crack line in link_grey pixels
    link_card: tuple | None = field(default=None, repr=False)  # the card: corners in link_grey pixels, its look (D67)
    link_card_size: float | None = None  # the card's long side / the crack photo's long side
    link_crack_element: np.ndarray | None = field(default=None, repr=False)  # crack line in the element photo (D66)
    anomaly: dict | None = field(default=None, repr=False)  # the autoencoder's heatmap and other defects (D89), shown only
    link_map: tuple | None = field(default=None, repr=False)  # the link itself (locate.Location.link_map, D75)
    link_network_px: list | None = field(default=None, repr=False)  # every branch of photo 1's crack, link_grey px (D75)
    link_width_px: tuple | None = field(default=None, repr=False)  # photo 1's crack width max, median (link_grey px)
    parts: dict | None = None  # which part the crack lies on, from SAM 3.1 (D70); None: no SAM or no element photo
    parts_preview: Image.Image | None = field(default=None, repr=False)  # element photo with SAM's masks; not shared
    views: dict = field(default_factory=dict, repr=False)  # name -> picture: heatmaps, enhanced view (D81)
    parts_things: list | None = field(default=None, repr=False)  # SAM's masks, in memory only (D74)
    photo: Image.Image | None = field(default=None, repr=False)  # photo 1 upright, for the crack viewer (D83); memory only
    pixel_crack: CrackMeasurement | None = field(default=None, repr=False)  # the crack in pixels, no card (D83)
    kinematics: list | None = field(default=None, repr=False)  # opening and sliding per crack (D99), shown only
    growth: dict | None = None  # the change since an earlier photo of the same crack (D99), shown only
    exposed_steel: dict | None = None  # possible exposed steel on the close-up (D107); None: not checked

    @property
    def measured(self) -> bool:
        return self.crack is not None and self.measure_note is None and self.crack.matrix_input() is not None

    def band(self) -> str | None:
        if not self.measured:
            return None
        m = self.crack.matrix_input()
        return BAND_NAMES[width_band(m["width_mm"] + m["width_uncertainty_mm"])]

    def pattern(self, element=None, at_joint=None, spalling=None, rust_stain=False) -> PatternSuggestion | None:
        """Draft rules (D43); never asked of the resident (D25). The crack's shape from photo 1 when measured there,
        else read in photo 1's pixels without a card (D65), else from photo 2 (D62: the same crack on the element)."""
        if self.crack is not None and self.crack.found:
            shape = self.crack.to_dict()["shape"]
        elif self.pixel_shape is not None:
            shape = self.pixel_shape  # no card: angles and ratios read in the photo (D65)
        elif self.on_element is not None and getattr(self.on_element, "shape", None):
            shape = self.on_element.shape
        else:
            return None
        angle, source = axis_angle(self.location, element)
        if angle is not None:  # the element's own edges (D66), else the room's true vertical (D69)
            shape = {**shape, "angle_to_vertical_deg": round(angle, 1), "angle_source": source}
        return suggest(shape, element, at_joint=at_joint, spalling=spalling, rust_stain=rust_stain)

    def to_dict(self) -> dict:
        return {"version": MEASURE_VERSION, "test": self.test, "statement": self.statement, "reason": self.reason,
                "confidence": None if self.confidence is None else round(self.confidence, 3),
                "card": None if self.card is None else self.card.to_dict(),
                "measured": self.measured, "measure_note": self.measure_note, "warnings": self.warnings,
                "crack": None if self.crack is None else self.crack.to_dict(),
                "shape_in_pixels": self.pixel_shape,
                "width_band_of_upper_end": self.band(),
                "element": None if self.element is None else element_dict(self.element),
                "pattern_suggestion": _suggestion_dict(self.suggested_pattern()),
                "triage": self.triage,
                "location": self.location,
                "parts": self.parts,
                "scene": None if self.scene is None else self.scene.to_dict(),
                "geometry": self.geometry if self.measured else None,
                "growth": self.growth,
                "exposed_steel": None if self.exposed_steel is None else {
                    k: self.exposed_steel.get(k) for k in ("seen", "score", "prompt")},
                "kinematics": None if not self.kinematics else [None if k is None else k.to_dict() for k in self.kinematics],
                "estimate": self.estimate,
                "photo2": None if self.on_element is None and self.spalling is None else {
                    "crack_on_element": None if self.on_element is None else self.on_element.to_dict(),
                    "spalling_seen": self.spalling,
                    "spalling_probability": None if self.spalling_p is None else round(self.spalling_p, 3)},
                "note": "Measurement only, never a safety statement. Thresholds illustrative until calibrated."}

    def suggested_pattern(self) -> PatternSuggestion | None:
        """The written rules' type for the model's element (D43, D53); a suggestion, never a level by itself."""
        from element.predict import matrix_element
        steel = bool((self.exposed_steel or {}).get("seen"))  # D107: exposed steel on the close-up
        return self.pattern(matrix_element(self.element), spalling=self.spalling or steel or None)


GEOMETRY_MAX_POINTS = 200  # the crack line sent to the 3D model (D57)
STATION_MM = 50.0  # a section of the crack every this many mm along it, in the detailed drawing (D81)


def wall_geometry(crack: CrackMeasurement, to_mm, card_corners_px, kinematics: list | None = None) -> dict | None:
    """The crack on the wall plane, in mm, for the 3D model (D57): x to the right and y up as the phone saw it (phone
    upright), origin at the card's centre. Built from the card's scale (to_mm); None when there is no crack line."""
    if crack is None or not crack.found or crack.main_path_px is None or len(crack.main_path_px) < 2:
        return None
    card_px = np.asarray(card_corners_px, dtype=float).reshape(4, 2)
    c_px = card_px.mean(axis=0)
    o = to_mm(card_px).mean(axis=0)  # the card's true centre on the wall (the photo's centre is off under tilt)
    base = to_mm(c_px)[0]
    ex = to_mm(c_px + (1.0, 0.0))[0] - base  # image right, on the wall
    up = to_mm(c_px + (0.0, -1.0))[0] - base  # image up (y grows downward in the photo)
    ex = ex / np.linalg.norm(ex)
    ey = up - np.dot(up, ex) * ex
    ey = ey / np.linalg.norm(ey)

    def frame(px):
        q = to_mm(np.asarray(px, dtype=float).reshape(-1, 2)) - o
        return np.column_stack([q @ ex, q @ ey])

    path = np.asarray(crack.main_path_px, dtype=float).reshape(-1, 2)
    if len(path) > GEOMETRY_MAX_POINTS:
        path = path[np.linspace(0, len(path) - 1, GEOMETRY_MAX_POINTS).round().astype(int)]
    rnd = lambda a: np.round(a, 2).tolist()  # noqa: E731
    extra = {}
    if crack.profile_px is not None and len(crack.profile_px) >= 3:  # the width all along the crack (D81)
        from scipy.spatial import cKDTree
        tree = cKDTree(crack.profile_px)
        near_px = max(6.0, 3 * (crack.px_across or 2))  # a path point takes the width measured within this distance
        d, k = tree.query(path)
        extra["width_along_mm"] = [None if dd > near_px else round(float(crack.profile_mm[kk]), 3)
                                   for dd, kk in zip(d, k)]
        pm = frame(path)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pm, axis=0), axis=1))])
        stations = []
        for i, at in enumerate(np.arange(STATION_MM, s[-1] - STATION_MM / 4, STATION_MM)):
            j = int(np.searchsorted(s, at).clip(0, len(path) - 1))
            dd, kk = tree.query(path[j])
            if dd > near_px:
                continue  # no width measured there (an end, a junction or a gap)
            a, b = max(0, j - 2), min(len(pm) - 1, j + 2)
            t = pm[b] - pm[a]
            t = t / max(np.linalg.norm(t), 1e-9)
            stations.append({"name": chr(65 + len(stations)) if len(stations) < 26 else f"S{len(stations) + 1}",
                             "s_mm": round(float(at), 1), "point_mm": rnd(pm[j]),
                             "width_mm": round(float(crack.profile_mm[kk]), 3),
                             "edges_mm": rnd(frame(crack.profile_edges_px[kk])),
                             "across": rnd(np.array([-t[1], t[0]]))})  # tension acted about along this (D81)
        extra["stations"] = stations
    k = next((x for x in (kinematics or []) if x is not None and x.stations), None)
    if k is not None:  # how the crack moved (D99), for the arrows in 3D: point, opening and sliding vectors, mm
        mv = []
        for st_ in k.stations:
            p0 = np.asarray(st_.point_px, float)
            n_, t_, d_ = np.asarray(st_.normal), np.asarray(st_.tangent), np.asarray(st_.vector_px, float)
            o_px, s_px = float(d_ @ n_) * n_, (None if st_.sliding is None else float(d_ @ t_) * t_)
            q = frame(np.vstack([p0, p0 + o_px, p0 + (s_px if s_px is not None else 0 * t_)]))
            mv.append({"point_mm": rnd(q[0]), "opening_mm": rnd(q[1] - q[0]),
                       "sliding_mm": None if s_px is None else rnd(q[2] - q[0])})
        extra["movement"] = {"reading": k.reading, "opening_median_mm": None if k.opening_median is None else
                             round(k.opening_median, 3), "sliding_median_mm": None if k.sliding_median is None else
                             round(k.sliding_median, 3), "stations": mv,
                             "method": "two crack faces matched ([48], as in [46]); shown only"}
    return {**extra, "frame": "wall plane, mm; x right and y up as the phone saw it (phone upright); origin at the card's centre",
            "crack_path_mm": rnd(frame(path)),
            "widest_point_mm": None if crack.widest_point_px is None else rnd(frame(crack.widest_point_px)[0]),
            "card_corners_mm": rnd(frame(card_px)),
            "width_max_mm": None if crack.width_max_mm is None else round(float(crack.width_max_mm), 3),
            "width_median_mm": None if crack.width_median_mm is None else round(float(crack.width_median_mm), 3),
            "length_main_mm": None if crack.length_main_mm is None else round(float(crack.length_main_mm), 1)}


def _full_mask(prob: np.ndarray, keep: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The U-Net map and the kept groups, scaled from the working size to the full photo. Since D91 the kept groups
    include fainter pixels joined to a confident crack (hysteresis), so the low threshold applies here too."""
    p = np.asarray(Image.fromarray(prob.astype(np.float32), "F").resize(size, Image.BILINEAR))
    k = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 0
    return (p > LOW_THRESHOLD) & k


def crack_line_mask(full: Image.Image, prob: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Where the crack runs, full size: the crack model's crack, extended by SAM 3.1's connected lines (D91)."""
    from segmentation.sam_extend import extended
    return extended(full, _full_mask(prob, keep, full.size))


def pixel_line(full: Image.Image, prob: np.ndarray, keep: np.ndarray) -> CrackMeasurement | None:
    """The U-Net's crack line measured in the photo's pixels (no millimetres), with the same code as with a card."""
    line = crack_line_mask(full, prob, keep)
    if not line.any():
        return None
    grey = np.asarray(full.convert("L"), dtype=float)
    m = cracks_mod.measure_all(line, lambda g: measure(grey, g, lambda p: np.asarray(p, dtype=float)))  # D88
    return m if m.found else None


def pixel_shape(m: CrackMeasurement | None) -> dict | None:
    """The crack's shape without a bank card (D65). Angles and ratios only; the angle is the photo's, not the wall's."""
    if m is None:
        return None
    shape = m.to_dict()["shape"]
    if any(shape[k] is None for k in ("angle_to_vertical_deg", "straightness", "coherence", "branch_ratio")):
        return None
    return {**shape, "source": "read in the photo's pixels, no bank card: phone tilt unknown"}


def axis_angle(location: dict | None, element: str | None) -> tuple[float | None, str | None]:
    """The angle the type rules read 'to vertical' (D43): against a column's or beam's own edges in the element
    photo (D66), else against the room's true vertical there (D69); (None, None) when neither was read."""
    if not location:
        return None, None
    if element in ("column", "beam") and location.get("crack_to_axis_deg") is not None:
        a = float(location["crack_to_axis_deg"])
        return (a if element == "column" else 90.0 - a), "against the element's edges in the element photo"
    if location.get("crack_to_vertical_deg") is not None:
        return float(location["crack_to_vertical_deg"]), "against the room's vertical lines in the element photo"
    return None, None


def _keep_for_link(r: MeasuredResult, full: Image.Image, line_px: np.ndarray | None, mask: np.ndarray | None = None,
                   width_px: tuple | None = None, paths: list | None = None) -> None:
    """A small grey copy of the crack photo and its crack line, kept in memory to link the element photo (D66); also
    every branch of the crack (mask: the crack model's, working size) and its width in the photo's pixels (D75)."""
    from element.locate import card_patch, card_view, to_link_size
    from element.photo2 import MIN_BRANCH_SHARE, trace_network
    r.link_grey, k = to_link_size(np.asarray(full.convert("L")))
    r.link_line_px = None if line_px is None else np.asarray(line_px, dtype=float) * k
    paths = [np.asarray(p, float) for p in (paths or []) if p is not None and len(p) >= 2]
    if paths:  # every branch of every traced crack, as measured (D88, D92): the element photo shows them all
        r.link_network_px = [p * k for p in paths]
    elif mask is not None and mask.any():
        to_full = full.width / mask.shape[1]  # working size -> full photo (padding is at the right and bottom only)
        r.link_network_px = [b * to_full * k for b in trace_network(mask, MIN_BRANCH_SHARE * max(mask.shape))]
    if width_px is not None and width_px[0] is not None:
        r.link_width_px = (float(width_px[0]) * k, float(width_px[1] or width_px[0]) * k)
    if r.card is not None:  # the bridge to the element photo, where the crack itself may be too thin to see (D67)
        patch, q = card_patch(np.asarray(full), r.card.corners_px)
        r.link_card = (q * k, patch, card_view(np.asarray(full), q))
        r.link_card_size = float(np.linalg.norm(q[1] - q[0])) / max(full.size)


def _draw_card_and_widest(overlay: Image.Image, scale: float, card, crack, full: Image.Image | None = None
                          ) -> Image.Image:
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
    if crack is not None and crack.found and crack.width_max_mm is not None:  # the numbers on the photo (D80)
        from app.annotate import draw_measurements
        info = out.info
        h = info.get("photo_height") or out.height
        photo = draw_measurements(out.crop((0, 0, out.width, h)), scale, crack, card.corners_px, full)
        out.paste(photo, (0, 0))
        out.info = info
    return out


STEEL_NO_CRACK = ("No crack line found, but possible exposed steel on the photo (SAM 3.1): the concrete cover may be "
                  "gone. An expert should look.")
STEEL_NOTE = ("Possible exposed steel on the photo (SAM 3.1): the traced crack lines may follow the steel and the "
              "broken concrete, so their widths are not used.")


def _check_steel(full: Image.Image) -> dict | None:
    """Exposed steel on the close-up (D107); None when SAM 3.1 is not on this laptop or fails (never stops the answer)."""
    try:
        from element.steel import find
        return find(full)
    except Exception:  # noqa: BLE001
        return None


def _steel_defects(r: MeasuredResult) -> None:
    """The steel's outline as a defect layer in the crack viewer (D107), beside the autoencoder's (D89)."""
    st = r.exposed_steel or {}
    if st.get("seen") and st.get("defects"):
        if r.anomaly is None:
            r.anomaly = {"score": None, "defects": [], "heat_png": None, "version": None}
        r.anomaly["defects"] = st["defects"] + r.anomaly["defects"]


def _add_kinematics(r: MeasuredResult) -> None:
    """Opening and sliding of every traced crack (D99): the method of [48] used in Corres Sojo's thesis [46], read on
    the crack's measured edges. In mm with a bank card that measured the width, else in pixels. Shown only."""
    if (r.exposed_steel or {}).get("seen"):  # D107: the lines may follow the steel, not a crack's two faces
        return
    try:
        from measure.kinematics import for_crack
        traced = r.crack if (r.crack is not None and r.crack.found) else r.pixel_crack
        if traced is None:
            return
        parts = [p for p in (getattr(traced, "cracks", None) or [traced]) if p is not None]
        ks = [for_crack(p, r.measured) for p in parts]  # one per crack, in the viewer's order (None: not read)
        r.kinematics = ks if any(k is not None for k in ks) else None
    except Exception as e:  # noqa: BLE001  shown only: never stops the answer
        r.warnings.append(f"Opening and sliding not read ({type(e).__name__}).")


def _add_anomaly(r: MeasuredResult, full: Image.Image, prob: np.ndarray, d: Decision, card) -> None:
    """What looks different from sound concrete (D89): heatmap, score and other defects, separate from the traced
    cracks and the card. Shown only, never a level; skipped quietly when the autoencoder is not installed."""
    try:
        from anomaly.predict import analyse_photo
        crack = _full_mask(prob, d.mask, full.size) if d.mask is not None and d.mask.any() else None
        exclude = (reference_mask((full.height, full.width), card.corners_px, CARD_PAD * max(full.size))
                   if card is not None else None)
        r.anomaly = analyse_photo(full, crack, exclude)
    except Exception as e:  # never stops the crack answer
        r.warnings.append(f"Unusual-surface check not run ({type(e).__name__}).")
    try:  # possible spalling or missing plaster, from SAM 3.1 and written rules (D93): unverified, shown only
        from anomaly import spalling_sam
        traced = getattr(r, "crack", None) or getattr(r, "pixel_crack", None)
        parts = [p.mask for p in (getattr(traced, "cracks", None) or [traced]) if p is not None and p.mask is not None]
        lines = np.logical_or.reduce(parts) if parts and all(p.shape == parts[0].shape for p in parts) else crack
        exclude = (reference_mask((full.height, full.width), card.corners_px, CARD_PAD * max(full.size))
                   if card is not None else None)
        spall = spalling_sam.find(full, lines, exclude)
        if spall:
            if r.anomaly is None:
                r.anomaly = {"score": None, "defects": [], "heat_png": None, "version": None}
            r.anomaly["defects"] = spall + r.anomaly["defects"]
    except Exception as e:
        r.warnings.append(f"Spalling check not run ({type(e).__name__}).")


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
    rgb = np.asarray(full)
    card = card_mod.detect_card(rgb) if card_placed else None  # first: a found card shows the photo is a wall (D68)
    scene = scene_mod.check(scene_model, full, device, "crack", card is not None) if scene_model is not None else None
    if scene is not None and not scene.ok:
        r = MeasuredResult("Retake", scene.reason, None, draw_overlay(work, np.zeros((h, w), bool), "Retake",
                                                                       scene.reason))
        r.scene = scene
        return r

    prob = probability(model, img, device)[:h, :w]
    k = w / full.width  # full photo -> working size
    if card is not None:  # the card's edges are straight dark lines: never read them as the crack
        prob = np.where(reference_mask((h, w), card.corners_px * k, CARD_PAD * max(w, h)), 0.0, prob)
    d = decide(prob, scale=max(w, h) / 416)
    line = f"{d.reason} Confidence {d.confidence:.0%}." if d.confidence else d.reason
    steel = _check_steel(full)  # D107: on every usable close-up, a crack line found or not
    seen_steel = bool((steel or {}).get("seen"))
    if seen_steel and d.statement == NO_CRACK:  # bare steel is never "no crack found": a person decides (rule 2)
        line = STEEL_NO_CRACK
    elif seen_steel and card is None:  # with a card, the note comes as "Width not measured: ..."
        line += " " + STEEL_NOTE

    if card is None:  # exactly the original's answer
        if d.statement != NO_CRACK:
            line += (" Bank card not found: lay it flat next to the crack, all four corners in the photo, and make "
                     "sure it stands out from the wall." if card_placed else
                     " Width unknown: add a bank card (or the scale card) next to the crack.")
        r = MeasuredResult(d.statement, line, d.confidence, draw_overlay(work, d.mask, d.statement, line), d)
        r.scene = scene
        r.exposed_steel = steel
        if seen_steel and d.statement == NO_CRACK:
            r.statement = UNSURE
            r.overlay = draw_overlay(work, d.mask, UNSURE, line)
        if d.statement in (CRACK, UNSURE):  # the type needs no scale; the answer above stays the original's
            m = pixel_line(full, prob, d.mask)
            r.pixel_shape = pixel_shape(m)
            r.photo, r.pixel_crack = full, m
            _keep_for_link(r, full, None if m is None else m.main_path_px, d.mask,
                           paths=None if m is None else m.branch_paths_px, width_px=
                           None if m is None else (m.width_max_mm, m.width_median_mm))  # pixels: no card
            r.views = _views(work, prob, k, None, False)
            _add_kinematics(r)
        _add_anomaly(r, full, prob, d, None)
        _steel_defects(r)
        return r

    r = MeasuredResult(d.statement, line, d.confidence, work, d, card)
    r.scene = scene
    r.exposed_steel = steel
    if seen_steel and d.statement == NO_CRACK:
        r.statement = UNSURE
    if seen_steel:  # D107: no width from lines that may follow the steel
        r.measure_note = STEEL_NOTE
    elif card.status == "retake":
        r.measure_note = card.message
    elif card.status == "tilted":
        r.warnings.append(card.message)
    if d.statement in (CRACK, UNSURE) and r.measure_note is None:
        grey = np.asarray(full.convert("L"), dtype=float)
        exclude = reference_mask(grey.shape, card.corners_px, CARD_PAD * max(full.size))
        unet = crack_line_mask(full, prob, d.mask) & ~exclude
        scale = from_rectangle(card.corners_px, card_mod.CARD_MM, "bank card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm",
                               long_first=card.long_first)
        centre = scale.to_mm(np.mean(card.corners_px, axis=0))[0]
        # Length and shape: the U-Net's crack line, which stays continuous on a faint crack.
        # Width: the U-Net's area is wider than the crack, so inside it the classical finder (dark line against its
        # background, src/measure/crack.py) gives the crack's edges; it may find a faint crack only in pieces, so
        # every piece along the line counts for the width, none of them for the length.
        # Every crack in the photo, not only the longest (D88): each is measured this way, and the level takes the
        # widest point over all of them (src/measure/cracks.py).
        found_edges = segment(grey, None, exclude, BACKGROUND_PX, None)
        from_area = []

        def one(line_mask):
            along = measure(grey, line_mask, scale.to_mm, scale.rel_uncertainty, centre)
            edges = found_edges & ndimage.binary_dilation(line_mask, iterations=UNET_GROW)
            wide = measure(grey, edges, scale.to_mm, scale.rel_uncertainty, centre)
            if along.found and wide.found and wide.width_max_mm is not None:
                return replace(wide, **{k: getattr(along, k) for k in LINE_FIELDS})
            from_area.append(along.found)
            return along

        r.crack = cracks_mod.measure_all(unet, one)
        if any(from_area):
            r.warnings.append("Width from the U-Net's crack area, which is wider than the crack: the width may "
                              "be too large.")
        for f in r.crack.flags:
            if "retake closer" in f and r.measure_note is None:
                r.measure_note = f
            elif "retake closer" not in f:
                r.warnings.append(f)
        if r.crack.matrix_input() is None and r.measure_note is None:
            r.measure_note = "The width could not be read: go closer so the crack is at least 3 pixels wide."
        if r.measure_note is None and r.crack.matrix_input() is not None:
            _add_kinematics(r)  # D99: in mm, before the 3D geometry, which draws it
        r.geometry = wall_geometry(r.crack, scale.to_mm, card.corners_px, r.kinematics)
    elif d.statement == NO_CRACK and not seen_steel:
        r.measure_note = "Bank card found; there is no crack to measure."

    if d.statement in (CRACK, UNSURE):
        found = r.crack is not None and r.crack.found
        m = None if found else pixel_line(full, prob, d.mask)  # a tilted card spoils the mm, not the crack's place
        r.photo, r.pixel_crack = full, m
        traced = r.crack if found else m
        _keep_for_link(r, full, r.crack.main_path_px if found else (None if m is None else m.main_path_px), d.mask,
                       paths=None if traced is None else traced.branch_paths_px)
    if r.measured:
        c = r.crack
        line = (f"{d.reason} Confidence {d.confidence:.0%}. Widest {c.width_max_mm:.2f} ± "
                f"{c.width_uncertainty_mm:.2f} mm, length {c.length_main_mm:.0f} mm (bank card).")
    elif d.statement != NO_CRACK:
        line += " Width not measured: " + r.measure_note[0].lower() + r.measure_note[1:]
    r.reason = line
    r.overlay = _draw_card_and_widest(draw_overlay(work, d.mask, r.statement, line), k, card,
                                      r.crack if r.measured else None, full)
    if d.statement in (CRACK, UNSURE):
        r.views = _views(work, prob, k, r.crack, r.measured)
        if r.kinematics is None:
            _add_kinematics(r)
    _add_anomaly(r, full, prob, d, card)
    _steel_defects(r)
    return r


def _views(work, prob, k, crack, measured) -> dict:
    """The heatmaps and the enhanced view of photo 1 (D81); never stops an answer if one fails."""
    try:
        from app.views import make
        return make(work, prob, k, crack, measured)
    except Exception as e:  # noqa: BLE001  (a view is extra: the answer stands without it)
        print("views:", type(e).__name__, e, flush=True)
        return {}


@dataclass
class WallEntry:
    number: int
    statement: str
    thumb: bytes  # JPEG, long side WALL_THUMB
    large: bytes  # JPEG, long side WALL_LARGE, for the result's own page
    stats: dict  # MeasuredResult.to_dict()
    photo2: bytes | None = None  # JPEG of photo 2 (the whole element) with the crack in red (D62)
    views: dict = field(default_factory=dict)  # name -> JPEG: heatmaps and enhanced view of photo 1 (D81)
    parts_img: bytes | None = None  # JPEG of photo 2 with SAM 3.1's parts in colour (D74); in memory only
    zoom: bytes | None = None  # JPEG of photo 1, long side 2000, without marks, for the crack viewer (D83); memory only
    marks: dict | None = None  # the crack's points and widths for the viewer (crack_view.view_data), never exported


EXPORT_DIR = Path(os.environ.get("SECOND_LOOK_EXPORT_DIR", Path(__file__).resolve().parents[2] / "data" / "results"
                                  / "cracks"))


def export_name(number: int) -> str:
    return f"{number:03d}.json"


def mesh_name(number: int) -> str:
    return f"{number:03d}.mesh.json"


@dataclass
class Wall:
    """Shared results for the projector, in memory only; gone when the app stops (as the original).
    With export_dir, each shared result's numbers (never the photo) are also written there as NNN.json for the 3D model
    (D57); the folder is emptied when the wall starts, so it forgets with the app."""
    size: int = WALL_SIZE
    export_dir: Path | None = None
    _items: deque = field(default_factory=deque)
    _counts: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next: int = 1

    def __post_init__(self):
        if self.export_dir is not None:
            self.export_dir = Path(self.export_dir)
            self.export_dir.mkdir(parents=True, exist_ok=True)
            for f in self.export_dir.glob("[0-9][0-9][0-9]*.json"):  # only our own files
                f.unlink()

    def _export(self, number: int, stats: dict) -> None:
        if self.export_dir is None:
            return
        tmp = self.export_dir / (export_name(number) + ".tmp")
        tmp.write_text(json.dumps({"number": number, **stats}, indent=1), encoding="utf-8")
        tmp.replace(self.export_dir / export_name(number))  # whole file at once: Grasshopper never reads half of it

    def mesh_path(self, number: int) -> Path | None:
        """The 3D model Grasshopper wrote for this result, if it is there (D57)."""
        if self.export_dir is None:
            return None
        p = self.export_dir / mesh_name(number)
        return p if p.exists() else None

    def add(self, statement: str, image: Image.Image, stats: dict | None = None, views: dict | None = None,
            photo: Image.Image | None = None, marks: dict | None = None) -> int:
        """Keeps a small JPEG for the wall, a larger one and the stats for the result's own page (added in the
        copy). Returns the result's number. The wall sorts by risk, so it shows the photo without the statement's
        band ("CRACK FOUND", D78); the sender's phone keeps it."""
        if image.info.get("photo_height"):
            image = image.crop((0, 0, image.width, image.info["photo_height"]))

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
            entry = WallEntry(n, statement, thumb, large, stats or {}, marks=marks)
            if photo is not None:  # the crack viewer's photo (D83), in memory only as the others
                from app.crack_view import photo_jpeg
                entry.zoom = photo_jpeg(photo)
            for name, v in (views or {}).items():  # in memory only, as the photo
                buf = io.BytesIO()
                im = v.convert("RGB")
                im.thumbnail((WALL_LARGE, WALL_LARGE))
                im.save(buf, "JPEG", quality=85)
                entry.views[name] = buf.getvalue()
            self._items.appendleft(entry)
            while len(self._items) > self.size:
                self._items.pop()
            self._counts[statement] += 1
        self._export(n, stats or {})
        return n

    def items(self) -> list[tuple[str, bytes]]:
        """(statement, small JPEG), newest first: as the original."""
        with self._lock:
            return [(e.statement, e.thumb) for e in self._items]

    def entries(self) -> list[WallEntry]:
        with self._lock:
            return list(self._items)

    def set_photo2(self, number: int, image: Image.Image, attr: str = "photo2") -> bool:
        """Photo 2 with its crack in red (D62), or with SAM 3.1's parts (attr "parts_img", D74), for the gallery and
        the result page; in memory only, as photo 1."""
        img = image.convert("RGB")
        img.thumbnail((WALL_LARGE, WALL_LARGE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        with self._lock:
            for e in self._items:
                if e.number == number:
                    setattr(e, attr, buf.getvalue())
                    return True
        return False

    def update(self, number: int, stats: dict) -> bool:
        """New stats for a result still on the wall (after the element photo); False if it is gone."""
        with self._lock:
            for e in self._items:
                if e.number == number:
                    e.stats = stats
                    break
            else:
                return False
        self._export(number, stats)
        return True

    def get(self, number: int) -> WallEntry | None:
        with self._lock:
            return next((e for e in self._items if e.number == number), None)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {s: self._counts[s] for s in STATEMENTS}


def analyse(data: bytes, consent: bool, share: bool, wall: Wall, model, device: str = "cpu",
            gpu: threading.Lock | None = None, scene_model=None, card_placed: bool = False,
            test: bool = False, previous: int | None = None) -> MeasuredResult:
    """One upload, as the original: consent, size and format, then the pipeline. Only pixels are read.
    previous: the wall number of an earlier photo of the same crack, to compare with (D99)."""
    if not consent:
        raise ConsentMissing("Tick the consent box first.")
    photo = _open_photo(data)
    with gpu or threading.Lock():
        result = run_measured(photo, model, device, scene_model, card_placed)
    result.test = test
    if previous is not None and result.statement in (CRACK, UNSURE):
        from measure.growth import compare
        before = wall.get(previous)
        result.growth = compare(None if before is None else before.stats, result.to_dict(), previous)
    if share:
        from app.crack_view import view_data
        result.wall_number = wall.add(result.statement, result.overlay, result.to_dict(), result.views,
                                      result.photo, view_data(result))
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
                    gpu: threading.Lock | None = None, scene_model=None, crack_model=None,
                    spalling_model=None, parts_model=None) -> ElementAnswer:
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
        if crack_model is not None:  # where the crack is on the element and how large next to it (D61)
            from element.photo2 import read as read_on_element
            result.on_element = read_on_element(crack_model, photo, device)
        if spalling_model is not None:
            from element.spalling import seen
            result.spalling, result.spalling_p = seen(spalling_model, photo, device)
    locate_crack(result, photo)
    crack_from_link(result, photo)
    if parts_model is not None:
        find_parts_for(result, photo, parts_model, gpu)
        frame_from_parts(result)
    result.estimate = estimate_from_photo2(result) if result.test else None
    if result.wall_number is not None and result.on_element is not None and result.on_element.overlay is not None:
        wall.set_photo2(result.wall_number, photo2_picture(result))
    if result.wall_number is not None and result.parts_preview is not None:
        wall.set_photo2(result.wall_number, result.parts_preview, "parts_img")
    auto_triage(result, wall)
    return result.element


LINKED = "photo 1's crack carried into photo 2 through the {via} (the same crack)"
NOT_LINKED = ("Photo 2 could not be linked to photo 1, and from far away the crack model can mark something else: "
              "the crack's place on the element is not known.")


def crack_from_link(result: MeasuredResult, photo: Image.Image) -> None:
    """The crack in photo 2 (D75). Linked (bank card or key points): photo 1's crack, every branch, carried across by
    the link: the same crack by construction, even where it is too thin to see from far. Not linked: photo 2's own
    reading only for a TEST case without a card (a crop of photo 2, D63); otherwise the place is not known, since the
    crack model on a far photo can mark a plant or a frame (Luis's field test, 11 Nov)."""
    from element import photo2
    on, loc = result.on_element, result.location or {}
    if on is None or not on.size_px:
        return
    linked = loc.get("linked") and result.link_crack_element is not None
    if not linked:
        if not (result.test and result.card is None) and on.found:
            plain = ImageOps.exif_transpose(photo).convert("RGB").resize(on.size_px)
            result.on_element = photo2.CrackOnElement(False, reason=NOT_LINKED, size_px=on.size_px)
            result.on_element.overlay = plain
        return
    from element.locate import Location
    lm = Location(True, "unknown", "")
    lm.link_map = result.link_map
    full_w = ImageOps.exif_transpose(photo).width
    s = on.size_px[0] / full_w  # element photo's own pixels -> photo 2 as analysed
    main = result.link_crack_element * s
    branches = [c * s for c in (lm.carry(b) for b in (result.link_network_px or [])) if c is not None and len(c) >= 2]
    # D95: the carried crack seeds the tracing of photo 2 itself, so the crack is followed across the whole element
    grown = photo2.grow_from_seed(photo, on.size_px, branches or [main], getattr(on, "mask", None))
    if grown:
        branches = grown
        main = max(grown, key=lambda b: float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum()))
    new = photo2.place(main, on.size_px, None, branches or None)
    via = loc.get("via") or "link"
    new.frame_source, new.source = photo2.WHOLE_PHOTO, LINKED.format(via=via) + (
        ", then traced on this photo from there" if grown else "")
    new.reason = f"{new.reason[:-1]}, {new.source}."
    if result.link_width_px is not None and len(main) >= 2:  # photo 1's width, carried at the link's local scale
        c = np.asarray(result.link_line_px, float)
        a = lm.carry(c[[0, -1]])
        k = float(np.linalg.norm(a[1] - a[0]) * s / max(np.linalg.norm(c[-1] - c[0]), 1e-6))
        new.width_px_max, new.width_px_median = result.link_width_px[0] * k, result.link_width_px[1] * k
        new.length_px = float(np.linalg.norm(np.diff(main, axis=0), axis=1).sum())
    img = ImageOps.exif_transpose(photo).convert("RGB").resize(on.size_px)
    d = ImageDraw.Draw(img)
    lw = max(3, img.width // 160)
    for b in (branches or [main]):
        d.line([tuple(p) for p in b], fill=(214, 39, 40), width=lw)
    new.overlay = img
    result.on_element = new


FRAME_PARTS = ("column", "beam", "wall")  # SAM parts that can be the element holding the crack (D74)
FRAME_MIN_SHARE = 0.25  # the outline must hold at least this share of the crack's points (illustrative)


def frame_from_parts(result: MeasuredResult) -> None:
    """The element's outline in photo 2 from SAM 3.1 (D74): the column, beam or wall mask holding most of photo 2's
    crack, the model's element first when it holds enough. The crack is then read relative to that outline, not to
    the whole photo. Without SAM, or no outline holding the crack, the whole photo stays the frame (said so)."""
    on, parts = result.on_element, result.parts
    things = getattr(result, "parts_things", None)
    if on is None or not on.found or not things or not on.network_px or not on.size_px:
        return
    w, h = on.size_px
    pts = np.vstack(on.network_px)
    want = (result.element.element if result.element is not None else None)
    best = None
    for part, mask, score in things:
        if part not in FRAME_PARTS:
            continue
        mh, mw = mask.shape
        xs = np.clip((pts[:, 0] / max(w - 1, 1) * (mw - 1)).round().astype(int), 0, mw - 1)
        ys = np.clip((pts[:, 1] / max(h - 1, 1) * (mh - 1)).round().astype(int), 0, mh - 1)
        share = float(mask[ys, xs].mean())
        key = (share >= FRAME_MIN_SHARE and part == want, share)
        if share >= FRAME_MIN_SHARE and (best is None or key > best[0]):
            best = (key, part, mask, score, share)
    if best is None:
        return
    _, part, mask, score, share = best
    yy, xx = np.nonzero(mask)
    mh, mw = mask.shape
    frame = (xx.min() / mw, yy.min() / mh, (xx.max() + 1) / mw, (yy.max() + 1) / mh)
    on.reframe(frame, f"SAM 3.1's outline of the {part} ({share:.0%} of the crack on it, score {score:.2f})")


def photo2_picture(result: MeasuredResult) -> Image.Image:
    """Photo 2 with the crack in red and, when SAM gave it, the element's outline box in white (D74)."""
    on = result.on_element
    img = on.overlay.convert("RGB").copy()
    if on.frame is not None and tuple(on.frame) != (0.0, 0.0, 1.0, 1.0):
        x0, y0, x1, y1 = on.frame
        d = ImageDraw.Draw(img)
        d.rectangle([x0 * img.width, y0 * img.height, x1 * img.width - 1, y1 * img.height - 1],
                    outline=(255, 255, 255), width=max(2, img.width // 200))
    return img


def estimate_from_photo2(result: MeasuredResult) -> dict | None:
    """TEST cases only (D62): the crack's width and length estimated from photo 2, assuming the element fills the photo
    and has its typical size. Coarse (one pixel of uncertainty). None without a crack measured on photo 2, or when the
    crack was measured with a bank card (then the measurement is used)."""
    from element.predict import matrix_element
    from gh3d.layout import MATRIX_TO_SHAPE, SIZES
    on = result.on_element
    if result.measured or result.card is not None or on is None or not on.found or on.width_px_max is None \
            or not on.size_px:  # never with a card in photo 1: a tilted card asks for a retake, not a guess (D75)
        return None
    shape = MATRIX_TO_SHAPE.get(matrix_element(result.element) or "", "wall")
    if shape == "slab":
        return None
    (dx, _, dz), _ = SIZES[shape]
    x0, y0, x1, y1 = on.frame or (0.0, 0.0, 1.0, 1.0)
    w, h = on.size_px[0] * (x1 - x0), on.size_px[1] * (y1 - y0)  # the element's size in photo 2's pixels
    mm_per_px = dx / w if shape == "beam" else dz / h  # a beam spans its outline's width, a column or wall its height
    return {"width_max_mm": round(on.width_px_max * mm_per_px, 1),
            "width_median_mm": round((on.width_px_median or on.width_px_max) * mm_per_px, 1),
            "uncertainty_mm": round(mm_per_px, 1), "length_mm": round((on.length_px or 0) * mm_per_px),
            "mm_per_px": round(mm_per_px, 2),
            "basis": f"the width of the damage marked on photo 2 (often a spalled strip, not the crack's opening), "
                     f"estimated with the {shape} assumed to fill {on.frame_source} and to be "
                     f"{dx if shape == 'beam' else dz} mm {'long' if shape == 'beam' else 'high'}; not measured (TEST only)"}


def locate_crack(result: MeasuredResult, photo: Image.Image) -> None:
    """Where on the element the crack is, from the two photos linked (D66). No crack photo kept: nothing is read."""
    from element.locate import locate
    from element.predict import matrix_element
    if result.link_grey is None:
        return
    loc = locate(result.link_grey, result.link_line_px, np.asarray(ImageOps.exif_transpose(photo).convert("RGB")),
                 matrix_element(result.element), result.link_card, result.link_card_size)
    result.location, result.location_preview = loc.to_dict(), loc.preview
    result.link_crack_element = loc.crack_px
    result.link_map = loc.link_map


PARTS_SIDE = 1600  # the element photo's long side for SAM 3.1 (it reads 644 px anyway)


def find_parts_for(result: MeasuredResult, photo: Image.Image, segment, gpu: threading.Lock | None = None) -> None:
    """Which part the crack lies on, from SAM 3.1's masks (D70). Shown only: the level, the element it uses, the
    position and the angle stay as they are (rule 5). If SAM fails (memory), the app goes on without it."""
    from element.parts import find_parts
    rgb = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
    k = min(1.0, PARTS_SIDE / max(rgb.shape[:2]))
    if k < 1.0:
        rgb = np.asarray(Image.fromarray(rgb).resize((round(rgb.shape[1] * k), round(rgb.shape[0] * k))))
    crack = None if result.link_crack_element is None else result.link_crack_element * k
    on = result.on_element
    network = None
    if on is not None and on.found and on.path_px is not None and on.size_px:
        # the crack exactly as the element photo shows it (D95): its line and every branch, the same crack in both
        s = rgb.shape[1] / on.size_px[0]
        crack = on.path_px * s
        network = [b * s for b in (on.network_px or []) if len(b) >= 2] or None
    try:
        with gpu or threading.Lock():
            p = find_parts(rgb, crack, segment, network)
    except Exception as e:  # noqa: BLE001  any failure of the optional model must not stop the triage
        result.parts = {"on": None, "name": None, "reason": f"SAM 3.1 could not run on this photo ({type(e).__name__}).",
                        "source": "SAM 3.1 (D70)"}
        result.parts_preview = None
        return
    result.parts, result.parts_preview = p.to_dict(), p.preview
    result.parts_things = [(part, m, s) for part, m, s in (p.things or [])]  # in memory only, for the frame (D74)
    try:
        whole_element(result, photo, rgb, p)
    except Exception as e:  # noqa: BLE001  the whole-element tracing is shown only; it never stops the triage
        result.warnings.append(f"Whole-element tracing not run ({type(e).__name__}).")


def whole_element(result: MeasuredResult, photo: Image.Image, rgb: np.ndarray, p) -> None:
    """Every crack and spalled area on the whole element (D96), inside the outline of the part the crack lies on;
    the element photo and the parts view are both drawn from it, so they always show the same crack."""
    from element import photo2
    from element.parts import preview
    on = result.on_element
    if on is None or not on.found or not on.size_px or not p.on or not p.things:
        return
    w, h = on.size_px
    region = np.zeros((rgb.shape[0], rgb.shape[1]), bool)
    s0 = rgb.shape[1] / w
    pts = np.vstack([np.asarray(b, float) for b in (on.network_px or [on.path_px]) if b is not None and len(b)]) * s0
    pts = np.clip(np.round(pts).astype(int), 0, [rgb.shape[1] - 1, rgb.shape[0] - 1])
    best = 0
    for part, m, _ in p.things:  # D97: the one outline holding the crack, not every wall SAM found
        if part == p.on and m.shape == region.shape:
            hit = int(m[pts[:, 1], pts[:, 0]].sum())
            if hit > best:
                best, region = hit, m.copy()
    if not region.any():
        for part, m, _ in p.things:
            if part == p.on:
                region |= m
    if not region.any():
        return
    small = np.asarray(Image.fromarray(region.astype(np.uint8) * 255).resize((w, h))) > 0
    lab, n = ndimage.label(small)  # D97: the piece of the outline holding the crack (SAM gives all walls as one)
    if n > 1:
        q = np.clip(np.round(pts / s0).astype(int), 0, [w - 1, h - 1])
        hits = np.bincount(lab[q[:, 1], q[:, 0]], minlength=n + 1)
        hits[0] = 0
        if hits.max() > 0:
            small = lab == hits.argmax()
    sam = getattr(photo2.grow_from_seed, "last_sam", None)
    nets = photo2.trace_whole_element(photo, (w, h), small, on.network_px or [on.path_px], getattr(on, "mask", None),
                                      sam if sam is not None and sam.shape == (h, w) else None)
    main = max(nets, key=lambda b: float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum()))
    new = photo2.place(main, (w, h), on.frame if on.frame is not None else None, nets)
    for k in ("path_uv", "extent", "centre_uv", "reason", "network_uv", "path_px", "network_px"):
        setattr(on, k, getattr(new, k))
    on.source = f"{on.source}; every crack inside the {p.on}'s outline traced on this photo (D96)"
    work = ImageOps.exif_transpose(photo).convert("RGB").resize((w, h))
    spall = []
    try:  # spalled areas on the element, the same rules as on the close-up (D93), inside the outline
        from anomaly import spalling_sam
        lines = np.zeros((h, w), bool)
        for b in nets:
            for q in b.astype(int):
                lines[min(h - 1, max(0, q[1])), min(w - 1, max(0, q[0]))] = True
        spall = [f for f in (spalling_sam.find(work, lines, ~small) or []) if f["share"] > 0]
    except Exception:
        spall = []
    on.spalling_px = [np.asarray(f["outline"], float) * (w, h) for f in spall]
    img = work.copy()
    d = ImageDraw.Draw(img)
    lw = max(3, img.width // 160)
    for poly in on.spalling_px:
        d.polygon([tuple(q) for q in poly], outline=(255, 92, 122), width=lw)
    for b in nets:
        d.line([tuple(q) for q in b], fill=(214, 39, 40), width=lw)
    on.overlay = img
    s = rgb.shape[1] / w  # the parts view: the same cracks and spalled areas
    pv = preview(rgb, p.things, [b * s for b in nets], p.on)
    if on.spalling_px:
        k = pv.width / rgb.shape[1]
        dd = ImageDraw.Draw(pv)
        for poly in on.spalling_px:
            dd.polygon([tuple(q) for q in poly * s * k], outline=(255, 92, 122), width=max(2, pv.width // 250))
    result.parts_preview = pv


def level_for(result: MeasuredResult, element: str | None, position: str, pattern: str | None,
              rust_stain: bool = False, confidence: float | None = None) -> LevelResult:
    """The written risk matrix's level for these inputs (rule 5). No width (no bank card, or too few pixels) gives
    Expert needed in the matrix itself; never a "safe" answer. confidence: the crack model's when not given."""
    c = result.crack.matrix_input() if result.measured else None
    if c is None and result.test and result.estimate:  # D62: test cases only, labelled as an estimate
        c = {"width_mm": result.estimate["width_max_mm"], "width_uncertainty_mm": result.estimate["uncertainty_mm"]}
    conf = confidence if confidence is not None else result.confidence
    conf = conf if conf is not None else float("nan")
    signs = (("rust_stain",) if rust_stain else ()) + (
        ("growing",) if (result.growth or {}).get("grown") is True else ())  # D103: grown beyond the error ([46], D23)
    return assess(pattern or "none", element or "unknown", c["width_mm"] if c else float("nan"), conf,
                  result.statement != "Retake", width_uncertainty_mm=c["width_uncertainty_mm"] if c else 0.0,
                  escalators=signs, location=POSITIONS[position])


def where_on_element(result: MeasuredResult, element: str | None) -> tuple[str, str | None]:
    """The crack's position on the element and why. First the two photos linked (D66, D67): the strict reading, which
    alone can say "middle" for a real upload. Else the crack found on photo 2 itself (D62): an end zone counts, but
    "middle" only for TEST cases, since it assumes the element fills the photo; a real upload then stays unknown
    (counted as an end zone, never lower; rule 2)."""
    loc = result.location or {}
    if loc.get("position", "unknown") != "unknown":
        return loc["position"], loc.get("reason")
    on = result.on_element
    if on is not None and getattr(on, "found", False):
        p = on.position("beam" if element == "beam" else element)
        if p == "end_zone" or (p == "middle" and result.test):
            return p, "read from the crack on the photo of the whole element" + (" (TEST case)" if p == "middle" else "")
    return "unknown", loc.get("reason")


def auto_triage(result: MeasuredResult, wall: Wall | None = None) -> LevelResult | None:
    """The level, given by itself once the element photo is in (D55): the model's element, the written rules' type
    and the written risk matrix. Any doubt gives Expert needed, never a category: an element the model is unsure of,
    a type the rules call unclear, no width. The position comes from the two photos when they can be linked
    (D66); otherwise it is unknown and counts as an end zone (never lower).
    Stored on the result and on its wall entry; None before the element photo."""
    from element.predict import matrix_element
    if result.element is None or result.statement not in ("Crack found", "Expert needed"):
        return None
    element = matrix_element(result.element)
    s = result.suggested_pattern()
    pattern = s.pattern if s is not None and s.pattern and s.clear else None
    conf = min(result.confidence if result.confidence is not None else 0.0, result.element.probability)
    position, position_reason = where_on_element(result, element)
    level = level_for(result, element, position, pattern, confidence=conf)
    why = [result.element.reason] if element is None else []
    on = result.on_element
    if on is not None and on.found:  # what photo 2 added (D61)
        why.append(f"From the photo of the element: the crack spans about {on.extent:.0%} of it, "
                   f"{POSITION_NAMES[position]}.")
    if result.spalling:
        why.append(f"Spalling or exposed bars seen on the photo of the element ({result.spalling_p:.0%}).")
    if (result.growth or {}).get("grown") is True:
        why.append(result.growth["reason"])
    if not result.measured and result.test and result.estimate:
        why.append(f"Width {result.estimate['width_max_mm']:g} ± {result.estimate['uncertainty_mm']:g} mm "
                   f"{result.estimate['basis']}.")
    elif not result.measured:
        why.append("The width needs a bank card next to the crack." if s is not None and s.pattern else
                   "The width needs a bank card next to the crack, and the crack type is not clear from the photo.")
    elif pattern is None:
        why.append("The crack type is not clear from the photo.")
    result.triage = {"source": AUTO_SOURCE, "element": element,
                     "element_name": ELEMENTS.get(element) if element else None,
                     "element_probability": round(result.element.probability, 3), "position": position,
                     "position_name": POSITION_NAMES[position],
                     "position_reason": position_reason, "pattern": pattern,
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
