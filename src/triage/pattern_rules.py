"""Crack pattern suggested by written rules from the crack's shape and the element. Spec: spec/features/pattern_rules.md.

Draft v0 (D43). Every angle and limit is illustrative until a structural engineer sets it. The result is a
suggestion for the field worker to confirm; it never goes into the risk matrix by itself and never carries a level.
"""

from dataclasses import dataclass

from triage.matrix import ELEMENTS, PATTERN_NAMES

VERSION = "pattern-rules-v0 (illustrative)"
PARALLEL_MAX_DEG = 25.0
DIAGONAL_DEG = (35.0, 55.0)
PERPENDICULAR_MIN_DEG = 65.0
MIN_STRAIGHTNESS = 0.6
MAP_MIN_LOOPS = 2
MAP_MAX_COHERENCE = 0.35
MAP_MIN_BRANCH_RATIO = 1.8
AXIS_TO_VERTICAL_DEG = {"column": 0.0, "beam": 90.0}  # walls and slabs have no axis
WALLS = {"load_bearing_wall", "infill_wall"}


@dataclass(frozen=True)
class PatternSuggestion:
    pattern: str | None  # a pattern key of the risk matrix, or None
    clear: bool
    rule: int
    reason: str

    def to_dict(self) -> dict:
        return {"version": VERSION, "pattern": self.pattern,
                "pattern_name": PATTERN_NAMES.get(self.pattern) if self.pattern else None,
                "clear": self.clear, "rule": self.rule, "reason": self.reason,
                "note": "Suggestion only: a field worker confirms it before it enters the risk matrix."}


def _band(phi: float) -> str | None:
    if phi <= PARALLEL_MAX_DEG:
        return "parallel"
    if DIAGONAL_DEG[0] <= phi <= DIAGONAL_DEG[1]:
        return "diagonal"
    if phi >= PERPENDICULAR_MIN_DEG:
        return "perpendicular"
    return None


def _nearest_band(phi: float) -> str:
    return "parallel" if phi < 30 else "diagonal" if phi < 60 else "perpendicular"


def _pattern_for(element: str, band: str, rust_stain: bool) -> tuple[str, bool, str]:
    if band == "diagonal":
        return "diagonal_shear", True, f"inclined across the {ELEMENTS[element]}"
    if band == "perpendicular":
        if element == "column":
            return "column_horizontal", True, "horizontal across the column"
        return "flexural", True, "across the beam"
    if rust_stain:
        return "along_bar", True, f"along the {ELEMENTS[element]} with a rust stain"
    if element == "column":
        return "along_bar", False, "vertical in the column without rust: it can also be crushing under load"
    return "along_bar", False, "along the beam without rust: it may follow a bar, or not"


def suggest(shape: dict | None, element: str | None, *, at_joint: bool | None = None,
            spalling: bool | None = None, rust_stain: bool = False) -> PatternSuggestion:
    """First match wins (spec table). `shape` is CrackMeasurement.to_dict()['shape'] or None."""
    if element is not None and element not in ELEMENTS:
        raise ValueError(f"unknown element: {element!r}")
    if not shape or shape.get("angle_to_vertical_deg") is None:
        return PatternSuggestion(None, False, 1, "No crack measured, so no pattern can be read.")
    if spalling:
        return PatternSuggestion("spalling_exposed_bar", True, 2, "Spalling or an exposed bar was seen.")
    loops, coh, ratio = shape.get("n_loops") or 0, shape.get("coherence"), shape.get("branch_ratio")
    if loops >= MAP_MIN_LOOPS or (coh is not None and ratio is not None and coh < MAP_MAX_COHERENCE
                                  and ratio >= MAP_MIN_BRANCH_RATIO):
        return PatternSuggestion("surface_map", True, 3,
                                 f"A network of cracks ({loops} closed cells, no main direction).")
    if element is None:
        return PatternSuggestion(None, False, 4, "Element unknown: the direction cannot be read against it.")
    if at_joint and element in ("column", "beam"):
        return PatternSuggestion("joint", True, 5, "The crack is at a beam-column joint.")
    straight = shape.get("straightness")
    if straight is not None and straight < MIN_STRAIGHTNESS:
        return PatternSuggestion(None, False, 6, f"The crack winds too much (straightness {straight:.2f}) to read "
                                                 "its direction.")
    a = float(shape["angle_to_vertical_deg"])
    if element == "slab":
        return PatternSuggestion(None, False, 7, "Slab seen from below: there is no up-down direction to read.")
    if element in WALLS:
        if DIAGONAL_DEG[0] <= a <= DIAGONAL_DEG[1]:
            return PatternSuggestion("diagonal_shear", True, 8, f"Diagonal crack in a wall ({a:.0f} degrees from "
                                                                "vertical).")
        return PatternSuggestion(None, False, 9, f"Crack in a wall at {a:.0f} degrees from vertical: no rule yet.")
    phi = abs(a - AXIS_TO_VERTICAL_DEG[element])  # angle between crack and element axis, 0 to 90
    band = _band(phi)
    if band is None:
        pattern, _, why = _pattern_for(element, _nearest_band(phi), rust_stain)
        return PatternSuggestion(pattern, False, 15, f"{phi:.0f} degrees to the {ELEMENTS[element]} axis, between "
                                                     f"two bands: closest is {why}.")
    pattern, clear, why = _pattern_for(element, band, rust_stain)
    rule = {"column_horizontal": 10, "diagonal_shear": 11, "flexural": 12}.get(pattern, 13 if clear else 14)
    return PatternSuggestion(pattern, clear, rule, f"{phi:.0f} degrees to the {ELEMENTS[element]} axis: {why}.")
