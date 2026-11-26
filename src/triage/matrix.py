"""Risk matrix v3 and decision rules. Every value is illustrative until calibrated with the engineering partners.

Spec: spec/features/risk_matrix.md (matrix) and spec/features/triage.md (decision order). Decisions: spec/decisions.md.
The numbers live in risk_matrix_v3.json; this module only applies them.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

_DATA = json.loads((Path(__file__).with_name("risk_matrix_v3.json")).read_text(encoding="utf-8"))

VERSION = _DATA["version"]
LEVELS = _DATA["levels"]  # Category 1 < Category 2 < Category 3
BANDS = [b["key"] for b in _DATA["width_bands"]]
BAND_NAMES = {b["key"]: b["name"] for b in _DATA["width_bands"]}
_BAND_UPPER = [b["upper_mm"] for b in _DATA["width_bands"]]
PATTERNS = list(_DATA["patterns"])
PATTERN_NAMES = {k: v["name"] for k, v in _DATA["patterns"].items()}
MATRIX = {band: [_DATA["patterns"][p]["levels"][i] for p in PATTERNS] for i, band in enumerate(BANDS)}
ELEMENTS = {k: v["name"] for k, v in _DATA["elements"].items()}
ELEMENT_RAISE = {k: v["raise"] for k, v in _DATA["elements"].items()}
LOCATIONS = {k: v["name"] for k, v in _DATA["locations"].items()}
_LOCATION_RULE = {k: _DATA["locations"][v.get("same_as", k)] for k, v in _DATA["locations"].items()}
ESCALATORS = {k: v["name"] for k, v in _DATA["escalators"].items()}
BASIS = {k: v["basis"] for k, v in {**_DATA["locations"], **_DATA["escalators"]}.items()}
_ESCALATOR_EFFECT = {k: v["effect"] for k, v in _DATA["escalators"].items()}
NEXT_STEPS = _DATA["next_steps"]
SUPPORTED_MATERIALS = {"concrete"}  # reinforced concrete only (decision D17)

CONFIDENCE_THRESHOLD = 0.8  # illustrative; fixed here so callers cannot lower it
MATERIAL_THRESHOLD = 0.8  # illustrative; below it the material counts as unsure
NEEDS_ENGINEER = {"Category 3", "Expert needed"}


@dataclass(frozen=True)
class Result:
    level: str  # Category 1, Category 2, Category 3, Retake or Expert needed
    reason: str
    engineer_review: bool
    next_step: str


def width_band(width_mm: float) -> str:
    """Lower bound inclusive: a crack on a boundary goes to the wider band."""
    for key, upper in zip(BANDS, _BAND_UPPER):
        if upper is None or width_mm < upper:
            return key
    return BANDS[-1]


def _allowed(pattern: str, element: str) -> bool:
    allowed = _DATA["patterns"][pattern]["elements"]
    return allowed == "all" or element in allowed


def _valid(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _result(level: str, reason: str) -> Result:
    return Result(level, reason, level in NEEDS_ENGINEER, NEXT_STEPS[level])


STEEL_REASON = ("Exposed steel bars are visible: the concrete cover is gone, which is critical whatever the crack looks "
                "like (D112).")


def assess(pattern: str, element: str, width_mm: float, confidence: float, image_ok: bool, *,
           width_uncertainty_mm: float = 0.0, material: str = "concrete", material_confidence: float = 1.0,
           escalators: tuple[str, ...] = (), location: str = "unknown", exposed_steel: bool = False) -> Result:
    """Return level, one-sentence reason, engineer-review flag and next step. Any doubt goes to an expert.
    Order: photo usable -> material -> element and pattern -> confidence -> width -> base level -> raises
    (element, then location on the element, then warning signs). An unknown location counts as an end zone."""
    unknown = [e for e in escalators if e not in ESCALATORS]
    if unknown:
        raise ValueError(f"unknown escalators: {unknown}")
    if location not in LOCATIONS:
        raise ValueError(f"unknown location: {location!r}")
    if image_ok is not True:
        return _result("Retake", "Photo not usable (blur, light or scale card missing): retake it.")
    if exposed_steel is True:  # D112: visible bars are critical whatever the crack looks like; no crack reading needed
        return _result("Category 3", STEEL_REASON)
    if not _valid(material_confidence) or not MATERIAL_THRESHOLD <= material_confidence <= 1:
        return _result("Expert needed", "Material unclear from the photo: an engineer decides.")
    if material not in SUPPORTED_MATERIALS:
        return _result("Expert needed", f"Material not supported ({material}): an engineer decides.")
    if element not in ELEMENTS:
        return _result("Expert needed", "Structural element not recognised.")
    if pattern not in PATTERNS:
        return _result("Expert needed", "Crack pattern not recognised.")
    if not _allowed(pattern, element):
        return _result("Expert needed",
                       f"A {PATTERN_NAMES[pattern]} on a {ELEMENTS[element]} does not fit: an engineer decides.")
    if not _valid(confidence) or not 0 <= confidence <= 1:
        return _result("Expert needed", "Model confidence missing or invalid.")
    if confidence < CONFIDENCE_THRESHOLD:
        return _result("Expert needed", f"Model not confident enough ({confidence:.2f}).")
    if not _valid(width_mm) or width_mm < 0 or not _valid(width_uncertainty_mm) or width_uncertainty_mm < 0:
        return _result("Expert needed", "Crack width could not be measured.")

    upper = width_mm + width_uncertainty_mm  # the safe end of the measurement
    band = width_band(upper)
    rank = LEVELS.index(MATRIX[band][PATTERNS.index(pattern)])
    raised_by = []
    if ELEMENT_RAISE[element]:
        rank += ELEMENT_RAISE[element]
        raised_by.append(f"it is a {ELEMENTS[element]}")
    rule = _LOCATION_RULE[location]
    if rule["raise"] and pattern in rule["patterns"]:
        rank += rule["raise"]
        raised_by.append("its location is unknown, so it counts as an end zone" if location == "unknown"
                         else "it is in an end zone, where bars are anchored")
    for e in dict.fromkeys(escalators):  # each sign counts once
        rank = len(LEVELS) - 1 if _ESCALATOR_EFFECT[e] == "Category 3" else rank + 1
        raised_by.append(ESCALATORS[e])
    level = LEVELS[min(rank, len(LEVELS) - 1)]

    width_text = f"{width_mm:g} mm wide" + (f" (up to {upper:g} mm)" if width_uncertainty_mm else "")
    reason = f"{PATTERN_NAMES[pattern]} in a {ELEMENTS[element]}, {width_text}, {BAND_NAMES[band]} band: {level}"
    reason += f"; raised because {', '.join(raised_by)}." if raised_by else "."
    return _result(level, reason)


def triage(pattern: str, element: str, width_mm: float, confidence: float, image_ok: bool, **kwargs) -> str:
    """Return Category 1, Category 2, Category 3, 'Retake' or 'Expert needed'."""
    return assess(pattern, element, width_mm, confidence, image_ok, **kwargs).level
