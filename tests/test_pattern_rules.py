"""Tests for the pattern rules (spec/features/pattern_rules.md) and the crack shape they read."""

import numpy as np
import pytest

from measure.crack import measure, segment
from measure.reference import from_length
from triage.pattern_rules import suggest

SCALE = from_length((0, 0), (500, 0), 50)  # 0.1 mm per px, image upright


def _crack(angle_to_vertical_deg, width_px=6.0, length_px=300):
    """Dark straight crack; angle 0 = vertical, 90 = horizontal (image y points down)."""
    h, w = 400, 500
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    a = np.radians(angle_to_vertical_deg)
    d = np.array([np.sin(a), np.cos(a)])  # direction (x, y)
    u = (xs - 250) * d[0] + (ys - 200) * d[1]
    v = -(xs - 250) * d[1] + (ys - 200) * d[0]
    return np.where((np.abs(v) <= width_px / 2) & (np.abs(u) <= length_px / 2), 50.0, 160.0)


def _shape(grey):
    r = measure(grey, segment(grey), SCALE.to_mm)
    assert r.found
    return r.to_dict()["shape"]


def shape(angle, **kw):
    return {"angle_to_vertical_deg": angle, "straightness": 0.97, "coherence": 0.95, "n_loops": 0,
            "branch_ratio": 1.0, **kw}


@pytest.mark.parametrize("angle", [0, 30, 45, 90])
def test_measured_angle(angle):
    s = _shape(_crack(angle))
    assert s["angle_to_vertical_deg"] == pytest.approx(angle, abs=3)
    assert s["straightness"] > 0.95 and s["coherence"] > 0.9 and s["n_loops"] == 0


def test_network_is_map_cracking():
    grey = np.full((400, 400), 160.0)
    rng = np.random.default_rng(0)
    for k in range(60, 360, 70):  # a wobbly grid: closed cells, no main direction
        grey[k + rng.integers(-3, 4):k + 4, 40:360] = 50
        grey[40:360, k + rng.integers(-3, 4):k + 4] = 50
    s = _shape(grey)
    assert s["n_loops"] >= 2
    assert suggest(s, "load_bearing_wall").pattern == "surface_map"


def test_beam():
    assert suggest(_shape(_crack(0)), "beam").pattern == "flexural"  # vertical = across a horizontal beam
    assert suggest(_shape(_crack(45)), "beam").pattern == "diagonal_shear"
    s = suggest(_shape(_crack(90)), "beam")
    assert s.pattern == "along_bar" and not s.clear
    assert suggest(_shape(_crack(90)), "beam", rust_stain=True).clear


def test_column():
    assert suggest(shape(88), "column").pattern == "column_horizontal"
    assert suggest(shape(40), "column").pattern == "diagonal_shear"
    vertical = suggest(shape(5), "column")
    assert vertical.pattern == "along_bar" and not vertical.clear and "crushing" in vertical.reason


def test_margins_are_unclear():
    for a in (30, 60):
        s = suggest(shape(a), "beam")
        assert not s.clear and s.rule == 15


def test_overrides_and_unknowns():
    assert suggest(shape(45), "beam", spalling=True).pattern == "spalling_exposed_bar"
    assert suggest(shape(10), "beam", at_joint=True).pattern == "joint"
    assert suggest(shape(10), "load_bearing_wall", at_joint=True).pattern is None  # joint only for column, beam
    assert suggest(shape(45), None).pattern is None
    assert suggest(shape(45), "slab").pattern is None
    assert suggest(shape(45, straightness=0.4), "beam").pattern is None
    assert suggest(None, "beam").pattern is None
    assert suggest(shape(45), "infill_wall").pattern == "diagonal_shear"
    assert suggest(shape(5), "infill_wall").pattern is None
    with pytest.raises(ValueError):
        suggest(shape(45), "roof")


def test_suggestion_never_carries_a_level():
    d = suggest(shape(45), "column").to_dict()
    assert not any(k in str(d) for k in ("Category", "Retake", "Expert needed", "safe"))
