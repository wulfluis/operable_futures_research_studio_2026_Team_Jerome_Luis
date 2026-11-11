"""What the second photo adds (spec/features/photo2.md, D61): the crack's place and extent on the element, spalling,
and the level from both photos. No model file is needed."""

import re

import numpy as np
import pytest

from app.core import Wall, auto_triage, run_measured
from element import photo2, spalling
from element.photo2 import CrackOnElement
from gh3d import layout
from test_app_measure import DEMO_MODEL, _demo
from triage.element_rule import ElementAnswer


@pytest.fixture
def measured():
    r = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    r.element = ElementAnswer("column", False, 0.95, "")
    return r


def test_place_in_element_coordinates():
    path = np.array([[0, 99], [50, 49], [99, 0]], float)  # bottom-left... a diagonal across a 100 x 100 photo
    on = photo2.place(path, (100, 100))
    assert on.found and on.path_uv[0] == [0.0, 0.0] and on.path_uv[-1] == [1.0, 1.0]  # v up
    assert on.extent == pytest.approx(1.0) and on.centre_uv == pytest.approx((0.5, 0.5), abs=0.01)


@pytest.mark.parametrize("centre, element, expected", [
    ((0.5, 0.9), "column", "end_zone"), ((0.5, 0.1), "load_bearing_wall", "end_zone"),
    ((0.5, 0.5), "column", "middle"), ((0.1, 0.5), "beam", "end_zone"), ((0.5, 0.9), "beam", "middle")])
def test_end_zone_or_middle_per_element(centre, element, expected):
    on = CrackOnElement(True, [[0, 0], [1, 1]], 0.5, centre)
    assert on.position(element) == expected


def test_no_crack_on_photo_2_means_unknown():
    on = photo2.place(None, (100, 100))
    assert not on.found and on.position("column") == "unknown" and "not known" in on.reason


def test_photo_2_position_reaches_the_matrix(measured):
    measured.on_element = CrackOnElement(True, [[0.5, 0.4], [0.5, 0.6]], 0.2, (0.5, 0.5))
    auto_triage(measured)
    assert measured.triage["position"] == "middle" and "photo of the element" in measured.triage["why"]


def test_spalling_on_photo_2_makes_the_type_spalling(measured):
    measured.spalling, measured.spalling_p = True, 0.93
    auto_triage(measured)
    assert measured.triage["pattern"] == "spalling_exposed_bar"
    assert "Spalling" in measured.triage["why"] and measured.triage["level"] in ("Category 2", "Category 3")


def test_no_spalling_model_raises_nothing(tmp_path):
    assert spalling.load(tmp_path / "none.pt") is None
    assert spalling.seen(None, None) == (None, None)


def test_spalling_threshold_catches_the_target_share():
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.05])
    c = spalling.choose_spall_min(p, np.ones(10, bool), target=0.8)
    assert np.mean(p >= c) >= 0.8 and np.mean(p >= c + 0.05) < 0.8


def test_the_3d_model_uses_photo_2_place_and_proportion(measured):
    measured.on_element = CrackOnElement(True, [[0.3, 0.2], [0.4, 0.5], [0.35, 0.8]], 0.6, (0.35, 0.5))
    auto_triage(measured)
    m = layout.build({"number": 1, **measured.to_dict()})
    assert m["placed_from_photo2"]
    line = np.asarray(next(o for o in m["objects"] if o["name"] == "crack_line")["points"])
    (dx, _, dz), z0 = layout.SIZES["column"]
    assert line[:, 2].min() == pytest.approx(z0 + 0.2 * dz) and line[:, 2].max() == pytest.approx(z0 + 0.8 * dz)
    assert line[:, 0].min() == pytest.approx((0.3 - 0.5) * dx)  # 60 % of the element's height, as on photo 2
    assert any("photo 2" in s for s in m["label"]) and not any(o["name"] == "card" for o in m["objects"])


def test_without_photo_2_crack_the_model_is_as_before(measured):
    m = layout.build({"number": 1, **measured.to_dict()})
    assert not m["placed_from_photo2"] and any(o["name"] == "card" for o in m["objects"])


def test_never_safe_with_both_photos(measured):
    measured.on_element = CrackOnElement(True, [[0.5, 0.1], [0.5, 0.9]], 0.8, (0.5, 0.5))
    measured.spalling, measured.spalling_p = False, 0.1
    auto_triage(measured, Wall())
    assert not re.search(r"\bsafe\b|clearance|permit", str(measured.to_dict()), re.I)
