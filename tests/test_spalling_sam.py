"""Possible spalling from SAM 3.1 and written rules (D93): an area, not a line, clearly different in colour from the
wall around it. Drawn images and stand-in SAM answers only."""
import numpy as np

from anomaly import spalling_sam as sp


def _wall(h=400, w=400):
    return np.full((h, w, 3), (170, 180, 190), np.uint8)  # a blue-grey plastered wall


def _box(y0, y1, x0, x1, shape=(400, 400)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_a_broken_away_patch_of_another_colour_is_possible_spalling():
    rgb = _wall()
    rgb[150:250, 150:260] = (190, 160, 120)  # the beige material below the plaster
    found = sp.areas(rgb, [("missing plaster", _box(150, 250, 150, 260), 0.86)])
    assert len(found) == 1 and found[0]["delta_e"] >= sp.MIN_DELTA_E


def test_an_area_the_same_colour_as_the_wall_is_not_spalling():
    rgb = _wall()
    rgb[150:250, 150:260] = (175, 183, 192)  # a door frame or a smooth patch: almost the wall's colour
    assert sp.areas(rgb, [("missing plaster", _box(150, 250, 150, 260), 0.86)]) == []


def test_a_thin_line_is_a_crack_not_spalling():
    rgb = _wall()
    rgb[50:350, 200:203] = (60, 55, 50)
    assert sp.areas(rgb, [("broken plaster", _box(50, 350, 199, 204), 0.9)]) == []


def test_low_scores_and_other_prompts_are_ignored():
    rgb = _wall()
    rgb[150:250, 150:260] = (190, 160, 120)
    m = _box(150, 250, 150, 260)
    assert sp.areas(rgb, [("missing plaster", m, 0.3)]) == []
    assert sp.areas(rgb, [("exposed concrete", m, 0.9)]) == []


def test_the_card_is_never_spalling():
    rgb = _wall()
    rgb[150:250, 150:260] = (30, 30, 35)
    m = _box(150, 250, 150, 260)
    assert sp.areas(rgb, [("missing plaster", m, 0.9)], exclude=m) == []
