"""D111 (a spall is not a crack) and D112 (exposed steel is Category 3). No SAM or model file is needed."""

import numpy as np

from anomaly import spalling_sam
from app.core import MeasuredResult, _cut_lines, steel_seen
from triage.matrix import assess


def test_exposed_steel_is_category_3_whatever_the_crack():
    r = assess("none", "unknown", float("nan"), float("nan"), True, exposed_steel=True)
    assert r.level == "Category 3" and r.engineer_review and "steel" in r.reason.lower()


def test_an_unusable_photo_stays_a_retake_even_with_steel():
    assert assess("none", "unknown", float("nan"), 0.9, False, exposed_steel=True).level == "Retake"


def test_without_steel_the_matrix_is_as_before():
    assert assess("none", "unknown", float("nan"), 0.9, True).level == "Expert needed"


def test_steel_on_either_photo_counts():
    r = MeasuredResult("Crack found", "x", 0.9, None)
    assert not steel_seen(r)
    r.steel_element = {"seen": True}
    assert steel_seen(r)
    r2 = MeasuredResult("Crack found", "x", 0.9, None)
    r2.exposed_steel = {"seen": True}
    assert steel_seen(r2)


def test_crack_lines_are_cut_where_they_cross_a_spall():
    cut = np.zeros((100, 100), bool)
    cut[40:60, :] = True
    line = np.array([[50, y] for y in range(0, 100)], float)
    pieces = _cut_lines([line], cut)
    assert len(pieces) == 2
    assert all(not cut[int(p[1]), int(p[0])] for piece in pieces for p in piece)


def _sam(mask, word="missing plaster", score=0.9):
    return [(word, mask, score)]


def test_a_spall_the_crack_model_marked_as_a_wide_crack_is_still_found():
    rgb = np.full((400, 400, 3), (60, 90, 200), np.uint8)  # blue paint
    rgb[100:300, 150:250] = (190, 180, 160)  # the broken-away area shows the light material below
    area = np.zeros((400, 400), bool)
    area[100:300, 150:250] = True
    crack = area.copy()  # the crack model marked the whole spall as crack
    found = spalling_sam.areas(rgb, _sam(area), crack)
    assert found and found[0]["share"] > 0.1


def test_an_open_cracks_dark_gap_is_not_a_spall():
    rgb = np.full((400, 400, 3), 200, np.uint8)
    rgb[50:350, 190:215] = 30  # a long, thin, very dark gap
    gap = np.zeros((400, 400), bool)
    gap[50:350, 190:215] = True
    assert spalling_sam.areas(rgb, _sam(gap)) == []


# ---------- D113: the crack model's wide, not-dark areas are patches; no width guessed for real photos ----------

def test_a_wide_light_blob_the_crack_model_painted_is_a_patch():
    rgb = np.full((500, 500, 3), (70, 110, 70), np.uint8)  # a painted wall
    rgb[150:320, 150:330] = (180, 175, 165)  # broken plaster shows the lighter material below
    crack = np.zeros((500, 500), bool)
    crack[150:320, 150:330] = True
    crack[320:480, 238:242] = True  # the real crack line leaving it
    p = spalling_sam.wide_patches(rgb, crack)
    assert len(p) == 1 and p[0]["kind"] == "spalling"


def test_a_wide_dark_gap_stays_a_crack():
    rgb = np.full((500, 500, 3), 200, np.uint8)
    rgb[150:320, 200:260] = 25  # an open, dark gap
    crack = np.zeros((500, 500), bool)
    crack[150:320, 200:260] = True
    assert spalling_sam.wide_patches(rgb, crack) == []


def test_a_thick_band_along_a_thin_crack_stays_a_crack():
    rgb = np.full((500, 500, 3), 200, np.uint8)
    rgb[20:480, 249:251] = 120  # a faint crack ...
    crack = np.zeros((500, 500), bool)
    crack[20:480, 235:265] = True  # ... the crack model marks as a thick band
    assert spalling_sam.wide_patches(rgb, crack) == []


def test_real_photos_get_no_width_guessed_from_photo_2():
    r = MeasuredResult("Crack found", "x", 0.9, None)
    r.test, r.real_photos = True, True
    assert r.real_photos and r.estimate is None
