"""Tests for bank-card detection, tilt check, the one-photo analysis and the app (spec/features/measurement.md)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from measure.analyse import analyse
from measure.card import detect_card, from_corners, order_corners
from measure.synthetic import camera_homography, make_demo


def _rgb(grey):
    return np.repeat(np.clip(grey, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


@pytest.mark.parametrize("lr, ud", [(0, 0), (12, -6), (0, 18), (25, 8)])
def test_card_found_with_corners_and_tilt(lr, ud):
    grey, truth = make_demo(1.0, tilt_lr_deg=lr, tilt_ud_deg=ud)
    det = detect_card(_rgb(grey))
    assert det is not None
    assert np.abs(det.corners_px - order_corners(truth["card_corners_px"])).max() < 1.0  # sub-pixel corners
    assert det.tilt_deg == pytest.approx(truth["tilt_deg"], abs=1.5)


def test_tilt_status():
    shape = (900, 1200, 3)
    for tilt, status in ((0, "straight"), (15, "tilted"), (28, "retake")):
        H, _ = camera_homography((1200, 900), tilt, 0)
        q = np.c_[[(15, 60), (100.6, 60), (100.6, 113.98), (15, 113.98)], np.ones(4)] @ H.T
        assert from_corners(q[:, :2] / q[:, 2:], shape).status == status


def test_no_card_found():
    rng = np.random.default_rng(3)
    grey = 150 + rng.normal(0, 4, (600, 800))
    assert detect_card(_rgb(grey)) is None
    a = analyse(_rgb(grey))
    assert a.outcome == "Retake" and a.matrix_input() is None
    assert any("Bank card not found" in r for r in a.retake)


def test_square_is_not_a_card():
    grey = np.full((600, 800), 150.0)
    grey[200:400, 300:500] = 225  # a 1:1 square, e.g. a tile: a card would need over 50 degrees of tilt
    assert detect_card(_rgb(grey)) is None


def test_card_chosen_among_other_rectangles():
    grey, truth = make_demo(1.0, tilt_lr_deg=5)
    grey[20:80, 1000:1180] = 230  # a 3:1 bright strip (a label): not card-shaped
    det = detect_card(_rgb(grey))
    assert np.abs(det.corners_px - order_corners(truth["card_corners_px"])).max() < 1.0


def test_analysis_end_to_end_without_search_box():
    grey, truth = make_demo(1.0, tilt_lr_deg=12, tilt_ud_deg=-4)
    a = analyse(_rgb(grey))
    assert a.outcome == "Measured"
    assert a.crack.width_max_mm == pytest.approx(1.0, abs=a.crack.width_uncertainty_mm)
    assert a.crack.width_median_mm == pytest.approx(1.0, abs=0.06)
    assert a.crack.length_main_mm == pytest.approx(truth["length_mm"], rel=0.03)
    assert a.matrix_input()["width_mm"] == pytest.approx(a.crack.width_max_mm, abs=1e-3)
    assert any("tilted" in x for x in a.warnings)


def test_strong_tilt_is_a_retake():
    grey, _ = make_demo(1.0, tilt_lr_deg=30, tilt_ud_deg=10)
    a = analyse(_rgb(grey))
    assert a.outcome == "Retake" and a.matrix_input() is None  # rule 2: doubt never reaches the matrix


def test_report_never_says_safe():
    grey, _ = make_demo(1.0)
    text = json.dumps(analyse(_rgb(grey)).to_dict()).lower()
    assert not re.search(r"\bsafe\b|clearance|permit", text)  # AGENTS.md rule 1


def test_app_runs_on_demo_photo():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = Path(__file__).resolve().parents[1] / "src" / "measure" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Width, widest point"].endswith("mm") and labels["Camera tilt"].endswith("°")


def test_light_card_on_light_wall_with_a_finger_over_its_edge():
    """Field case of 4 Nov: an access card (ID-1 size) a little darker than the wall, a finger across the bottom
    edge, so there is no closed outline. The straight-edge fallback must still find it."""
    import cv2
    from measure.card import _line_quads, detect_card
    rng = np.random.default_rng(3)
    h, w = 1400, 1070
    img = np.full((h, w), 142.0) + rng.normal(0, 3, (h, w))
    corners = np.array([[386, 846], [736, 845], [746, 1081], [379, 1080]], np.float64)
    card = np.zeros((h, w), np.uint8)
    cv2.fillPoly(card, [corners.astype(np.int32)], 1)
    img[card > 0] = 118 + rng.normal(0, 3, int(card.sum()))
    cv2.ellipse(img, (560, 1250), (55, 200), 0, 0, 360, 175, -1)  # the finger, lighter than the card
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 1.2)
    d = detect_card(rgb)
    assert d is not None
    dist = np.linalg.norm(corners[:, None, :] - np.asarray(d.corners_px)[None], axis=2).min(axis=1)
    assert dist.max() < 4, dist  # every true corner has a found corner within 4 px
    small = cv2.resize(rgb, (round(w * 1000 / h), 1000), interpolation=cv2.INTER_AREA)
    assert _line_quads(small)  # the straight-edge search on its own finds it too


def test_no_card_is_invented_on_a_plain_wall_with_a_crack():
    from measure.card import detect_card
    rng = np.random.default_rng(4)
    img = np.full((900, 1200), 150.0) + rng.normal(0, 4, (900, 1200))
    img[440:446, 100:1100] = 60
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    assert detect_card(rgb) is None


def test_tilt_hint_says_which_way_to_straighten_the_phone():
    from measure.card import tilt_hint
    wide_bottom = [[300, 1030], [650, 1030], [667, 1260], [283, 1260]]
    assert "lower edge" in tilt_hint(wide_bottom) and "bring the top" in tilt_hint(wide_bottom)
    tall_left = [[100, 100], [400, 120], [400, 280], [100, 300]]
    assert "left edge" in tilt_hint(tall_left) and "bring the right" in tilt_hint(tall_left)
    assert tilt_hint([[0, 0], [856, 0], [856, 540], [0, 540]]) == ""  # straight: no hint


def test_card_wins_over_print_on_it_and_a_shadow_running_to_the_edge():
    """Field case of 4 Nov: a card-shaped print on the card (a logo box) and a dark shadow joined to the photo's
    left edge must not be taken for the card."""
    import cv2
    from measure.card import detect_card
    rng = np.random.default_rng(5)
    h, w = 1400, 1070
    img = np.full((h, w), 150.0) + rng.normal(0, 3, (h, w))
    cv2.fillPoly(img, [np.array([[0, 180], [70, 200], [60, 600], [0, 620]], np.int32)], 70)  # shadow at the edge
    corners = np.array([[60, 210], [654, 220], [650, 581], [56, 575]], np.float64)
    cv2.fillPoly(img, [corners.astype(np.int32)], 225)  # light card
    cv2.fillPoly(img, [np.array([[400, 470], [480, 470], [480, 520], [400, 520]], np.int32)], 40)  # logo box
    rgb = cv2.GaussianBlur(np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2), (5, 5), 1.2)
    d = detect_card(rgb)
    assert d is not None
    dist = np.linalg.norm(corners[:, None, :] - np.asarray(d.corners_px)[None], axis=2).min(axis=1)
    assert dist.max() < 4, dist
