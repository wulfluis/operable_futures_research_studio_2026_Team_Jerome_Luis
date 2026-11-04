"""Tests for crack measurement (spec/features/measurement.md). Synthetic images with known answers."""

import json
import re

import numpy as np
import pytest

from measure.crack import measure, segment, skeletonize
from measure.measure_crack import run
from measure.reference import from_length, from_preset, from_rectangle, homography
from measure.synthetic import make_demo


def test_homography_recovers_known_mapping():
    H = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 12], [1e-4, 2e-4, 1]])
    src = np.array([(0, 0), (400, 10), (390, 300), (5, 310), (200, 150)], dtype=float)
    q = np.c_[src, np.ones(5)] @ H.T
    dst = q[:, :2] / q[:, 2:]
    assert np.allclose(homography(src, dst), H / H[2, 2], atol=1e-6)


def test_rectangle_scale_in_any_click_order():
    corners = [(100, 100), (100 + 856, 100), (100 + 856, 100 + 539.8), (100, 100 + 539.8)]  # 10 px per mm
    for order in (corners, corners[1:] + corners[:1], corners[::-1]):
        s = from_preset("card", order)
        a, b = s.to_mm([(200, 300), (500, 700)])
        assert np.linalg.norm(b - a) == pytest.approx(50.0, rel=1e-6)
        assert s.mm_per_px((300, 300)) == pytest.approx(0.1, rel=1e-6)


def test_bad_rectangle_is_refused():
    with pytest.raises(ValueError):
        from_rectangle([(0, 0), (100, 100), (100, 0), (0, 100)], (85.6, 53.98))  # crossed, not around
    with pytest.raises(ValueError):
        from_length((0, 0), (3, 0), 50)  # too short to trust


def test_length_scale():
    s = from_length((10, 10), (510, 10), 50)
    assert s.mm_per_px((0, 0), (0, 1)) == pytest.approx(0.1)
    assert s.rel_uncertainty == pytest.approx(np.sqrt(2) / 500)


def test_skeleton_of_a_bar_is_a_line():
    m = np.zeros((40, 120), dtype=bool)
    m[15:24, 10:110] = True
    sk = skeletonize(m)
    assert sk[:, 30:90].sum(axis=0).max() == 1  # one pixel thick
    assert sk[:, 20:100].any(axis=0).all()  # continuous


def _straight_crack(width_px, length_px=300, angle=0.0):
    """Dark straight crack on a grey wall, anti-aliased by supersampling."""
    h, w = 400, 500
    acc = np.zeros((h, w))
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    c, s = np.cos(angle), np.sin(angle)
    for oy in (0.25, 0.75):
        for ox in (0.25, 0.75):
            u = (xs + ox - 0.5 - 250) * c + (ys + oy - 0.5 - 200) * s
            v = -(xs + ox - 0.5 - 250) * s + (ys + oy - 0.5 - 200) * c
            acc += np.where((np.abs(v) <= width_px / 2) & (np.abs(u) <= length_px / 2), 50.0, 160.0)
    return acc / 4


@pytest.mark.parametrize("angle", [0.0, 0.5, np.pi / 4])
def test_straight_crack_width_and_length(angle):
    grey = _straight_crack(8.0, angle=angle)
    scale = from_length((0, 0), (500, 0), 50)  # 0.1 mm per px
    r = measure(grey, segment(grey), scale.to_mm, scale.rel_uncertainty)
    assert r.found
    assert r.width_median_mm == pytest.approx(0.8, abs=0.05)
    assert r.width_max_mm == pytest.approx(0.8, abs=0.1)
    assert r.length_main_mm == pytest.approx(30.0, rel=0.05)
    assert r.width_uncertainty_mm == pytest.approx(0.1, abs=0.01)
    assert r.matrix_input() is not None


def test_demo_with_perspective_matches_truth():
    grey, truth = make_demo(1.0)
    scale = from_preset("card", truth["card_corners_px"])
    r = measure(grey, segment(grey, truth["roi_px"]), scale.to_mm, scale.rel_uncertainty)
    assert r.width_max_mm == pytest.approx(1.0, abs=r.width_uncertainty_mm)
    assert r.width_median_mm == pytest.approx(1.0, abs=0.05)
    assert r.length_main_mm == pytest.approx(truth["length_mm"], rel=0.02)
    assert r.n_pieces == 1


def test_too_few_pixels_is_not_trusted():
    grey = _straight_crack(2.0)
    scale = from_length((0, 0), (500, 0), 50)
    r = measure(grey, segment(grey), scale.to_mm)
    assert r.found and r.px_across < 3
    assert r.matrix_input() is None  # rule 2: doubt goes to an expert, never to Category 1
    assert any("retake closer" in f for f in r.flags)


def test_no_crack_found():
    grey = np.full((200, 200), 150.0) + np.random.default_rng(1).normal(0, 3, (200, 200))
    scale = from_length((0, 0), (200, 0), 20)
    r = measure(grey, segment(grey), scale.to_mm)
    assert not r.found and r.width_max_mm is None and r.matrix_input() is None


def test_cli_demo_writes_report(tmp_path):
    report = run(["--demo", "--no-show", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "demo_1mm_measurement.json").read_text())
    assert saved["crack"]["width_max_mm"] == report["crack"]["width_max_mm"]
    assert saved["outcome"] == "Measured" and saved["card"]["status"] == "tilted"  # demo camera tilt 13 degrees
    assert (tmp_path / "demo_1mm_overlay.png").stat().st_size > 0
    assert not re.search(r"\bsafe\b|clearance|permit", json.dumps(saved).lower())  # AGENTS.md rule 1
