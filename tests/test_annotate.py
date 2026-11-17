"""The crack's measurements on the photo itself (D80): only with a bank card, never on an unmeasured photo."""

import numpy as np
import pytest
from PIL import Image

from app import annotate
from app.core import run_measured
from measure.crack import CrackMeasurement
from test_app_measure import DEMO_MODEL, _demo


def _crack(**kw):
    base = dict(found=True, width_max_mm=0.452, width_uncertainty_mm=0.109, width_median_mm=0.34,
                length_main_mm=392.0, length_total_mm=392.0, angle_to_vertical_deg=28.0, mm_per_px=0.5,
                widest_point_px=(100, 60), main_path_px=np.array([[50, 20], [100, 60], [150, 180]], float))
    return CrackMeasurement(**{**base, **kw})


def test_the_instruments_carry_only_short_numbers():
    w = annotate.labels(_crack())
    assert w == {"width": "0.45 mm", "length": "392 mm", "angle": "28°"}  # the drawing does the explaining


@pytest.mark.parametrize("length, steps", [(60, (5, 25)), (392, (10, 50)), (2400, (50, 250))])
def test_the_ruler_ticks_suit_the_crack_length(length, steps):
    assert annotate.ruler_steps(length) == steps


@pytest.mark.parametrize("mm_per_px, width, expected", [(0.5, 400, 50), (0.1, 400, 10), (0.25, 1000, 50),
                                                         (0.001, 100, None)])
def test_the_scale_bar_is_a_round_length_that_fits(mm_per_px, width, expected):
    assert annotate.scale_bar_mm(mm_per_px, width) == expected


def test_the_instruments_are_drawn_only_when_measured():
    img = Image.new("RGB", (300, 240), "grey")
    full = Image.new("RGB", (300, 240), "grey")
    card = np.array([[20, 150], [110, 150], [110, 205], [20, 205]], float)
    c = _crack(widest_edges_px=np.array([[100, 58], [100, 62]], float), px_across=4.0)
    drawn = annotate.draw_measurements(img, 1.0, c, card, full)
    assert np.abs(np.asarray(drawn, int) - np.asarray(img, int)).sum() > 0
    no_loupe = annotate.draw_measurements(img, 1.0, c, card)  # without the full photo: ruler and bar, no loupe
    assert not np.array_equal(np.asarray(drawn), np.asarray(no_loupe))
    for none in (None, CrackMeasurement(False), _crack(width_max_mm=None)):
        assert np.array_equal(np.asarray(annotate.draw_measurements(img, 1.0, none, card, full)), np.asarray(img))


def test_a_measured_app_photo_carries_the_numbers_and_keeps_its_band():
    grey = _demo(1.0)[0]
    r = run_measured(grey, DEMO_MODEL, card_placed=True)
    assert r.measured
    h = r.overlay.info["photo_height"]  # the band under the photo is still there, for the sender (D78)
    assert r.overlay.height > h
    plain = run_measured(grey, DEMO_MODEL, card_placed=False)  # no card: nothing in mm on the photo
    assert not plain.measured and plain.overlay.info["photo_height"] == h
    diff = np.abs(np.asarray(r.overlay.crop((0, 0, r.overlay.width, h)), int)
                  - np.asarray(plain.overlay.crop((0, 0, plain.overlay.width, h)), int))
    assert diff.sum() > 0
