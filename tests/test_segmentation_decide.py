import numpy as np
import pytest

from segmentation.decide import CRACK, NO_CRACK, UNSURE, decide, pixel_scores, rates


def _line(length, p=0.9, size=416):
    prob = np.full((size, size), 0.05)
    prob[200, 100:100 + length] = p
    return prob


def test_long_line_is_a_crack_with_its_confidence():
    d = decide(_line(200))
    assert d.statement == CRACK and d.largest_group == 200
    assert d.confidence == pytest.approx(0.9)
    assert d.reason


def test_blank_map_is_no_crack():
    d = decide(np.full((416, 416), 0.1))
    assert d.statement == NO_CRACK and d.confidence == 0 and not d.mask.any()


def test_scattered_specks_are_no_crack():
    prob = np.full((416, 416), 0.05)
    prob[::40, ::40] = 0.99  # many single pixels, no line
    assert decide(prob).statement == NO_CRACK


def test_short_line_goes_to_expert():
    assert decide(_line(100)).statement == UNSURE


def test_thresholds_grow_with_the_photo():
    assert decide(_line(200, size=832), scale=2).statement == UNSURE  # same pixels, twice the photo: shorter crack
    assert decide(_line(400, size=832), scale=2).statement == CRACK


def test_diagonal_pixels_count_as_one_line():
    prob = np.full((416, 416), 0.05)
    idx = np.arange(160)
    prob[idx + 100, idx + 100] = 0.8
    assert decide(prob).largest_group == 160


def test_pixel_scores_and_rates():
    truth = np.zeros((10, 10), bool); truth[5, :] = True
    pred = np.zeros((10, 10), bool); pred[5, :5] = True; pred[0, 0] = True
    c = pixel_scores(pred, truth)
    assert c == dict(tp=5, fp=1, fn=5)
    r = rates(c)
    assert r["precision"] == pytest.approx(5 / 6) and r["recall"] == pytest.approx(0.5) and r["iou"] == pytest.approx(5 / 11)
    assert rates(dict(tp=0, fp=0, fn=0))["f1"] == 0
