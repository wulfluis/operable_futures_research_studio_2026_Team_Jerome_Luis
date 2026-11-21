"""The drawn showcase cases (D90): every photo is drawn, with a bank card, and its cracks wander (D51)."""
import numpy as np

from app import showcase


def test_every_case_is_a_drawn_colour_photo():
    cases = list(showcase.cases())
    assert len(cases) == len(showcase.CASES) >= 6
    for img, element, index, name in cases:
        assert img.dtype == np.uint8 and img.shape == (900, 1200, 3) and element in {"wall", "column", "beam"}


def test_a_drawn_crack_wanders_but_keeps_its_ends():
    line = showcase.wander([(0, 0), (0, 100)], seed=1)
    xs = np.array([p[0] for p in line])
    assert line[0] == (0.0, 0.0) and line[-1] == (0.0, 100.0)
    assert np.abs(xs).max() > 0.3  # not a straight line, which the app would read as an edge


def test_the_card_is_drawn_light_and_the_cracks_dark():
    img = showcase.draw([([(150, 10), (150, 140)], 2.0)], seed=0)
    grey = img.mean(axis=2)
    assert grey.max() > 210 and grey.min() < 80
