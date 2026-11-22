"""D99: opening and sliding read from the crack's two faces (method of [48], own code)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from measure.kinematics import from_path  # noqa: E402


def jagged(n=600, seed=3, rough=4.0):
    """A rough face: the height of the lower face along x."""
    rng = np.random.default_rng(seed)
    f = np.convolve(rng.normal(0, 1, n), np.ones(5) / 5, mode="same")  # jagged at a few pixels, as a crack face
    f += 0.05 * np.cumsum(rng.normal(0, 1, n))  # and wandering at a larger scale
    f = (f - f.mean()) / (f.std() + 1e-9) * rough
    return f


def crack(opening, sliding, rough=4.0, n=600, h=200):
    """A horizontal crack: the lower face y = 100 + f(x); the upper face is the lower one moved by (sliding, -opening)
    (image y down, so the upper face moved up by the opening). Returns the mask and the centre line."""
    f = jagged(n, rough=rough)
    xs = np.arange(n)
    lower = 100 + f
    upper = 100 + np.interp(xs - sliding, xs, f) - opening
    yy = np.arange(h)[:, None]
    mask = (yy >= upper[None, :]) & (yy <= lower[None, :])
    keep = slice(40, n - 40)
    path = np.column_stack([xs[keep], ((upper + lower) / 2)[keep]])
    mask[:, :40] = mask[:, n - 40:] = False
    return mask, path


@pytest.mark.parametrize("opening, sliding", [(6, 0), (6, 3), (8, 6)])
def test_a_known_displacement_is_read_back(opening, sliding):
    mask, path = crack(opening, sliding)
    k = from_path(mask, path)
    assert k.opening_median == pytest.approx(opening, abs=1.2)
    assert k.sliding_median is not None
    assert k.sliding_median == pytest.approx(sliding, abs=1.2)


def test_a_pure_opening_reads_mostly_opening():
    mask, path = crack(7, 0)
    assert from_path(mask, path).reading == "mostly opening"


def test_a_straight_smooth_crack_gives_no_sliding_never_zero():
    mask, path = crack(6, 0, rough=0.0)
    k = from_path(mask, path)
    assert k.sliding_median is None and "not readable" in k.reading


def test_the_scale_gives_millimetres():
    mask, path = crack(6, 3)
    px, mm = from_path(mask, path), from_path(mask, path, mm_per_px=0.1)
    assert mm.unit == "mm" and mm.opening_median == pytest.approx(px.opening_median * 0.1, rel=1e-6)


def test_too_thin_is_not_read():
    mask, path = crack(1.5, 0)
    assert from_path(mask, path).opening_median is None
