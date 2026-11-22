"""D97: a crack line on the element photo is traced whole, never cut into dashes; the element's straight edge is not."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from element import photo2  # noqa: E402


def _drawn(lines, size=(400, 300), width=12):
    im = Image.new("L", size, 0)
    d = ImageDraw.Draw(im)
    for ln in lines:
        d.line(ln, fill=255, width=width)
    return np.asarray(im) > 0


def test_an_x_is_four_long_arms():
    m = _drawn([[(10, 10), (390, 290)], [(10, 290), (390, 10)]])
    nets = photo2.trace_network(m, 12)
    long = [b for b in nets if photo2._length(b) > 100]
    assert len(long) == 4
    assert sum(photo2._length(b) for b in long) > 0.9 * sum(photo2._length(b) for b in nets)


def test_a_wandering_thick_crack_stays_one_line():
    pts = [(10 + 38 * i, 150 + (25 if i % 2 else -25)) for i in range(11)]
    m = _drawn([pts], width=14)
    nets = photo2.trace_network(m, 12)
    total = sum(photo2._length(b) for b in nets)
    assert max(photo2._length(b) for b in nets) > 0.85 * total  # not dashes


def test_specks_are_dropped():
    m = _drawn([[(10, 150), (390, 150)]], width=6)
    m[20:23, 20:23] = True
    nets = photo2.trace_network(m, 12)
    assert len(nets) == 1


def test_the_elements_straight_edge_is_not_a_crack():
    region = np.zeros((300, 400), bool)
    region[20:280, 100:300] = True
    edge = np.array([[300.0, y] for y in range(30, 270)])
    crack = np.array([[150 + 20 * np.sin(y / 15), y] for y in range(30, 270)], float)
    assert photo2.outline_edge(edge, region)
    assert not photo2.outline_edge(crack, region)


def test_photo1s_crack_is_kept_even_along_the_edge():
    region = np.zeros((300, 400), bool)
    region[20:280, 100:300] = True
    edge = np.array([[300.0, y] for y in range(30, 270)])
    seed = np.zeros_like(region)
    seed[:, 295:306] = True
    assert not photo2.outline_edge(edge, region, seed)
