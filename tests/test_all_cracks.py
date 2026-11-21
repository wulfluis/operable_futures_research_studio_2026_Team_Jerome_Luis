"""Every crack in the photo, not only the longest (D88): spec/features/all_cracks.md. Drawn images only."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.crack_view import view_data
from measure.crack import measure
from measure.cracks import combine, group_pieces, measure_all


def _mask(*boxes, shape=(400, 400)):
    m = np.zeros(shape, bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def test_far_apart_lines_are_two_cracks():
    groups = group_pieces(_mask((20, 380, 50, 53), (20, 380, 300, 303)), gap_px=20, min_extent_px=20)
    assert len(groups) == 2


def test_a_small_gap_stays_one_crack():
    groups = group_pieces(_mask((20, 190, 50, 53), (200, 380, 50, 53)), gap_px=20, min_extent_px=20)
    assert len(groups) == 1 and groups[0].sum() == (170 + 180) * 3


def test_a_speck_is_left_out_but_a_lone_small_crack_is_kept():
    groups = group_pieces(_mask((20, 380, 50, 53), (200, 203, 300, 303)), gap_px=20, min_extent_px=20)
    assert len(groups) == 1 and not groups[0][201, 301]
    assert len(group_pieces(_mask((200, 203, 300, 303)), gap_px=20, min_extent_px=20)) == 1


def _two_cracks():
    """A long thin crack and a short wide one, far apart, dark on a light wall (1 px = 1 mm)."""
    grey = np.full((400, 400), 200.0)
    grey[20:380, 99:101] = 40.0  # long, 2 px wide
    grey[150:250, 297:303] = 40.0  # short, 6 px wide
    mask = grey < 100
    found = measure_all(mask, lambda g: measure(grey, g, lambda p: np.asarray(p, float)), gap_px=20,
                        min_extent_px=20)
    return grey, found


def test_the_level_width_is_the_widest_over_all_cracks():
    _, c = _two_cracks()
    assert c.found and c.n_cracks == 2 and len(c.cracks) == 2
    widths = sorted(p.width_max_mm for p in c.cracks)
    assert c.width_max_mm == widths[-1] and widths[-1] > 2 * widths[0]  # the short, wide crack decides the width
    assert c.widest_point_px[0] > 250  # and its widest point lies on that crack
    assert c.length_main_mm > 300  # the length is the main (long) crack's
    assert c.angle_to_vertical_deg < 5  # and so is the angle
    assert c.length_total_mm > c.length_main_mm  # all cracks together


def test_one_crack_is_unchanged():
    grey = np.full((400, 400), 200.0)
    grey[20:380, 99:101] = 40.0
    one = measure(grey, grey < 100, lambda p: np.asarray(p, float))
    c = combine([one])
    assert c.n_cracks == 1 and c.width_max_mm == one.width_max_mm and c.length_main_mm == one.length_main_mm


def test_the_viewer_gets_every_crack_with_its_own_paths():
    _, c = _two_cracks()
    r = SimpleNamespace(crack=c, measured=True, pixel_crack=None, card=None, photo=Image.new("RGB", (400, 400)))
    d = view_data(r)
    assert len(d["cracks"]) == 2 and all(cr["paths"] for cr in d["cracks"])
    assert max(cr["max"] for cr in d["cracks"]) == round(c.width_max_mm, 4)
    assert d["widest"]["x"] > 0.6  # the widest marker sits on the short, wide crack (x about 300 / 400)
