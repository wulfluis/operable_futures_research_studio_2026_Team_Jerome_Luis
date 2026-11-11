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


# ---------- straight edges across the photo are not cracks (D51) ----------

def _wavy(prob, y0=200, amp=15, period=30, p=0.9):
    """A crack that wanders up and down as it runs across the whole photo."""
    for x in range(prob.shape[1]):
        y = int(round(y0 + amp * np.sin(x / period)))
        prob[y - 1:y + 2, x] = p
    return prob


def test_wall_ceiling_line_across_the_photo_is_no_crack():
    prob = np.full((416, 416), 0.05)
    prob[120:124, :] = 0.95  # a straight dark line from the left side to the right side
    d = decide(prob)
    assert d.statement == NO_CRACK and d.edges_ignored == 1
    assert not d.mask.any() and d.confidence == 0
    assert "corner or edge" in d.reason and "not checked" in d.reason


def test_wall_corner_top_to_bottom_is_no_crack():
    prob = np.full((416, 416), 0.05)
    prob[:, 300:306] = 0.9  # a vertical corner, thick
    assert decide(prob).statement == NO_CRACK


def test_tilted_edge_from_top_to_side_is_no_crack():
    prob = np.full((416, 416), 0.05)
    for y in range(0, 300):  # a straight diagonal from the top side to the right side
        x = 116 + y
        prob[y, x - 1:x + 2] = 0.9
    assert decide(prob).statement == NO_CRACK


def test_wall_ceiling_line_ending_at_the_room_corner_is_no_crack():
    """As in the app on 10 Nov: from the top side down to the room corner inside the photo, then a short bend down."""
    prob = np.full((832, 624), 0.05)
    for y in range(0, 600):  # the straight wall-ceiling line, from the top side
        x = 160 + int(0.3 * y)
        prob[y, x - 2:x + 3] = 0.9
    prob[600:640, 338:343] = 0.9  # the short vertical corner below it, inside the photo
    d = decide(prob, scale=832 / 416)
    assert d.statement == NO_CRACK and d.edges_ignored == 1


def test_straight_line_from_one_side_but_short_is_kept():
    prob = np.full((416, 416), 0.05)
    prob[200:203, 0:120] = 0.9  # reaches the left side, but shorter than 40 % of the photo
    assert decide(prob).edges_ignored == 0


def test_edge_works_on_the_app_photo_size():
    prob = np.full((624, 832), 0.05)
    prob[400:406, :] = 0.9
    assert decide(prob, scale=832 / 416).statement == NO_CRACK


def test_edge_plus_a_wavy_crack_finds_the_crack_only():
    prob = np.full((416, 416), 0.05)
    prob[20:24, :] = 0.95  # the wall-ceiling line
    prob = _wavy(prob, y0=250)
    d = decide(prob)
    assert d.statement == CRACK and d.edges_ignored == 1
    assert not d.mask[20:24, :].any()  # the edge is not drawn
    assert d.mask[230:271, :].any()  # the crack is
    assert "corner or edge" in d.reason


def test_wavy_crack_across_the_whole_photo_is_still_a_crack():
    d = decide(_wavy(np.full((416, 416), 0.05)))
    assert d.statement == CRACK and d.edges_ignored == 0
    assert "corner or edge" not in d.reason


def test_straight_line_inside_the_photo_is_still_a_crack():
    d = decide(_line(200))  # straight, but it does not reach the sides of the photo
    assert d.statement == CRACK and d.edges_ignored == 0


def test_short_straight_piece_across_a_photo_corner_is_kept():
    prob = np.full((416, 416), 0.05)
    for i in range(110):  # touches the top and the left side, but it is short
        prob[i, 109 - i] = 0.9
        prob[i, max(0, 108 - i)] = 0.9
    assert decide(prob).edges_ignored == 0


def _wavy_down(prob, x0, y_from, y_to, amp=12, period=25, p=0.9):
    """A crack that wanders left and right as it runs down."""
    for y in range(y_from, y_to):
        x = int(round(x0 + amp * np.sin((y - y_from) / period)))
        prob[y, x - 1:x + 2] = p
    return prob


def test_crack_leaving_the_edge_is_kept_and_the_edge_dropped():
    prob = np.full((416, 416), 0.05)
    prob[20:24, :] = 0.95  # the wall-ceiling line
    prob = _wavy_down(prob, 200, 24, 300)  # a crack running down from it: one group with the edge
    d = decide(prob)
    assert d.statement == CRACK and d.edges_ignored == 1
    assert not d.mask[20:24, :100].any()  # the edge is not drawn
    assert d.mask[150:300, :].any()  # the crack is


def test_straight_stub_hanging_from_the_edge_goes_with_it():
    prob = np.full((416, 416), 0.05)
    prob[20:24, :] = 0.95
    prob[24:120, 200:203] = 0.9  # a straight stub: where two walls meet below the ceiling
    assert decide(prob).statement == NO_CRACK


def test_pixel_scores_and_rates():
    truth = np.zeros((10, 10), bool); truth[5, :] = True
    pred = np.zeros((10, 10), bool); pred[5, :5] = True; pred[0, 0] = True
    c = pixel_scores(pred, truth)
    assert c == dict(tp=5, fp=1, fn=5)
    r = rates(c)
    assert r["precision"] == pytest.approx(5 / 6) and r["recall"] == pytest.approx(0.5) and r["iou"] == pytest.approx(5 / 11)
    assert rates(dict(tp=0, fp=0, fn=0))["f1"] == 0


# ---------- a crack is a line: thick round patches are not cracks (D54) ----------

def _disk(prob, cy, cx, r, p=0.9):
    yy, xx = np.ogrid[:prob.shape[0], :prob.shape[1]]
    prob[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = p
    return prob


def test_elongation_of_a_line_and_a_patch():
    from segmentation.decide import elongation
    line = np.zeros((20, 220), bool); line[8:12, 10:210] = True
    disk = _disk(np.zeros((60, 60)), 30, 30, 20) > 0.5
    assert elongation(line) > 20 and elongation(disk) < 2


def test_large_patch_alone_goes_to_an_expert_never_no_crack():
    d = decide(_disk(np.full((416, 416), 0.05), 200, 200, 25))  # about 2,000 px: a dark object, or spalling
    assert d.statement == UNSURE and d.patch_only and "patch" in d.reason and "spalling" in d.reason
    assert d.mask.any()  # drawn, so the expert sees what was marked


def test_patch_next_to_a_crack_is_not_counted_or_drawn():
    prob = _disk(np.full((416, 416), 0.05), 80, 330, 25)
    prob[300, 50:350] = 0.9  # a thin crack
    d = decide(prob)
    assert d.statement == CRACK and d.patches == 1 and not d.mask[80, 330] and d.mask[300, 200]
    assert "not line-shaped" in d.reason


def test_small_patch_alone_is_no_crack():
    d = decide(_disk(np.full((416, 416), 0.05), 200, 200, 6))  # about 110 px, under the crack size
    assert d.statement in (NO_CRACK, UNSURE)
    assert not (d.statement == CRACK)


def test_a_wide_crack_is_still_a_crack():
    """The widest cracks matter most (Category 3): width alone never makes a patch."""
    prob = np.full((416, 416), 0.05)
    prob[190:210, 60:360] = 0.9  # 20 px wide, 300 px long
    d = decide(prob)
    assert d.statement == CRACK and d.patches == 0


def test_a_wavy_wide_crack_with_a_wide_spot_is_still_a_crack():
    prob = _wavy(np.full((416, 416), 0.05), y0=200, amp=12)
    prob = _disk(prob, 200, 150, 12)  # one wide spot along it
    assert decide(prob).statement == CRACK
