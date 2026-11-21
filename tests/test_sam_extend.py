"""SAM 3.1 extends the crack model's crack, never decides (D91): spec/features/sam_vision.md. No SAM needed."""
import numpy as np
from PIL import Image

from segmentation import sam_extend
from segmentation.decide import decide, grown


def _box(y0, y1, x0, x1, shape=(200, 200)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_a_connected_sam_line_is_added_and_a_far_one_is_not():
    model = _box(20, 100, 50, 53)
    sam = _box(98, 180, 50, 54) | _box(20, 100, 150, 154)  # a continuation, and a line far away
    out = sam_extend.extend(model, sam, join_share=0.02)
    assert out[150, 51] and not out[50, 151]


def test_no_model_crack_means_sam_adds_nothing():
    assert not sam_extend.extend(np.zeros((50, 50), bool), _box(0, 50, 10, 12, (50, 50))).any()


def test_sam_missing_or_failing_leaves_the_model_crack(monkeypatch):
    model = _box(20, 100, 50, 53)
    monkeypatch.setattr(sam_extend, "sam_cracks", lambda photo: None)
    assert (sam_extend.extended(Image.new("RGB", (200, 200)), model) == model).all()

    def boom(photo):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(sam_extend, "sam_cracks", boom)
    assert (sam_extend.extended(Image.new("RGB", (200, 200)), model) == model).all()


def test_faint_pixels_count_only_when_joined_to_a_confident_crack():
    prob = np.zeros((100, 100))
    prob[10:60, 20:23] = 0.9   # confident crack
    prob[60:90, 20:23] = 0.3   # its faint continuation
    prob[10:40, 70:73] = 0.3   # a faint line on its own
    g = grown(prob, 0.25)
    assert g[75, 21] and not g[20, 71]
    assert decide(prob, low=0.25).mask[75, 21]


def test_a_long_straight_edge_is_dropped_from_sams_lines_and_a_wandering_crack_kept():
    m = np.zeros((400, 400), bool)
    m[:, 10:14] = True  # a door frame: straight, the whole height, at the side
    ys = np.arange(40, 360)
    xs = (200 + 12 * np.sin(ys / 15.0)).astype(int)  # a crack that wanders
    for dx in range(3):
        m[ys, xs + dx] = True
    out = sam_extend.without_edges(m)
    assert not out[200, 11] and out[200, xs[160] + 1]


def test_every_sam_line_counts_once_the_model_found_a_crack(monkeypatch):
    model = _box(20, 100, 50, 53)
    far = _box(20, 100, 150, 154)
    monkeypatch.setattr(sam_extend, "sam_cracks", lambda photo: far)
    out = sam_extend.extended(Image.new("RGB", (200, 200)), model)
    assert out[50, 151] == (not sam_extend.CONNECTED_ONLY)
    assert not sam_extend.extended(Image.new("RGB", (200, 200)), np.zeros((200, 200), bool)).any()


def test_the_whole_element_keeps_cracks_inside_its_outline_and_drops_the_rest():
    from element import photo2
    region = _box(0, 200, 0, 120)  # the element: the left part of the photo
    inside = _box(30, 170, 60, 63)
    outside = _box(30, 170, 170, 173)  # a crack-like line on something else
    nets = photo2.trace_whole_element(Image.new("RGB", (200, 200)), (200, 200), region,
                                      [np.array([[61.0, 100.0], [61.0, 120.0]])], inside | outside, np.zeros((200, 200), bool))
    xs = np.concatenate([b[:, 0] for b in nets])
    assert xs.min() >= 55 and xs.max() <= 70  # the crack inside is traced, the line outside is not
