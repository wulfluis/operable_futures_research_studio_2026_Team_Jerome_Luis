"""Which part the crack lies on, from SAM 3.1's masks of the element photo (spec/features/element_parts.md, D62).
SAM itself is not loaded (3.5 GB, gated): a fake segmenter returns masks as SAM 3.1 would."""

import numpy as np
import pytest
from PIL import Image

from app.core import Wall, analyse_element, run_measured
from element import parts
from element.parts import as_parts, find_parts, lies_on
from test_app_measure import FakeModel
from test_element_position import COLUMN, ENDS, CLOSER, _png, pair
from test_element_predict import FakeElementModel

H, W = 600, 800


def _box(x0, y0, x1, y1, shape=(H, W)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


WALL = _box(0, 0, W, H) & ~_box(350, 100, 450, 500)
COL = _box(350, 100, 450, 500)
SLAB = _box(0, 0, W, 60)


def _line(x0, y0, x1, y1, n=20):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def test_a_crack_inside_a_column_lies_on_the_column():
    p = lies_on([("column", COL, 0.9), ("wall", WALL, 0.8)], _line(380, 250, 420, 300), (H, W))
    assert p.on == "column" and p.border_with is None and p.share_of_crack > 0.9
    assert "column" in p.reason and p.found == {"column": 1, "wall": 1}


def test_a_crack_along_two_parts_is_on_their_border():
    p = lies_on([("column", COL, 0.9), ("wall", WALL, 0.8)], _line(400, 200, 500, 200), (H, W))
    assert {p.on, p.border_with} == {"column", "wall"} and "border" in p.reason


def test_a_crack_off_every_part_is_not_clear():
    p = lies_on([("column", COL, 0.9)], _line(50, 550, 150, 580), (H, W))
    assert p.on is None and "none of the parts" in p.reason


def test_without_the_link_the_crack_is_not_placed_but_the_parts_are_listed():
    p = lies_on([("wall", WALL, 0.8), ("slab", SLAB, 0.95)], None, (H, W))
    assert p.on is None and "not linked" in p.reason and "wall (1)" in p.reason and "slab (1)" in p.reason


def test_two_walls_count_together_and_two_words_for_one_thing_count_once():
    left, right = _box(0, 0, 400, H), _box(400, 0, W, H)
    p = lies_on([("wall", left, 0.8), ("wall", right, 0.7)], _line(300, 300, 500, 300), (H, W))
    assert p.on == "wall" and p.border_with is None and p.share_of_crack == pytest.approx(1.0)
    kept = as_parts([("column", COL, 0.9), ("concrete column", COL, 0.7), ("concrete wall", WALL, 0.6)])
    assert [k[0] for k in kept] == ["column", "wall"] and kept[0][2] == 0.9


def test_a_weak_find_is_not_used():
    """A mirror was read as a window at 0.41 (11 Nov): below MIN_CONF nothing is shown or used."""
    seg = lambda rgb: [("window", COL, 0.41), ("concrete wall", WALL, 0.8)]  # noqa: E731
    p = find_parts(np.zeros((H, W, 3), np.uint8), _line(380, 250, 420, 300), seg)
    assert p.on is None and "window" not in p.found and p.preview is not None


def test_the_answer_says_where_it_comes_from_and_never_safe():
    d = lies_on([("wall", WALL, 0.8)], _line(100, 300, 200, 350), (H, W)).to_dict()
    assert d["name"] == "Wall" and d["note"] and "SAM 3.1" in d["source"] and "level does not use it" in d["source"]
    assert "safe" not in str(d).lower()


def test_without_the_weights_the_app_runs_without_sam(monkeypatch, tmp_path):
    monkeypatch.setattr(parts, "WEIGHTS", tmp_path / "missing.pt")
    assert parts.load_parts_model() is None


def _fake_sam(rgb):
    """SAM's answer for the synthetic element photo: the column (from test_element_position) and the wall around."""
    h, w = rgb.shape[:2]
    col = _box(COLUMN[0] // CLOSER, ENDS[0] // CLOSER, COLUMN[1] // CLOSER, ENDS[1] // CLOSER, (h, w))
    return [("column", col, 0.92), ("concrete wall", ~col, 0.7)]


def _through_the_app(parts_model):
    close, _, element = pair(1200, card=True, plain=True)
    r = run_measured(Image.fromarray(np.repeat(close[..., None], 3, axis=2)), FakeModel(), card_placed=True)
    analyse_element(_png(element), r, Wall(), FakeElementModel("column"), parts_model=parts_model)
    return r


def test_through_the_app_the_crack_lies_on_the_column_and_the_level_is_unchanged():
    without, with_sam = _through_the_app(None), _through_the_app(_fake_sam)
    assert without.parts is None
    assert with_sam.parts["on"] == "column" and with_sam.parts_preview is not None
    assert "parts" in with_sam.to_dict()
    assert with_sam.triage == without.triage  # shown only: level, element, position and angle stay as they were
    assert with_sam.location == without.location


def test_if_sam_fails_the_app_goes_on():
    def broken(rgb):
        raise RuntimeError("MPS out of memory")
    r = _through_the_app(broken)
    assert r.parts["on"] is None and "could not run" in r.parts["reason"] and r.triage is not None
