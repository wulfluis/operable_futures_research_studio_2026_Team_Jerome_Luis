"""Real two-photo test cases (spec/features/photo2.md, D62, D63): the close-up is cut around photo 2's crack, the width
is only ever estimated for TEST cases, and both photos reach the wall. No dataset or model file is needed."""

import json

import numpy as np
import pytest
from PIL import Image

from app import pick_cases, simulate
from app.core import Wall, auto_triage, estimate_from_photo2, run_measured
from element.photo2 import CrackOnElement
from test_app_measure import FakeModel, _plain_wall_with_line
from triage.element_rule import ElementAnswer


def _no_card_result(test: bool):
    r = run_measured(_plain_wall_with_line(), FakeModel())  # no card: nothing measured
    r.test = test
    r.element = ElementAnswer("column", False, 0.95, "")
    r.on_element = CrackOnElement(True, [[0.5, 0.2], [0.5, 0.8]], 0.6, (0.5, 0.5), size_px=(624, 832),
                                  width_px_max=10.0, width_px_median=8.0, length_px=500.0,
                                  shape={"angle_to_vertical_deg": 85.0, "straightness": 0.9, "coherence": 0.9,
                                         "n_loops": 0, "branch_ratio": 1.0})
    return r


def test_the_close_up_box_is_centred_on_photo_2_crack_and_inside_the_photo():
    box = pick_cases.crop_box((300, 400, 340, 500), (624, 832), (2848, 4288))  # crack found on the working copy
    x0, y0, x1, y1 = box
    side = pick_cases.CROP_SIDE * 2848
    assert x1 - x0 == pytest.approx(side, abs=1) and y1 - y0 == pytest.approx(side, abs=1)
    assert x0 <= 320 / 624 * 2848 <= x1 and y0 <= 450 / 832 * 4288 <= y1
    edge = pick_cases.crop_box((0, 0, 10, 10), (624, 832), (2848, 4288))
    assert edge[0] == 0 and edge[1] == 0  # never outside the photo


def test_the_width_is_estimated_only_for_test_cases():
    assert estimate_from_photo2(_no_card_result(test=True)) is not None
    real = _no_card_result(test=False)
    real.estimate = estimate_from_photo2(real) if real.test else None
    assert real.estimate is None
    level = auto_triage(real)
    assert level.level == "Expert needed" and "bank card" in real.triage["why"]  # a real upload needs the card


def test_the_estimate_scales_with_the_assumed_element_and_says_so():
    r = _no_card_result(test=True)
    e = estimate_from_photo2(r)
    assert e["mm_per_px"] == pytest.approx(3000 / 832, abs=0.01)  # a 3 m column filling 832 px
    assert e["width_max_mm"] == pytest.approx(10 * 3000 / 832, abs=0.1)
    assert "not measured" in e["basis"] and "TEST" in e["basis"]


def test_a_test_case_gets_its_level_from_both_photos():
    r = _no_card_result(test=True)
    r.estimate = estimate_from_photo2(r)
    level = auto_triage(r)
    assert r.triage["pattern"] == "column_horizontal"  # the shape came from photo 2: the same crack
    assert level.level == "Category 3" and "not measured" in r.triage["why"]


def test_both_photos_reach_the_wall():
    wall = Wall()
    r = _no_card_result(test=True)
    n = wall.add(r.statement, r.overlay, r.to_dict())
    assert wall.set_photo2(n, Image.new("RGB", (300, 400), (90, 90, 90)))
    assert wall.get(n).photo2[:2] == b"\xff\xd8"  # a JPEG
    assert not wall.set_photo2(999, Image.new("RGB", (10, 10)))


def test_real_cases_cut_photo_1_from_photo_2(tmp_path, monkeypatch):
    big = np.random.default_rng(0).integers(0, 255, (1200, 900, 3), dtype=np.uint8)
    Image.fromarray(big).save(tmp_path / "a.jpg")
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"cases": [{"photo": str(tmp_path / "a.jpg"), "crop": [100, 200, 400, 500],
                                            "element": "column"}]}))
    monkeypatch.setattr(simulate, "PICKS", picks)
    monkeypatch.setattr(simulate, "ROOT", tmp_path.parent)  # an absolute path in the picks is kept as it is
    close_up, whole, meta = next(simulate.real_cases(1))
    assert close_up.size == (300, 300) and whole.size == (900, 1200) and meta["card_placed"] is False
    assert "same crack in both photos" in meta["kind"]
