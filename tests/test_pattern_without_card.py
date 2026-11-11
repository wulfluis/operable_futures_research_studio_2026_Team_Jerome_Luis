"""The crack type without a bank card: shape read in the photo's pixels (spec/features/pattern_rules.md, D57)."""

import re

import numpy as np
from PIL import Image

from app.core import Wall, auto_triage, run_measured
from segmentation.predict import run
from test_app_measure import DEMO_MODEL, FakeModel, _demo, _plain_wall_with_line
from triage.element_rule import ElementAnswer


def test_the_answer_without_a_card_is_still_the_original():
    photo, _ = _demo(1.0)
    mine, original = run_measured(photo, DEMO_MODEL), run(photo, DEMO_MODEL)
    assert mine.card is None and mine.crack is None and mine.pixel_shape is not None
    assert (mine.statement, mine.reason, mine.confidence) == (original.statement, original.reason, original.confidence)
    assert np.array_equal(np.asarray(mine.overlay), np.asarray(original.overlay))


def test_the_shape_in_pixels_agrees_with_the_shape_measured_with_the_card():
    photo, _ = _demo(1.0)
    no_card = run_measured(photo, DEMO_MODEL).pixel_shape
    card = run_measured(photo, DEMO_MODEL, card_placed=True).crack.to_dict()["shape"]
    assert abs(no_card["angle_to_vertical_deg"] - card["angle_to_vertical_deg"]) < 10  # tilt of the demo about 9 deg
    assert abs(no_card["straightness"] - card["straightness"]) < 0.1
    assert no_card["n_loops"] == card["n_loops"] and "no bank card" in no_card["source"]


def test_a_horizontal_crack_in_a_column_gets_its_type_but_no_category():
    r = run_measured(_plain_wall_with_line(), FakeModel())
    assert r.statement == "Crack found" and r.pixel_shape["angle_to_vertical_deg"] > 80
    r.element = ElementAnswer("column", False, 0.95, "")
    s = r.suggested_pattern()
    assert s.pattern == "column_horizontal" and s.clear
    level = auto_triage(r)
    assert level.level == "Expert needed" and "width" in level.reason.lower()  # the matrix needs the width
    assert r.triage["pattern"] == "column_horizontal" and "bank card" in r.triage["why"]
    assert "not clear" not in r.triage["why"]


def test_without_a_card_the_level_is_never_a_category():
    for element in ("column", "beam", "load_bearing_wall"):
        for photo, model in ((_plain_wall_with_line(), FakeModel()), (_demo(1.0)[0], DEMO_MODEL)):
            r = run_measured(photo, model)
            r.element = ElementAnswer(element.replace("load_bearing_", ""), False, 0.99, "")
            level = auto_triage(r)
            assert level.level == "Expert needed" and level.engineer_review


def test_no_shape_without_a_crack():
    a = np.asarray(_plain_wall_with_line()).copy()
    a[440:460] = a[400:420]  # the same textured wall, without the dark line
    r = run_measured(Image.fromarray(a), FakeModel())
    assert r.statement == "No crack found" and r.pixel_shape is None and r.suggested_pattern() is None


def test_the_wall_keeps_the_shape_and_never_says_safe():
    r = run_measured(_plain_wall_with_line(), FakeModel())
    wall = Wall()
    n = wall.add(r.statement, r.overlay, r.to_dict())
    d = wall.get(n).stats
    assert d["shape_in_pixels"]["angle_to_vertical_deg"] > 80 and not d["measured"]
    assert not re.search(r"\bsafe\b|clearance|permit", str(d), re.I)
