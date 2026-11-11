"""Crack type and location in the app, and the automatic risk level (spec/features/app_classification.md, D53, D55)."""

import io
import re

import pytest
from PIL import Image

from app.core import POSITIONS, WAITING, PhotoRejected, Wall, analyse_element, auto_triage, level_for, run_measured
from test_app_measure import DEMO_MODEL, FakeModel, _demo, _plain_wall_with_line
from test_element_predict import FakeElementModel
from triage.element_rule import ElementAnswer

LEVELS = ["Category 1", "Category 2", "Category 3"]


@pytest.fixture
def shared_result():
    """A measured crack (bank card) on the wall."""
    photo, _ = _demo(1.0)
    r = run_measured(photo, DEMO_MODEL, card_placed=True)
    wall = Wall()
    r.wall_number = wall.add(r.statement, r.overlay, r.to_dict())
    return r, wall


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def test_no_level_before_the_element_photo(shared_result):
    r, _ = shared_result
    assert auto_triage(r) is None and r.to_dict()["triage"] is None
    assert "photo of the whole element" in WAITING


def test_the_rules_suggest_a_type_once_the_element_is_known(shared_result):
    r, _ = shared_result
    assert r.suggested_pattern().pattern is None  # no element yet: rule 4, element unknown
    r.element = ElementAnswer("beam", False, 0.92, "")
    s = r.suggested_pattern()
    assert s is not None and s.rule != 4 and r.to_dict()["pattern_suggestion"]["rule"] == s.rule


def test_element_photo_gives_the_level_by_itself_and_the_wall_shows_it(shared_result):
    r, wall = shared_result
    a = analyse_element(_jpeg(_demo(1.0)[0]), r, wall, FakeElementModel("beam"))
    assert a.element == "beam" and r.triage is not None
    t = wall.get(r.wall_number).stats["triage"]
    assert t["level"] == r.triage["level"] and t["element_name"] == "beam" and "automatic" in t["source"]
    assert t["position"] == "unknown"  # not known from the photos: counts as an end zone


def test_a_clear_type_and_a_confident_element_give_a_category(shared_result):
    r, _ = shared_result
    r.element = ElementAnswer("beam", False, 0.95, "")
    s = r.suggested_pattern()
    level = auto_triage(r)
    if s.clear:
        assert level.level in LEVELS + ["Expert needed"] and r.triage["pattern"] == s.pattern
    else:
        assert level.level == "Expert needed" and r.triage["pattern"] is None


def test_unclear_type_is_never_a_category(shared_result):
    """A vertical crack in a column without rust can be crushing (rule 14, unclear): an engineer decides."""
    r, _ = shared_result
    r.element = ElementAnswer("column", False, 0.95, "")
    s = r.suggested_pattern()
    level = auto_triage(r)
    if not s.clear:
        assert level.level == "Expert needed" and level.engineer_review


@pytest.mark.parametrize("answer", ["unsure", "others"])
def test_no_element_is_expert_needed_with_the_model_reason(shared_result, answer):
    r, _ = shared_result
    r.element = ElementAnswer(answer, answer == "unsure", 0.5, "The model is unsure (wall, 50%): a person decides.")
    level = auto_triage(r)
    assert level.level == "Expert needed" and level.engineer_review
    assert r.triage["element"] is None and "unsure" in r.triage["why"]


def test_without_a_card_it_is_expert_needed_and_says_why():
    r = run_measured(_plain_wall_with_line(), FakeModel())  # no bank card: no width, no direction
    r.element = ElementAnswer("beam", False, 0.95, "")
    level = auto_triage(r)
    assert level.level == "Expert needed" and "bank card" in r.triage["why"]


def test_the_element_models_doubt_lowers_the_confidence(shared_result):
    r, _ = shared_result
    r.element = ElementAnswer("beam", False, 0.81, "")
    auto_triage(r)
    assert r.triage["confidence"] == pytest.approx(min(r.confidence, 0.81), abs=1e-3)


def test_no_crack_gets_no_level():
    import numpy as np
    wall_only = np.random.default_rng(0).integers(110, 190, (900, 1200, 3)).astype(np.uint8)  # textured, no line
    r = run_measured(Image.fromarray(wall_only), FakeModel())
    r.element = ElementAnswer("beam", False, 0.95, "")
    assert r.statement == "No crack found" and auto_triage(r) is None


def test_element_photo_without_the_model_is_expert_needed(shared_result):
    r, wall = shared_result
    assert analyse_element(_jpeg(_demo(1.0)[0]), r, wall, None).element == "unsure"
    assert r.triage["level"] == "Expert needed"


def test_element_photo_must_be_a_photo(shared_result):
    r, wall = shared_result
    with pytest.raises(PhotoRejected):
        analyse_element(b"not a photo", r, wall, None)


def test_unknown_position_never_gives_a_lower_level_than_the_middle(shared_result):
    r, _ = shared_result
    for pattern, element in (("along_bar", "column"), ("diagonal_shear", "beam"), ("flexural", "beam")):
        unknown = level_for(r, element, "unknown", pattern)
        middle = level_for(r, element, "middle", pattern)
        assert LEVELS.index(unknown.level) >= LEVELS.index(middle.level)
        assert level_for(r, element, "joint", pattern).level == unknown.level  # a joint is an end zone


def test_every_position_is_a_matrix_location(shared_result):
    r, _ = shared_result
    for p in POSITIONS:
        assert level_for(r, "beam", p, "flexural").level


def test_category_3_and_expert_needed_say_an_engineer_must_confirm(shared_result):
    r, _ = shared_result
    for answer in ("column", "beam", "wall", "unsure"):
        r.element = ElementAnswer(answer, False, 0.95, "")
        level = auto_triage(r)
        assert level.engineer_review == (level.level in ("Category 3", "Expert needed"))


def test_never_says_safe(shared_result):
    r, wall = shared_result
    for answer in ("column", "beam", "wall", "others", "unsure"):
        r.element = ElementAnswer(answer, False, 0.95, "")
        auto_triage(r, wall)
        assert not re.search(r"\bsafe\b|clearance|permit", str(r.to_dict()), re.I)
