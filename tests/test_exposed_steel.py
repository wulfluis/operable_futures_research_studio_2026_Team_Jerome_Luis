"""D107: exposed steel on the close-up: the rule on SAM's answers, and what the app does when steel is seen."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from element.steel import decide  # noqa: E402


def _mask(share=0.05, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[: int(shape[0] * share * 10), :10] = True
    return m


def test_a_confident_bar_is_seen():
    out = decide([("rebar", _mask(), 0.8), ("exposed rebar", _mask(), 0.6)])
    assert out["seen"] and out["prompt"] == "rebar" and out["score"] == 0.8


def test_a_low_score_or_a_speck_is_not_seen():
    assert not decide([("rebar", _mask(), 0.45)])["seen"]
    speck = np.zeros((100, 100), bool)
    speck[0, 0] = True
    assert not decide([("rebar", speck, 0.9)])["seen"]


def test_steel_bar_answers_are_ignored():
    """'steel bar' also finds railings and gates."""
    assert not decide([("steel bar", _mask(), 0.95)])["seen"]


def test_no_answer_is_not_seen():
    assert decide([])["seen"] is False


def test_with_steel_seen_the_width_is_not_used_and_the_type_is_spalling_with_exposed_bar(monkeypatch):
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    monkeypatch.setattr(core, "_check_steel", lambda full: {"seen": True, "score": 0.8, "prompt": "rebar",
                                                             "defects": []})
    r = core.run_measured(photo, DEMO_MODEL, card_placed=True)
    assert r.statement == "Crack found"
    assert not r.measured and r.measure_note == core.STEEL_NOTE
    assert r.kinematics is None
    assert r.to_dict()["exposed_steel"] == {"seen": True, "score": 0.8, "prompt": "rebar", "causes": []}
    s = r.suggested_pattern()
    assert s is None or s.pattern == "spalling_exposed_bar"


def test_without_steel_nothing_changes(monkeypatch):
    from app import core
    from test_app_measure import DEMO_MODEL, _demo
    photo, _ = _demo(1.0)
    monkeypatch.setattr(core, "_check_steel", lambda full: {"seen": False, "score": None, "prompt": None})
    r = core.run_measured(photo, DEMO_MODEL, card_placed=True)
    assert r.measured and r.measure_note is None


def test_bare_steel_without_a_crack_line_is_never_no_crack_found(monkeypatch):
    """Rule 2: with no crack line but steel seen, a person decides."""
    from PIL import Image
    from app import core
    from test_app_measure import DEMO_MODEL
    rng = np.random.default_rng(0)
    wall = Image.fromarray(rng.integers(110, 190, (480, 640, 3), dtype=np.uint8))  # a textured wall, no crack line
    plain = core.run_measured(wall, DEMO_MODEL)
    assert plain.statement == "No crack found"
    monkeypatch.setattr(core, "_check_steel", lambda full: {"seen": True, "score": 0.9, "prompt": "rebar", "defects": []})
    r = core.run_measured(wall, DEMO_MODEL)
    assert r.statement == "Expert needed" and "exposed steel" in r.reason
