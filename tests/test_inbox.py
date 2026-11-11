"""Simulated residents: the test inbox and the simulator (spec/features/class_demo.md, D59). No model file needed."""

import json
import threading

import numpy as np
import pytest
from PIL import Image

from app import inbox, simulate
from app.core import Wall
from measure.synthetic import CRACK_MM, make_demo
from test_app_measure import DEMO_MODEL
from test_element_predict import FakeElementModel


def _case(folder, name, crack, element=None, meta=None):
    simulate._write(folder / name, crack, element, meta or {"card_placed": True, "kind": "test"})


def _crack():
    grey, _ = make_demo(1.0)
    return Image.fromarray(np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2))


def test_a_case_goes_through_the_pipeline_onto_the_wall_marked_test(tmp_path):
    box = tmp_path / "inbox"
    box.mkdir()
    _case(box, "a", _crack(), _crack().convert("L").convert("RGB"))
    wall = Wall(export_dir=tmp_path / "cracks")
    info = inbox.process_case(inbox.ready_cases(box)[0], wall, DEMO_MODEL, "cpu", threading.Lock(),
                              FakeElementModel("beam"), None)
    e = wall.get(info["number"])
    assert e.stats["test"] is True and e.stats["measured"] and e.stats["element"]["answer"] == "beam"
    assert e.stats["triage"]["level"]
    assert json.loads((tmp_path / "cracks" / f"{info['number']:03d}.json").read_text(encoding="utf-8"))["test"]
    assert not list(box.glob("a")) and list((box / "done").iterdir())  # moved to done/


def test_half_written_cases_are_left_alone(tmp_path):
    (tmp_path / "x.part").mkdir()
    (tmp_path / "x.part" / "case.json").write_text("{}")
    (tmp_path / "y").mkdir()  # no case.json yet
    assert inbox.ready_cases(tmp_path) == []


def test_a_broken_case_never_stops_the_inbox(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "case.json").write_text("{}")  # no crack.jpg
    info = inbox.process_case(tmp_path / "bad", Wall(), DEMO_MODEL, "cpu", threading.Lock(), None, None)
    assert "error" in info and (tmp_path / "done").exists()


def test_real_uploads_are_not_marked_test():
    from app.core import analyse
    import io
    buf = io.BytesIO()
    _crack().save(buf, "JPEG")
    assert analyse(buf.getvalue(), True, False, Wall(), DEMO_MODEL).test is False


def test_the_crack_shape_can_be_chosen_and_the_default_is_unchanged():
    a, ta = make_demo(1.0)
    assert ta["length_mm"] == pytest.approx(np.linalg.norm(np.diff(CRACK_MM, axis=0), axis=1).sum())
    _, tb = make_demo(1.0, crack_mm=simulate.SHAPES["horizontal"])
    assert tb["length_mm"] != ta["length_mm"]


def test_the_simulated_cracks_stay_clear_of_the_card_and_inside_the_photo():
    for name, line in simulate.SHAPES.items():
        p = np.asarray(line, float)
        assert (p[:, 0] > 2).all() and (p[:, 0] < 198).all() and (p[:, 1] > 2).all() and (p[:, 1] < 148).all(), name
        on_card = (p[:, 0] >= 15) & (p[:, 0] <= 101) & (p[:, 1] >= 60) & (p[:, 1] <= 114)
        assert not on_card.any(), name


def test_ten_cases_with_a_card_and_an_element_photo():
    assert len(simulate.CASES) == 10 and {c[2] for c in simulate.CASES} == {"column", "beam", "wall"}
    assert all(isinstance(c[3], int) for c in simulate.CASES)  # a real PHI-Net photo for photo 2
    assert all(c[1] >= 0.6 for c in simulate.CASES)  # thinner is under 3 px at the synthetic scale


def test_sam_reaches_the_simulated_residents_too(tmp_path):
    """D70: the test inbox hands SAM 3.1 to each case like the phone page, so the wall shows 'Lies on' for them."""
    box = tmp_path / "inbox"
    box.mkdir()
    _case(box, "a", _crack(), _crack().convert("L").convert("RGB"))
    wall = Wall(export_dir=tmp_path / "cracks")
    calls = []
    info = inbox.process_case(inbox.ready_cases(box)[0], wall, DEMO_MODEL, "cpu", threading.Lock(),
                              FakeElementModel("beam"), None, parts_model=lambda rgb: calls.append(1) or [])
    assert calls and wall.get(info["number"]).stats["parts"] is not None
