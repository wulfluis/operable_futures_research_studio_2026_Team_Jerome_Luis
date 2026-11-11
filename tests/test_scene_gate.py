"""Scene check: is the photo a close-up? (spec/features/scene_gate.md, D54). No model file is needed."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.core import PhotoRejected, Wall, analyse_element, run_measured
from quality import scene
from test_app_measure import DEMO_MODEL, _demo
from test_element_predict import FakeElementModel


class FakeScene(torch.nn.Module):
    """Always reads the same scene."""

    def __init__(self, kind):
        super().__init__()
        self.k = scene.SCENES.index(kind)

    def forward(self, x):
        out = torch.zeros(x.shape[0], len(scene.SCENES))
        out[:, self.k] = 6.0
        return out


class Spy(torch.nn.Module):
    """A crack model that records whether it was asked."""

    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, x):
        self.called = True
        return DEMO_MODEL(x)


def _jpeg(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def test_close_up_passes():
    a = scene.crack_photo_check({"object": 0.1, "close_up": 0.85, "structure": 0.05})
    assert a.ok and a.scene == "close_up"


@pytest.mark.parametrize("probs, word", [({"object": 0.05, "close_up": 0.02, "structure": 0.93}, "building"),
                                         ({"object": 0.9, "close_up": 0.05, "structure": 0.05}, "element")])
def test_wide_view_is_a_retake_with_a_hint(probs, word):
    a = scene.crack_photo_check(probs, close_up_min=0.1)
    assert not a.ok and word in a.reason and "20 to 40 cm" in a.reason


def test_element_photo_must_not_be_a_close_up():
    assert not scene.element_photo_check({"object": 0.2, "close_up": 0.7, "structure": 0.1}).ok
    assert scene.element_photo_check({"object": 0.8, "close_up": 0.1, "structure": 0.1}).ok


def test_no_model_checks_nothing(tmp_path):
    assert scene.load(tmp_path / "none.pt") is None
    assert scene.crack_photo_check(None).ok and scene.element_photo_check(None).ok


def test_threshold_is_the_highest_that_keeps_close_up_retakes_under_target():
    p = np.array([0.95, 0.9, 0.8, 0.6, 0.3, 0.12, 0.08, 0.05, 0.02, 0.01])  # probabilities of real close-ups
    is_cu = np.ones(10, bool)
    c = scene.choose_close_up_min(p, is_cu, target=0.2)  # at most 2 of 10 sent to retake
    assert np.mean(p < c) <= 0.2 and np.mean(p < c + 0.05) > 0.2


def test_a_wide_view_never_reaches_the_crack_model():
    spy = Spy()
    r = run_measured(_demo(1.0)[0], spy, scene_model=FakeScene("structure"), card_placed=True)
    assert r.statement == "Retake" and not spy.called and "close-up" in r.reason
    assert r.to_dict()["scene"]["ok"] is False


def test_a_close_up_goes_on_to_the_crack_model():
    spy = Spy()
    r = run_measured(_demo(1.0)[0], spy, scene_model=FakeScene("close_up"), card_placed=True)
    assert spy.called and r.statement == "Crack found" and r.to_dict()["scene"]["ok"] is True


def test_without_a_scene_model_the_answer_is_as_before():
    with_none = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    assert with_none.scene is None and with_none.statement == "Crack found"


def test_a_close_up_sent_as_the_element_photo_is_refused():
    r = run_measured(_demo(1.0)[0], DEMO_MODEL, card_placed=True)
    with pytest.raises(PhotoRejected, match="step back"):
        analyse_element(_jpeg(_demo(1.0)[0]), r, Wall(), FakeElementModel("column"), scene_model=FakeScene("close_up"))
    a = analyse_element(_jpeg(_demo(1.0)[0]), r, Wall(), FakeElementModel("column"), scene_model=FakeScene("object"))
    assert a.element == "column"


@pytest.mark.skipif(not scene.MODEL_FILE.exists(), reason="the trained scene model is not on this computer")
def test_the_trained_scene_model_loads():
    m = scene.load()
    p = scene.probabilities(m, _demo(1.0)[0])
    assert set(p) == set(scene.SCENES) and abs(sum(p.values()) - 1) < 1e-5
