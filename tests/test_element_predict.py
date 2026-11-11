"""The element model in the app (spec/features/app_classification.md, D53). No model file is needed."""

import numpy as np
import pytest
import torch
from PIL import Image

from element import predict
from triage.element_rule import ElementAnswer


class FakeElementModel(torch.nn.Module):
    """Always leans to one class, whatever the photo."""

    def __init__(self, cls="column", strength=4.0):
        super().__init__()
        self.k, self.s = predict.CLASSES.index(cls), strength

    def forward(self, x):
        out = torch.zeros(x.shape[0], len(predict.CLASSES))
        out[:, self.k] = self.s
        return out


def _photo():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 400, 3), dtype=np.uint8))


def test_missing_model_file_gives_none_and_an_unsure_answer(tmp_path):
    assert predict.load(tmp_path / "nothing.pt") is None
    a = predict.read_element(None, _photo())
    assert a.element == "unsure" and "not installed" in a.reason


def test_probabilities_are_calibrated_and_sum_to_one():
    p = predict.probabilities(FakeElementModel("beam"), _photo())
    assert set(p) == set(predict.CLASSES) and sum(p.values()) == pytest.approx(1.0)
    raw = torch.softmax(FakeElementModel("beam")(torch.zeros(1, 3, 224, 224)), 1)[0, 0].item()
    assert p["beam"] > raw  # temperature under 1 sharpens


def test_confident_column_and_weak_answer():
    assert predict.read_element(FakeElementModel("column"), _photo()).element == "column"
    assert predict.read_element(FakeElementModel("wall", strength=0.5), _photo()).element == "unsure"


@pytest.mark.parametrize("answer, key", [("column", "column"), ("beam", "beam"), ("wall", "load_bearing_wall"),
                                         ("others", None), ("unsure", None)])
def test_model_answer_to_matrix_element(answer, key):
    assert predict.matrix_element(ElementAnswer(answer, False, 0.9, "")) == key


def test_no_answer_maps_to_no_element():
    assert predict.matrix_element(None) is None


@pytest.mark.skipif(not predict.MODEL_FILE.exists(), reason="the trained element model is not on this computer")
def test_the_trained_model_loads_and_answers():
    m = predict.load()
    a = predict.read_element(m, _photo())
    assert a.element in predict.CLASSES + ["unsure"] and 0 <= a.probability <= 1
