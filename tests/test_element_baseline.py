"""Tests for the element-model experiment (spec/features/element_model.md). No model is downloaded here."""

import numpy as np
import pytest

from experiments.element_baseline import CLASSES, UNSURE, decide, load_task6, phinet_names, report

README = """Task 6: component type
0: Beam
1: Column
2: Others
3: Wall
"""


def test_phinet_names_map_to_ours():
    assert phinet_names(README) == {0: "beam", 1: "column", 2: "others", 3: "wall"}


def test_decide_threshold_and_empty():
    assert decide({"column": 0.6, "wall": 0.3}, 0.5) == "column"
    assert decide({"column": 0.4}, 0.5) == UNSURE
    assert decide({}, 0.1) == UNSURE


def test_report_counts():
    truth = ["column", "column", "column", "beam", "wall", "others"]
    pred = ["column", "wall", UNSURE, "beam", "wall", "column"]
    r = report(truth, pred)
    assert r["n"] == 6
    assert r["coverage"] == pytest.approx(5 / 6, abs=1e-3)
    assert r["accuracy_confident"] == pytest.approx(3 / 5, abs=1e-3)
    assert r["recall"]["column"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["column_confidently_wrong"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["others_called_element"] == 1.0
    assert r["confusion"]["column"][UNSURE] == 1


def test_loader_on_fake_arrays(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 20), ("test", 8)):
        np.save(tmp_path / f"X_{split}.npy", rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8))
        np.save(tmp_path / f"y_{split}.npy", np.arange(n) % 4)
    (tmp_path / "README.txt").write_text(README)
    d = load_task6(tmp_path, limit=6)
    x, y, idx = d["test"]
    assert len(idx) == 6 and set(CLASSES[v] for v in y) == set(CLASSES)
    assert CLASSES[y[3]] == "wall"  # PHI-Net code 3 is wall


def test_predictions_come_back_in_the_order_asked_for():
    """A shuffled index list (the validation split) must line up with its labels."""
    import torch
    from experiments.element_baseline import predict_resnet

    class Brightness(torch.nn.Module):  # 'class' = which quarter of the brightness range the image is in
        def forward(self, t):
            level = t.mean((1, 2, 3))
            return torch.stack([-(level - c) ** 2 * 50 for c in (-1.5, -0.5, 0.5, 1.5)], 1)

    x = np.stack([np.full((224, 224, 3), v, np.uint8) for v in (10, 90, 160, 250, 15, 245)])
    idx = np.array([5, 0, 3, 1, 4, 2])
    got = predict_resnet(Brightness(), x, idx, "cpu", batch=4).argmax(1)
    straight = predict_resnet(Brightness(), x, np.arange(6), "cpu").argmax(1)
    assert list(got) == list(straight[idx])
