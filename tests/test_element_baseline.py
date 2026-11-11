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


# ---------- second run (D52) ----------

def test_augment_keeps_shape_and_type_and_changes_the_image():
    from experiments.element_baseline import augment
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    a = augment(x, rng)
    assert a.shape == x.shape and a.dtype == np.uint8 and not np.array_equal(a, x)


def test_augment_never_turns_the_image_upside_down():
    """A beam turned upright looks like a column: a dark top half must stay at the top."""
    from experiments.element_baseline import augment
    x = np.full((8, 224, 224, 3), 230, np.uint8)
    x[:, :112] = 20
    a = augment(x, np.random.default_rng(1)).astype(float)
    assert (a[:, :90].mean(axis=(1, 2, 3)) < a[:, 134:].mean(axis=(1, 2, 3))).all()


def test_split_is_the_first_runs_split():
    from experiments.element_baseline import split_train_val
    idx = np.arange(100)
    val, tr = split_train_val(idx)
    assert len(val) == 10 and len(tr) == 90 and not set(val) & set(tr)
    assert list(val) == list(np.random.default_rng(0).permutation(idx)[:10])


def test_evaluate_network_sets_doubt_on_validation_and_scores_test():
    from experiments.element_baseline import evaluate_network
    # validation: a column the model confidently calls a wall (0.85) with column 0.15 -> doubt must be at most 0.15
    val_truth = ["column", "column", "wall", "wall"]
    val_prob = np.array([[0.0, 0.9, 0.1, 0.0], [0.0, 0.15, 0.85, 0.0], [0.0, 0.05, 0.95, 0.0], [0.0, 0.1, 0.9, 0.0]])
    test_truth = ["column", "wall"]
    test_prob = np.array([[0.0, 0.14, 0.86, 0.0], [0.0, 0.02, 0.98, 0.0]])
    r = evaluate_network(val_truth, val_prob, test_truth, test_prob)
    assert r["column_doubt"] == 0.14  # the largest step of 0.02 that still catches the validation column at 0.15
    assert r["test_threshold_only"]["confusion"]["column"]["wall"] == 1  # without the rule: confidently wrong
    assert r["validation_with_rule"]["column_confidently_wrong"] == 0
    assert r["test_with_rule"]["confusion"]["column"][UNSURE] == 1  # the hidden column goes to a person
    assert r["test_with_rule"]["confusion"]["wall"]["wall"] == 1


@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "convnext_tiny"])
def test_build_model_has_four_outputs(arch, monkeypatch):
    """Without downloading weights: the networks are built empty."""
    import torch
    import torchvision.models as tvm
    from experiments import element_baseline as eb
    for name in ("resnet18", "resnet50", "convnext_tiny"):
        real = getattr(tvm, name)
        monkeypatch.setattr(tvm, name, lambda weights=None, _r=real: _r(weights=None))
    out = eb.build_model(arch).eval()(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, len(CLASSES))


def test_calibrate_sharpens_and_softens_and_keeps_the_order():
    from experiments.element_baseline import calibrate
    p = np.array([[0.5, 0.3, 0.15, 0.05]])
    sharp, soft = calibrate(p, 0.5), calibrate(p, 2.0)
    assert sharp[0, 0] > 0.5 > soft[0, 0]
    assert np.allclose(sharp.sum(1), 1) and list(sharp[0].argsort()) == list(p[0].argsort())
    assert np.allclose(calibrate(p, 1.0), p)


def test_fit_temperature_sharpens_an_under_confident_model():
    from experiments.element_baseline import fit_temperature
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    p = np.full((400, 4), 0.15)
    p[np.arange(400), labels] = 0.55  # always right, never sure: the best temperature is well below 1
    assert fit_temperature(p, labels) < 0.5
