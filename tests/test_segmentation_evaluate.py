import json

import numpy as np
import pytest
import torch
from PIL import Image

from segmentation.evaluate import main, outcome
from segmentation.train import build_model


@pytest.mark.parametrize("statement, has_crack, expected", [
    ("Crack found", True, "crack_found"),
    ("Crack found", False, "false_alarm"),
    ("No crack found", True, "missed_crack"),
    ("No crack found", False, "no_crack_found"),
    ("Expert needed", True, "expert_needed"),
    ("Expert needed", False, "expert_needed"),
])
def test_every_statement_lands_in_one_outcome(statement, has_crack, expected):
    assert outcome(statement, has_crack) == expected


def test_end_to_end_with_an_untrained_model(tmp_path):
    data = tmp_path / "seg"
    for sub in ("images", "masks"):
        (data / "test" / sub).mkdir(parents=True)
    for name, crack in (("crackseg_1.png", True), ("cs9k_noncrack_a_0.png", False)):
        Image.new("RGB", (416, 416), "grey").save(data / "test" / "images" / name)
        m = np.zeros((416, 416), np.uint8)
        m[200, 50:350] = 255 if crack else 0
        Image.fromarray(m).save(data / "test" / "masks" / name)
    torch.save({"model": build_model(pretrained=False).state_dict()}, tmp_path / "m.pt")
    main([str(data), "--model", str(tmp_path / "m.pt"), "--out", str(tmp_path / "res")])
    scores = json.loads((tmp_path / "res" / "test" / "scores.json").read_text())
    assert scores["photos"] == 2 and set(scores["pixels"]) == {"crack-seg", "CrackSeg9k", "all"}
    assert sum(scores["photos_by_outcome"].values()) == 2
    assert len(list((tmp_path / "res" / "test").glob("*/*.png"))) == 2
