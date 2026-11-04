import csv
from collections import defaultdict

import numpy as np
import pytest
from PIL import Image

from segmentation.prepare import main, polygons_to_mask, soft_mask_to_binary, source_key, split_of


def test_polygon_fills_the_right_pixels():
    mask = polygons_to_mask("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75", size=100)
    assert set(np.unique(mask)) == {0, 255}
    assert mask[50, 50] == 255 and mask[5, 5] == 0
    assert 0.23 < (mask > 0).mean() < 0.28  # a quarter of the image


def test_empty_or_short_label_gives_empty_mask():
    assert polygons_to_mask("", size=32).max() == 0
    assert polygons_to_mask("0 0.1 0.1 0.2 0.2", size=32).max() == 0  # two points are not a polygon


def test_soft_mask_threshold_keeps_crack_and_drops_halo():
    soft = np.zeros((40, 40), np.uint8)
    soft[:, 19:21] = 255  # crack core
    soft[:, 18] = soft[:, 21] = 60  # soft halo, below the threshold
    binary = soft_mask_to_binary(Image.fromarray(soft), size=40)
    assert set(np.unique(binary)) == {0, 255}
    assert binary[:, 19:21].all() and not binary[:, 17].any()


@pytest.mark.parametrize("name, key", [
    ("Rissbilder_for_Florian_9S6A2793_2047_533_3461_3029.png", "Rissbilder_for_Florian_9S6A2793"),
    ("GAPS384_train_0691_1_1.png", "GAPS384_train_0691"),
    ("DeepCrack_11289-7.png", "DeepCrack_11289"),
    ("Volker_DSC01627_695_410_1138_1343.png", "Volker_DSC01627"),
    ("noncrack_noncrack_concrete_wall_0_1.png", "noncrack_noncrack_concrete_wall_0"),
    ("h_34_155.png", "h_34"),
    ("CFD_001.png", "CFD_001"),
])
def test_tiles_of_one_source_photo_share_a_key(name, key):
    assert source_key(name) == key


def test_split_is_repeatable_and_roughly_80_10_10():
    groups = [f"photo_{i}" for i in range(3000)]
    first = [split_of(g) for g in groups]
    assert first == [split_of(g) for g in groups]
    share = {s: first.count(s) / len(groups) for s in ("train", "val", "test")}
    assert 0.76 < share["train"] < 0.84 and 0.07 < share["val"] < 0.13 and 0.07 < share["test"] < 0.13


def _fake_sets(tmp_path):
    cs = tmp_path / "crack-seg"
    for i, split in enumerate(("train", "val", "test")):  # one photo id per split, as in the real set
        (cs / "images" / split).mkdir(parents=True)
        (cs / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (416, 416), "grey").save(cs / "images" / split / f"{i}.rf.a.jpg")
        (cs / "labels" / split / f"{i}.rf.a.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.2")
    Image.new("RGB", (416, 416)).save(cs / "images" / "train" / "8.rf.x.jpg")
    (cs / "labels" / "train" / "8.rf.x.txt").write_text("")  # empty label: dropped
    k9 = tmp_path / "cs9k"
    (k9 / "Images").mkdir(parents=True)
    (k9 / "Final_Masks" / "Masks").mkdir(parents=True)
    for i in range(6):
        for t in range(3):
            name = f"Volker_DSC{i:05d}_{t}_0_10_10.png"
            Image.new("RGB", (400, 400), "white").save(k9 / "Images" / name)
            Image.new("L", (400, 400), 200).save(k9 / "Final_Masks" / "Masks" / name)
    Image.new("RGB", (400, 400)).save(k9 / "Images" / "no_mask_here.png")  # unpaired: skipped
    return cs, k9


def test_end_to_end_on_fake_sets(tmp_path):
    cs, k9 = _fake_sets(tmp_path)
    out = tmp_path / "out"
    main([str(cs), str(k9), str(out)])
    rows = list(csv.DictReader(open(out / "index.csv", encoding="utf-8")))
    assert len(rows) == 3 + 18
    assert not any("8.rf" in r["file"] for r in rows)
    splits = defaultdict(set)
    for r in rows:
        splits[r["group"]].add(r["split"])
        img = Image.open(out / r["split"] / "images" / r["file"])
        mask = np.array(Image.open(out / r["split"] / "masks" / r["file"]))
        assert img.size == (416, 416) and mask.shape == (416, 416)
        assert set(np.unique(mask)) <= {0, 255}
    assert all(len(s) == 1 for s in splits.values())  # no source photo in two splits


def test_training_set_can_leave_out_and_repeat_sets(tmp_path):
    from segmentation.train import CrackSet
    for sub in ("images", "masks"):
        (tmp_path / "train" / sub).mkdir(parents=True)
    for name in ("cs9k_a.png", "metu_1.png", "darb_x.png"):
        for sub in ("images", "masks"):
            Image.new("RGB" if sub == "images" else "L", (32, 32)).save(tmp_path / "train" / sub / name)
    s = CrackSet(tmp_path, "train", exclude=("metu_",), repeat={"darb_": 3})
    assert sorted(f.name for f in s.files) == ["cs9k_a.png"] + ["darb_x.png"] * 3
