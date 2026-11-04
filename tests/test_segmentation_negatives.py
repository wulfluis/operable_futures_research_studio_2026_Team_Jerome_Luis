import csv
from collections import defaultdict

import numpy as np
from PIL import Image

from segmentation.negatives import DARBHANGA, main, sources


def _raw(tmp_path):
    raw = tmp_path / "raw"
    (raw / "metu" / "Negative").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (227, 227), "grey").save(raw / "metu" / "Negative" / f"{i:05d}.jpg")
    (raw / "sdnet2018" / "W" / "UW").mkdir(parents=True)
    for photo in (7069, 7070):
        for t in range(1, 4):
            Image.new("RGB", (256, 256), "grey").save(raw / "sdnet2018" / "W" / "UW" / f"{photo}-{t}.jpg")
    for split in ("train", "val", "test"):
        (raw / DARBHANGA / split / "images").mkdir(parents=True)
        (raw / DARBHANGA / split / "labels").mkdir(parents=True)
        for i, box in ((1, True), (2, False)):
            Image.new("RGB", (768, 768), "red").save(raw / DARBHANGA / split / "images" / f"{split} ({i}).jpg")
            if box and split != "test":
                (raw / DARBHANGA / split / "labels" / f"{split} ({i}).txt").write_text("0 0.5 0.5 0.1 0.1")
    return raw


def test_only_no_crack_images_are_taken(tmp_path):
    found = sources(_raw(tmp_path))
    darb = sorted(p.name for d, _, p, _ in found if d == "Darbhanga")
    assert darb == ["train (2).jpg", "val (2).jpg"]  # boxed images and the unlabelled test folder are left out
    assert sum(d == "METU" for d, *_ in found) == 4 and sum(d == "SDNET2018" for d, *_ in found) == 6


def test_added_with_empty_masks_and_tiles_of_one_photo_in_one_split(tmp_path):
    raw, out = _raw(tmp_path), tmp_path / "seg"
    out.mkdir()
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "dataset", "split", "group", "crack_fraction"],
                                 ["cs9k_a.png", "CrackSeg9k", "train", "a", "0.02"]])
    main([str(raw), str(out)])
    main([str(raw), str(out)])  # running twice adds nothing twice
    rows = list(csv.DictReader(open(out / "index.csv", encoding="utf-8")))
    assert len(rows) == 1 + 4 + 6 + 2
    splits = defaultdict(set)
    for r in rows[1:]:
        splits[r["group"]].add(r["split"])
        mask = np.array(Image.open(out / r["split"] / "masks" / r["file"]))
        assert mask.shape == (416, 416) and mask.max() == 0 and r["crack_fraction"] == "0.0"
    assert all(len(s) == 1 for s in splits.values())
