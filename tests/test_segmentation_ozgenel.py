"""Özgenel photos cut into tiles for model v4 (spec/features/segmentation.md)."""

import csv

import numpy as np
from PIL import Image

from segmentation.ozgenel import main, mask_to_binary, pairs, tile_boxes
from segmentation.prepare import SIZE, split_of


def test_tiles_are_square_inside_the_photo_and_cover_both_sizes():
    for w, h, n in ((2448, 3264, 12), (3264, 2448, 12), (4032, 3024, 12)):
        boxes = tile_boxes(w, h)
        assert len(boxes) == n
        for x0, y0, x1, y1 in boxes:
            assert x1 - x0 == y1 - y0 == min(w, h) // 3 and x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h


def test_a_two_pixel_crack_survives_the_shrink():
    m = np.zeros((816, 816), np.uint8)
    m[400:402, 100:700] = 255  # 2 px wide, shrinks to about 1 px at 416
    out = mask_to_binary(Image.fromarray(m))
    assert out.shape == (SIZE, SIZE) and out.any()
    assert np.isin(out, (0, 255)).all()


def test_jpeg_noise_in_an_empty_mask_is_not_crack():
    noise = np.random.default_rng(0).integers(0, 60, (816, 816)).astype(np.uint8)
    assert not mask_to_binary(Image.fromarray(noise)).any()


def test_pairs_match_photos_and_masks_by_name(tmp_path):
    for folder, names in (("rgb", ("001", "002", "003")), ("BW", ("001", "002"))):
        (tmp_path / folder).mkdir()
        for n in names:
            Image.new("L", (30, 30)).save(tmp_path / folder / f"{n}.jpg")
    assert [p.stem for p, _ in pairs(tmp_path)] == ["001", "002"]


def test_a_photo_with_an_exif_rotation_is_turned_to_match_its_mask(tmp_path):
    """The bug of 9 Nov: a photo stored on its side with an EXIF 'rotate 90°' tag, its mask stored upright."""
    root, out = tmp_path / "raw", tmp_path / "out"
    (root / "rgb").mkdir(parents=True)
    (root / "BW").mkdir()
    shown = np.full((400, 300, 3), 200, np.uint8)
    shown[:, 140:150] = 10  # a dark vertical crack as the photo is shown
    stored = Image.fromarray(shown).transpose(Image.Transpose.ROTATE_90)  # what a phone writes: pixels on their side
    exif = Image.Exif()
    exif[0x0112] = 6  # EXIF orientation: turn 90° clockwise to show
    stored.save(root / "rgb" / "009.jpg", exif=exif, quality=95)
    mask = np.zeros((400, 300), np.uint8)
    mask[:, 140:150] = 255
    Image.fromarray(mask).save(root / "BW" / "009.png")
    out.mkdir()
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file", "dataset", "split", "group", "crack_fraction"])
    main([str(root), str(out)])
    split = split_of("ozg_009")
    for r in csv.DictReader(open(out / "index.csv", encoding="utf-8")):
        tile = np.asarray(Image.open(out / split / "images" / r["file"]).convert("L"))
        m = np.asarray(Image.open(out / split / "masks" / r["file"])) > 0
        if m.any():  # the masked pixels are the dark crack pixels of the photo
            assert tile[m].mean() < 100 < tile[~m].mean()


def test_main_adds_tiles_of_one_photo_to_one_split_and_reruns_cleanly(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    (root / "rgb").mkdir(parents=True)
    (root / "BW").mkdir()
    Image.new("RGB", (300, 400), (120, 120, 120)).save(root / "rgb" / "007.jpg")
    mask = np.zeros((400, 300), np.uint8)
    mask[:, 140:150] = 255
    Image.fromarray(mask).save(root / "BW" / "007.png")
    out.mkdir()
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["file", "dataset", "split", "group", "crack_fraction"])
    for _ in range(2):
        main([str(root), str(out)])
    rows = list(csv.DictReader(open(out / "index.csv", encoding="utf-8")))
    assert len(rows) == 12 and {r["split"] for r in rows} == {split_of("ozg_007")}
    assert sum(float(r["crack_fraction"]) > 0 for r in rows) == 4  # the column of tiles the crack runs through
