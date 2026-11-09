"""Add the Özgenel concrete crack photos with masks [49] to the prepared set, cut into tiles: model v4.
Spec: spec/features/segmentation.md ("Building concrete with crack masks").

Usage, from the repository root (after prepare.py and negatives.py; the RAR extracted to data/interim/ozgenel):
    py -I src/segmentation/ozgenel.py data/interim/ozgenel data/processed/segmentation
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.prepare import SIZE, _save, split_of

GRID_SHORT = 3  # tiles along the photo's short side; the tile side is short side / 3 (illustrative)
MASK_SHARE = 0.25  # a resized mask pixel is crack when at least this share of it was crack (illustrative)
MASK_FOLDER = "BW"  # the archive's black-and-white masks; the photos sit in the other image folder


def tile_boxes(width: int, height: int, per_short: int = GRID_SHORT) -> list[tuple[int, int, int, int]]:
    """Square tiles of side short/per_short, centred so the leftover margin is split on both sides."""
    side = min(width, height) // per_short
    cols, rows = width // side, height // side
    x0, y0 = (width - cols * side) // 2, (height - rows * side) // 2
    return [(x0 + c * side, y0 + r * side, x0 + (c + 1) * side, y0 + (r + 1) * side)
            for r in range(rows) for c in range(cols)]


def mask_to_binary(mask: Image.Image, size: int = SIZE, share: float = MASK_SHARE) -> np.ndarray:
    """Area-average the mask down to size, then crack where at least `share` of the area was crack: keeps
    cracks one or two pixels wide that a 0.5 threshold would erase when the tile shrinks."""
    crack = (np.asarray(mask.convert("L")) > 127).astype(np.float32)
    small = np.asarray(Image.fromarray(crack, "F").resize((size, size), Image.BOX))
    return np.where(small >= share, 255, 0).astype(np.uint8)


def pairs(root: Path) -> list[tuple[Path, Path]]:
    """(photo, mask) pairs by file stem: masks in the BW folder, photos in any other folder."""
    images = [p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    masks = {p.stem: p for p in images if p.parent.name == MASK_FOLDER}
    photos = {p.stem: p for p in images if p.parent.name != MASK_FOLDER}
    return [(photos[s], masks[s]) for s in sorted(photos) if s in masks]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    index = args.out / "index.csv"
    with open(index, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["file"].startswith("ozg_")]  # re-running replaces them
    added: list = []
    found = pairs(args.root)
    for photo_path, mask_path in found:
        group = f"ozg_{photo_path.stem}"
        split = split_of(group)
        with Image.open(photo_path) as photo, Image.open(mask_path) as mask:
            # 194 of the 458 photos carry an EXIF rotation (90° or 180°) that their masks do not: the masks are drawn
            # on the photo as it is shown, so the photo is turned by its tag first. A size that still differs is an
            # error, never stretched (9 Nov: stretching misaligned 42 % of the masks).
            photo, mask = ImageOps.exif_transpose(photo).convert("RGB"), ImageOps.exif_transpose(mask).convert("L")
            if mask.size != photo.size:
                raise ValueError(f"{photo_path.name}: photo {photo.size} and mask {mask.size} differ after rotation")
            for i, box in enumerate(tile_boxes(*photo.size)):
                _save(args.out, split, f"ozg_{photo_path.stem}_{i:02d}.png", photo.crop(box),
                      mask_to_binary(mask.crop(box)), added, dataset="Ozgenel", group=group)
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dataset", "split", "group", "crack_fraction"])
        w.writeheader()
        w.writerows(rows + added)
    print(f"{len(found)} photos ->", {s: sum(r["split"] == s for r in added) for s in ("train", "val", "test")},
          "tiles; with crack:", sum(float(r["crack_fraction"]) > 0 for r in added))


if __name__ == "__main__":
    main()
