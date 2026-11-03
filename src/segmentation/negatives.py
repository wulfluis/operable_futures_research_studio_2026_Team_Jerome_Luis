"""Add close-ups with no crack (empty masks) to the prepared set: model v2. Spec: spec/features/segmentation.md.

Usage, from the repository root (after src/segmentation/prepare.py):
    py -I src/segmentation/negatives.py data/raw data/processed/segmentation
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.prepare import SIZE, _save, split_of

SEED = 0
METU, SDNET = 3000, 1500  # how many to take (illustrative)
DARBHANGA = "darbhanga/Crack Data-Darbhanga Fort"


def has_box(label: Path) -> bool:
    return label.exists() and bool(label.read_text(encoding="utf-8").strip())


def sources(raw: Path) -> list[tuple[str, str, Path, str]]:
    """(dataset, file prefix, image, source-photo group) for every no-crack image to add."""
    rng = random.Random(SEED)
    out = []
    metu = sorted((raw / "metu" / "Negative").glob("*.jpg"))
    out += [("METU", "metu", p, "metu_" + p.stem) for p in rng.sample(metu, min(METU, len(metu)))]
    sdnet = sorted((raw / "sdnet2018" / "W" / "UW").glob("*.jpg"))
    out += [("SDNET2018", "sdnet", p, "sdnet_" + p.stem.split("-")[0]) for p in rng.sample(sdnet, min(SDNET, len(sdnet)))]
    for split in ("train", "val"):  # the test folder has no labels, so its images cannot be trusted as no-crack
        folder = raw / DARBHANGA / split
        for p in sorted((folder / "images").glob("*.jpg")):
            if not has_box(folder / "labels" / (p.stem + ".txt")):
                out.append(("Darbhanga", "darb", p, f"darb_{split}_{p.stem}"))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    index = args.out / "index.csv"
    with open(index, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    known = {r["file"] for r in rows}
    empty = np.zeros((SIZE, SIZE), np.uint8)
    added = []
    for dataset, prefix, path, group in sources(args.raw):
        name = f"{prefix}_{path.stem}.png".replace(" ", "")  # stems are unique per set (Darbhanga: "train (1)")
        if name in known:
            continue
        with Image.open(path) as img:
            _save(args.out, split_of(group), name, img, empty, added, dataset=dataset, group=group)
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dataset", "split", "group", "crack_fraction"])
        w.writeheader()
        w.writerows(rows + added)
    for ds in ("METU", "SDNET2018", "Darbhanga"):
        print(ds, {s: sum(r["dataset"] == ds and r["split"] == s for r in added) for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
