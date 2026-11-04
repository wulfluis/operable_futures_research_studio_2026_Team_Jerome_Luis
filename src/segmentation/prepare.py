"""Prepare public crack sets for U-Net training. Spec: spec/features/segmentation.md.

Usage, from the repository root (CrackSeg9k Vol1 extracted to data/interim/crackseg9k first):
    py -I src/segmentation/prepare.py data/raw/crack-seg data/interim/crackseg9k/Final-Dataset-Vol1 data/processed/segmentation
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 416  # crack-seg's own size, a multiple of 32 for the U-Net (illustrative)
MASK_THRESHOLD = 127  # CrackSeg9k masks have soft edges; crack = above this (assumption)
SPLIT = (80, 10, 10)  # train / val / test, % of source photos (illustrative)


def polygons_to_mask(label_text: str, size: int = SIZE) -> np.ndarray:
    """YOLO segmentation lines ('class x1 y1 x2 y2 ...', normalised 0 to 1) filled into a 0/255 mask."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for line in label_text.splitlines():
        values = [float(v) for v in line.split()[1:]]
        if len(values) >= 6:
            draw.polygon([(x * size, y * size) for x, y in zip(values[0::2], values[1::2])], fill=255)
    return np.array(mask)


def soft_mask_to_binary(mask: Image.Image, size: int = SIZE) -> np.ndarray:
    """Resize as grey (keeps thin cracks), then threshold to 0/255."""
    grey = np.array(mask.convert("L").resize((size, size), Image.BILINEAR))
    return np.where(grey > MASK_THRESHOLD, 255, 0).astype(np.uint8)


def source_key(name: str) -> str:
    """CrackSeg9k file name -> the source photo it was cut from, so its tiles stay in one split."""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    prefix = parts[0]
    if prefix == "Rissbilder":  # Rissbilder_for_Florian_<photo>_<x>_<y>_<w>_<h>
        return "_".join(parts[:4])
    if prefix == "GAPS384":  # GAPS384_<set>_<photo>_<i>_<j>
        return "_".join(parts[:3])
    if prefix == "DeepCrack":  # DeepCrack_<photo>-<tile>
        return "DeepCrack_" + parts[1].split("-")[0]
    if prefix in ("Volker", "h", "d"):  # Volker_<photo>_<crop>, h_<photo>_<tile>
        return "_".join(parts[:2])
    if prefix == "noncrack" and len(parts) > 1:  # noncrack_..._<photo>_<tile>
        return "_".join(parts[:-1])
    return stem


def split_of(group: str, split: tuple[int, int, int] = SPLIT) -> str:
    """Fixed split from a hash of the group: same answer on every machine and every run."""
    bucket = int(hashlib.sha1(group.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < split[0]:
        return "train"
    return "val" if bucket < split[0] + split[1] else "test"


def _save(out: Path, split: str, name: str, image: Image.Image, mask: np.ndarray, rows: list, **info) -> None:
    for sub in ("images", "masks"):
        (out / split / sub).mkdir(parents=True, exist_ok=True)
    image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(out / split / "images" / name)
    Image.fromarray(mask).save(out / split / "masks" / name)
    rows.append(dict(file=name, split=split, crack_fraction=round(float((mask > 0).mean()), 5), **info))


def prepare_crack_seg(root: Path, out: Path, rows: list) -> list[str]:
    """Keep the published split; drop images whose label file is empty."""
    dropped = []
    for split in ("train", "val", "test"):
        for img_path in sorted((root / "images" / split).glob("*.jpg")):
            text = (root / "labels" / split / (img_path.stem + ".txt")).read_text(encoding="utf-8")
            if not text.strip():
                dropped.append(img_path.name)
                continue
            with Image.open(img_path) as img:
                _save(out, split, "crackseg_" + img_path.stem + ".png", img, polygons_to_mask(text), rows,
                      dataset="crack-seg", group="crack-seg_" + img_path.name.split(".rf.")[0])
    return dropped


def prepare_crackseg9k(root: Path, out: Path, rows: list) -> None:
    """Pairs image + mask by file name; split by source photo (the published lists leak)."""
    for img_path in sorted((root / "Images").glob("*.png")):
        mask_path = root / "Final_Masks" / "Masks" / img_path.name
        if not mask_path.exists():
            continue
        group = source_key(img_path.name)
        with Image.open(img_path) as img, Image.open(mask_path) as m:
            _save(out, split_of(group), "cs9k_" + img_path.name, img, soft_mask_to_binary(m), rows,
                  dataset="CrackSeg9k", group=group)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("crack_seg", type=Path)
    ap.add_argument("crackseg9k", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    rows: list = []
    dropped = prepare_crack_seg(args.crack_seg, args.out, rows)
    prepare_crackseg9k(args.crackseg9k, args.out, rows)
    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dataset", "split", "group", "crack_fraction"])
        w.writeheader()
        w.writerows(rows)
    for ds in ("crack-seg", "CrackSeg9k"):
        counts = {s: sum(r["dataset"] == ds and r["split"] == s for r in rows) for s in ("train", "val", "test")}
        print(ds, counts)
    print("dropped (empty label):", dropped)


if __name__ == "__main__":
    main()
