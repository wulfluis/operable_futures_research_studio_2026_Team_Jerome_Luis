"""Add the EPFL crack sets of E. Corres Sojo's group [46] to the prepared set: candidate model v5 (D100).
Spec: spec/features/segmentation.md ("EPFL sets").

- TOPO-Loss "WILD" set (Pantoja-Rosero et al. 2022, Zenodo 6769028 [108], CC BY 4.0): 256 px tiles of photos of
  cracked plastered and masonry buildings after earthquakes, with thin crack masks. Prefix topo_.
- DIC crack set (Rezaie et al. 2020 [47], Zenodo 4307686 [109], CC BY 4.0): 256 px grey tiles of lab walls painted with a
  speckle pattern, with crack masks. Prefix dic_.
Tiles keep their photo together in one split (split_of on the photo's name), so no photo is in train and test.

Usage, from the repository root (both archives extracted in data/raw/epfl_topo and data/raw/epfl_dic):
    py -I src/segmentation/epfl.py data/raw data/processed/segmentation
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.ozgenel import mask_to_binary
from segmentation.prepare import _save, split_of

PREFIXES = ("topo_", "dic_")


def topo_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """(photo tile, mask, photo name) for the WILD training and validation tiles."""
    out = []
    for part in ("training_set", "validation_set"):
        d = root / "epfl_topo" / "data" / part
        for img in sorted((d / "c_images").glob("*.png")):
            m = d / "c_mask" / img.name
            if m.exists():
                out.append((img, m, img.name.split("_")[0]))  # AD0036_14_256_256_mask.png -> AD0036
    return out


def dic_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """(photo tile, mask, photo name) for the DIC tiles (train, valid and test folders)."""
    out = []
    d = root / "epfl_dic" / "DIC_crack_dataset"
    for part in ("train", "valid", "test"):
        for img in sorted((d / part).glob("*.tif")):
            m = d / f"{part}_GT" / f"{img.stem}_mask.png"
            if m.exists():
                photo = re.sub(r"_\d+_\d+_\d+$", "", img.stem)  # ..._0476_1_29_512_768 -> the photo before the tile
                out.append((img, m, photo))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    index = args.out / "index.csv"
    with open(index, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["file"].startswith(PREFIXES)]  # re-running replaces them
    added: list = []
    topo = topo_pairs(args.raw)
    topo_train = {photo for img, _, photo in topo if img.parent.parent.name == "training_set"}

    def split_for(prefix: str, photo: str) -> str:
        """DIC: our usual split by photo. TOPO has few photos with many tiles, so a hash split is lopsided: a photo
        of the authors' training set is ours too (6 photos are in both of theirs); the validation-only photos are
        divided between our validation and test, so no photo is on both sides."""
        if prefix != "topo_":
            return split_of(f"{prefix}{photo}")
        if photo in topo_train:
            return "train"
        return "val" if split_of(f"{prefix}{photo}", (0, 50, 50)) == "val" else "test"

    for prefix, dataset, found in (("topo_", "EPFL TOPO-Loss WILD", topo),
                                   ("dic_", "EPFL DIC cracks", dic_pairs(args.raw))):
        if prefix == "topo_":
            found = topo
        for img, msk, photo in found:
            group = f"{prefix}{photo}"
            with Image.open(img) as im, Image.open(msk) as mk:
                _save(args.out, split_for(prefix, photo), f"{prefix}{img.stem}.png", im.convert("RGB"), mask_to_binary(mk),
                      added, dataset=dataset, group=group)
        print(prefix, len(found), "tiles ->", {s: sum(r["split"] == s and r["file"].startswith(prefix) for r in added)
                                              for s in ("train", "val", "test")})
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dataset", "split", "group", "crack_fraction"])
        w.writeheader()
        w.writerows(rows + added)


if __name__ == "__main__":
    main()
