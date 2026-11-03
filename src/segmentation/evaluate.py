"""Score the trained U-Net once on a held-out split. Spec: spec/features/segmentation.md.

Usage, from the repository root:
    py src/segmentation/evaluate.py data/processed/segmentation --split test

Writes to data/results/segmentation/<split>/: scores.json, photos.csv, and one overlay per photo sorted into
folders by outcome (crack_found, no_crack_found, missed_crack, false_alarm, expert_needed).
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.decide import CRACK, NO_CRACK, UNSURE, decide, pixel_scores, rates
from segmentation.predict import MODEL, draw_overlay
from segmentation.train import load_checkpoint, to_tensor

DATASET = {"crackseg": "crack-seg", "cs9k": "CrackSeg9k", "metu": "METU", "sdnet": "SDNET2018", "darb": "Darbhanga"}


def outcome(statement: str, has_crack: bool) -> str:
    if statement == UNSURE:
        return "expert_needed"
    if statement == CRACK:
        return "crack_found" if has_crack else "false_alarm"
    return "missed_crack" if has_crack else "no_crack_found"


def side_by_side(img: Image.Image, truth: np.ndarray, model_view: Image.Image) -> Image.Image:
    """Left: the dataset's own crack outline (white). Right: the model's crack (red) and its statement."""
    left = np.array(img)
    left[truth] = 255
    out = Image.new("RGB", (img.width * 2 + 8, model_view.height), "white")
    out.paste(Image.fromarray(left), (0, 0))
    out.paste(model_view, (img.width + 8, 0))
    return out


@torch.no_grad()
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--out", type=Path, default=Path("data/results/segmentation"))
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.model, device)
    out = args.out / args.split

    pixels = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    table = Counter()
    rows = []
    for path in sorted((args.data / args.split / "images").glob("*.png")):
        img = Image.open(path).convert("RGB")
        truth = np.array(Image.open(args.data / args.split / "masks" / path.name)) > 0
        prob = torch.sigmoid(model(to_tensor(np.array(img))[None].to(device)).float())[0, 0].cpu().numpy()
        d = decide(prob)
        dataset = DATASET[path.name.split("_")[0]]
        for key in (dataset, "all"):
            for k, v in pixel_scores(prob > 0.5, truth).items():
                pixels[key][k] += v
        result = outcome(d.statement, bool(truth.any()))
        table[(dataset, result)] += 1
        rows.append(dict(file=path.name, dataset=dataset, has_crack=bool(truth.any()), statement=d.statement,
                         confidence=round(d.confidence, 3), largest_group=d.largest_group, outcome=result))
        (out / result).mkdir(parents=True, exist_ok=True)
        view = draw_overlay(img, d.mask, d.statement, f"{d.reason} Confidence {d.confidence:.0%}.")
        side_by_side(img, truth, view).save(out / result / path.name)

    scores = {"split": args.split, "photos": len(rows),
              "pixels": {k: {m: round(x, 4) for m, x in rates(c).items()} for k, c in pixels.items()},
              "photos_by_outcome": {f"{d} / {o}": n for (d, o), n in sorted(table.items())}}
    (out / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    with open(out / "photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
