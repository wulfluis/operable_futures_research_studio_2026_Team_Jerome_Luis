"""Video of the crack model on unseen test photos: photo, crack appears in red, statement. Spec: spec/features/figures.md.

Usage, from the repository root (after src/segmentation/evaluate.py):
    py src/viz/segmentation_video.py data/processed/segmentation data/results/segmentation/second_look_test.mp4

Shows photos from every outcome, misses and false alarms included. Test photos come from CrackSeg9k (CC0) and
crack-seg (public domain); credit them in the presentation. The video stays in data/ (not committed).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.decide import decide
from segmentation.predict import MODEL, RED
from segmentation.train import load_checkpoint, to_tensor

W, H, FPS = 1920, 1080, 25
PHOTO = 700  # px, each of the two photos
INK, INK_2, GREY = "#111111", "#4d4d4d", "#8a8a8a"
PER_OUTCOME = {"crack_found": 5, "no_crack_found": 3, "expert_needed": 2, "missed_crack": 2, "false_alarm": 2}
LABEL = {"crack_found": "CORRECT", "no_crack_found": "CORRECT", "expert_needed": "SENT TO AN EXPERT",
         "missed_crack": "MISSED CRACK", "false_alarm": "FALSE ALARM"}


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    return ImageFont.truetype(path, size)


def frame(photo: np.ndarray, mask: np.ndarray, alpha: float, statement: str, line: str, verdict: str,
          index: int, total: int) -> np.ndarray:
    """One 1920 x 1080 frame: title, the photo, the photo with the crack in red (faded by alpha), the statement."""
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((60, 40), "SECOND LOOK: CRACK OR NO CRACK", fill=INK, font=_font(54))
    d.text((60, 110), "Unseen test photos. Left: the photo. Right: what the model marks as crack.", fill=INK_2,
           font=_font(26))
    d.text((W - 60, 52), f"{index} / {total}", fill=GREY, font=_font(34), anchor="ra")
    rgb = photo.astype(np.float32)
    red = rgb.copy()
    red[mask] = (1 - 0.65 * alpha) * red[mask] + 0.65 * alpha * RED
    y = 200
    for i, a in enumerate((rgb, red)):
        img = Image.fromarray(a.astype(np.uint8)).resize((PHOTO, PHOTO), Image.BILINEAR)
        canvas.paste(img, (60 + i * (PHOTO + 40), y))
    x = 60 + 2 * PHOTO + 90
    d.text((x, y), "MODEL SAYS", fill=GREY, font=_font(24))
    if alpha >= 1:
        d.text((x, y + 40), statement.upper().replace(" ", "\n", 1), fill=INK, font=_font(44), spacing=8)
        d.multiline_text((x, y + 170), _wrap(line, 22), fill=INK_2, font=_font(22), spacing=8)
        d.text((x, y + PHOTO - 130), "DATASET SAYS", fill=GREY, font=_font(24))
        d.multiline_text((x, y + PHOTO - 88), verdict, fill=INK, font=_font(26), spacing=8)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def _wrap(text: str, width: int) -> str:
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return "\n".join(lines + [cur])


def pick(photos_csv: Path, seed: int = 0) -> list[dict]:
    rows = list(csv.DictReader(open(photos_csv, encoding="utf-8")))
    rng = random.Random(seed)
    chosen = []
    for outcome, n in PER_OUTCOME.items():
        pool = [r for r in rows if r["outcome"] == outcome]
        chosen += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(chosen)
    return chosen


@torch.no_grad()
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.model, device)
    chosen = pick(Path("data/results/segmentation") / args.split / "photos.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i, row in enumerate(chosen, 1):
        photo = np.array(Image.open(args.data / args.split / "images" / row["file"]).convert("RGB"))
        prob = torch.sigmoid(model(to_tensor(photo)[None].to(device)).float())[0, 0].cpu().numpy()
        dec = decide(prob)
        has_crack = row["has_crack"] == "True"
        verdict = f"{'Crack' if has_crack else 'No crack'}\n{LABEL[row['outcome']]}"
        line = f"{dec.reason} Confidence {dec.confidence:.0%}." if dec.confidence else dec.reason
        for t in range(int(3.8 * FPS)):  # 1 s photo, 1 s crack fades in, 1.8 s statement
            alpha = min(1.0, max(0.0, (t - FPS) / FPS))
            video.write(frame(photo, dec.mask, alpha, dec.statement, line, verdict, i, len(chosen)))
    video.release()
    print(f"{len(chosen)} photos -> {args.out}")


if __name__ == "__main__":
    main()
