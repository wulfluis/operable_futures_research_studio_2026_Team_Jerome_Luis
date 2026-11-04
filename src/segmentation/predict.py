"""One photo in, one statement out: quality gate, U-Net, written crack rule. Spec: spec/features/segmentation.md.

Usage, from the repository root:
    py src/segmentation/predict.py <photo> [--out overlay.png]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality.gate import check
from segmentation.decide import Decision, decide
from segmentation.train import load_checkpoint, to_tensor

MODEL = Path("models/unet_crack.pt")
WORK_SIZE = 832  # long side the photo is scaled to before the U-Net (illustrative, 2 x the training size)
RED = np.array([214, 39, 40], np.float32)


@dataclass
class Result:
    statement: str  # Retake, Crack found, No crack found or Expert needed
    reason: str
    confidence: float | None
    overlay: Image.Image
    decision: Decision | None = None


def working_image(img: Image.Image, long_side: int = WORK_SIZE) -> tuple[Image.Image, tuple[int, int]]:
    """Shrink the long side to long_side and pad to a multiple of 32 (the U-Net halves the size five times).
    Returns the padded image and the size of the photo inside it."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > long_side:
        img.thumbnail((long_side, long_side), Image.BILINEAR)
    w, h = img.size
    padded = Image.new("RGB", (-(-w // 32) * 32, -(-h // 32) * 32), (127, 127, 127))
    padded.paste(img, (0, 0))
    return padded, (w, h)


@torch.no_grad()
def probability(model: torch.nn.Module, img: Image.Image, device: str) -> np.ndarray:
    logits = model(to_tensor(np.array(img))[None].to(device))
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def draw_overlay(img: Image.Image, mask: np.ndarray, title: str, line: str) -> Image.Image:
    """The photo with the crack in red, and the statement in a white band below."""
    rgb = np.array(img, np.float32)
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * RED
    w, h = img.size
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    size = max(16, w // 22)
    big, small = ImageFont.truetype(bold, size), ImageFont.truetype(bold, max(11, size // 2))
    lines = [""]
    for word in line.split():  # wrap to the photo's width
        joined = f"{lines[-1]} {word}".strip()
        if lines[-1] and small.getlength(joined) > w - 24:
            lines.append(word)
        else:
            lines[-1] = joined
    out = Image.new("RGB", (w, h + 24 + size + len(lines) * (small.size + 4)), "white")
    out.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((12, h + 8), title.upper(), fill="#111111", font=big)
    draw.multiline_text((12, h + 14 + size), "\n".join(lines), fill="#4d4d4d", font=small, spacing=4)
    return out


def run(photo: Image.Image, model: torch.nn.Module, device: str = "cpu") -> Result:
    img, (w, h) = working_image(photo)
    gate = check(np.asarray(img.crop((0, 0, w, h)).convert("L"), dtype=np.float64))  # the photo, not the padding
    problems = [f for f in gate.failed if f != "scale card not confirmed"]
    if problems:
        reason = f"The photo is {' and '.join(problems)}; please take it again."
        return Result("Retake", reason, None, draw_overlay(img.crop((0, 0, w, h)), np.zeros((h, w), bool), "Retake", reason))
    prob = probability(model, img, device)[:h, :w]
    d = decide(prob, scale=max(w, h) / 416)
    line = f"{d.reason} Confidence {d.confidence:.0%}." if d.confidence else d.reason
    if d.statement != "No crack found":
        line += " Width unknown: add a bank card (or the scale card) next to the crack."
    return Result(d.statement, line, d.confidence, draw_overlay(img.crop((0, 0, w, h)), d.mask, d.statement, line), d)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with Image.open(args.photo) as photo:
        result = run(photo, load_checkpoint(args.model, device), device)
    print(f"{result.statement}: {result.reason}")
    out = args.out or args.photo.with_name(args.photo.stem + "_second_look.png")
    result.overlay.save(out)
    print("overlay:", out)


if __name__ == "__main__":
    main()
