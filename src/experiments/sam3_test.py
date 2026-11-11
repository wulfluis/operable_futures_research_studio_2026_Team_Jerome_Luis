"""Offline test: does SAM 3.1 find the structural elements by text? (reading list [81][82])

Not part of the app. Nothing here sets a level (rule 5); it only shows SAM 3.1's masks on our own photos.

    python3 src/experiments/sam3_test.py data/samples/room.jpg [more photos] [--device mps|cpu] [--conf 0.25]
    python3 src/experiments/sam3_test.py photo.jpg --prompts "door,window"   (free words instead)

By default only the elements in ELEMENTS are searched: column, wall, beam, slab, balcony, stairs, window, expansion
joint (Luis's list, 11 Nov 2026).
Each is asked for with the words that worked on test photos (ELEMENTS); a thing found by two of its words counts
once. A wall stays "wall": load-bearing or infill cannot be read from a photo.

Weights: `models/sam3.1/sam3.1_multiplex.pt` (Hugging Face `facebook/sam3.1`, gated, SAM License; never in git).
Output (in `data/results/sam3/`, git-ignored): `<photo>_sam3.png` (masks drawn on the photo, one colour per
prompt) and `<photo>_sam3.json` (per prompt: how many objects, best confidence, share of the photo, time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# one list for the app and these tests (D62)
from element.parts import (IMGSZ, PARTS as ELEMENTS, SAME_THING_IOU, WALL_NOTE, WEIGHTS,  # noqa: E402,F401
                           as_parts as as_elements, make_predictor as _make_predictor)

OUT = ROOT / "data" / "results" / "sam3"
COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


LOAD_REPORT: dict = {}


def watch_weight_loading() -> None:
    """Ultralytics loads SAM 3 with strict=False and drops the result, so a checkpoint that does not fit would
    load silently with random weights. Record the first big load's missing keys instead."""
    import torch

    original = torch.nn.Module.load_state_dict

    def recording(self, state_dict, strict=True, assign=False):
        result = original(self, state_dict, strict=strict, assign=assign)
        if not LOAD_REPORT and len(state_dict) > 500:
            own = self.state_dict()
            LOAD_REPORT.update({
                "model_tensors": len(own), "filled_from_file": len(own) - len(result.missing_keys),
                "missing": len(result.missing_keys), "missing_examples": list(result.missing_keys)[:5],
                "unused_in_file": len(result.unexpected_keys)})
        return result

    torch.nn.Module.load_state_dict = recording


def make_predictor(device: str, conf: float):
    return _make_predictor(device, conf)


def draw(photo: Image.Image, found: list[tuple[str, np.ndarray, float]], prompts: list[str]) -> Image.Image:
    base = np.asarray(photo.convert("RGB")).astype(np.float32)
    out = base.copy()
    for name, mask, _ in found:
        colour = np.array(COLOURS[prompts.index(name) % len(COLOURS)], np.float32)
        out[mask] = 0.45 * base[mask] + 0.55 * colour
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    size = max(14, img.width // 60)
    y = 10
    for i, name in enumerate(prompts):
        hits = [c for n, _, c in found if n == name]
        label = f"{name}: {len(hits)}" + (f" (best {max(hits):.2f})" if hits else "")
        d.rectangle([10, y, 10 + size, y + size], fill=COLOURS[i % len(COLOURS)], outline=(0, 0, 0))
        d.text((16 + size, y), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0),
               font_size=size)
        y += size + 6
    return img


def run(photo_path: Path, predictor, prompts: list[str], device: str, elements: bool = False) -> dict:
    photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    t0 = time.perf_counter()
    predictor.set_image(np.asarray(photo)[:, :, ::-1].copy())  # Ultralytics expects BGR arrays
    t_image = time.perf_counter() - t0
    t1 = time.perf_counter()
    results = predictor(text=prompts)
    t_text = time.perf_counter() - t1

    found: list[tuple[str, np.ndarray, float]] = []
    r = results[0] if results else None
    if r is not None and r.masks is not None and len(r.masks):
        masks = r.masks.data.cpu().numpy() > 0.5
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for m, c, p in zip(masks, cls, confs):
            if m.shape != (photo.height, photo.width):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(photo.size)) > 127
            found.append((prompts[c] if c < len(prompts) else str(c), m, float(p)))

    labels = prompts
    if elements:
        found, labels = as_elements(found), list(ELEMENTS)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "elements" if elements else "sam3"
    draw(photo, found, labels).save(OUT / f"{photo_path.stem}_{tag}.png")
    summary = {
        "photo": photo_path.name, "size_px": list(photo.size), "device": device,
        "seconds_image": round(t_image, 2), "seconds_prompts": round(t_text, 2),
        "prompts": {
            name: {"objects": sum(1 for n, _, _ in found if n == name),
                   "best_confidence": round(max([c for n, _, c in found if n == name], default=0.0), 3),
                   "share_of_photo": round(float(np.mean(np.any(
                       [m for n, m, _ in found if n == name] or [np.zeros((1, 1), bool)], axis=0))), 3)}
            for name in labels},
    }
    if elements:
        summary["asked_with"] = ELEMENTS
        summary["prompts"]["wall"]["note"] = WALL_NOTE
    (OUT / f"{photo_path.stem}_{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--prompts", default=None, help="free words instead of the elements")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if not WEIGHTS.exists():
        raise SystemExit(f"Weights missing: {WEIGHTS} (download facebook/sam3.1 from Hugging Face)")
    elements = a.prompts is None
    prompts = ([w for words in ELEMENTS.values() for w in words] if elements
               else [p.strip() for p in a.prompts.split(",") if p.strip()])
    watch_weight_loading()
    predictor = make_predictor(a.device, a.conf)
    for i, p in enumerate(a.photos):
        t = time.perf_counter()
        summary = run(p, predictor, prompts, a.device, elements)
        if i == 0:
            summary["first_photo_seconds_incl_model_load"] = round(time.perf_counter() - t, 1)
            summary["weights"] = LOAD_REPORT
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
