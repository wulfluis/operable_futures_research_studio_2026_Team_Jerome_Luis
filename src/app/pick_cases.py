"""Pick real two-photo test cases from a folder of high-resolution photos of damaged buildings (D63): photo 2 = the
whole photo of an element, photo 1 = a close-up cut from the same photo around the crack the crack model finds. Each
case must pass the app's own checks; nothing is switched off. Writes data/inbox_cases/picks.json (local, ignored by
git); the photos themselves are never copied into the repository.

Usage (Haiti 2010 survey photos, local only until their licence is confirmed):
    py src/app/pick_cases.py data/raw/haiti2010/Photos --n 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PICKS = ROOT / "data" / "inbox_cases" / "picks.json"
WANT = {"column": 4, "wall": 4, "beam": 2}  # Haiti has two beam photos that pass every check (11 Nov)
CROP_SIDE = 0.30  # photo 1: a square of 30 % of the photo's short side around the crack (a 12 MP photo: about 850 px)
MIN_EXTENT = 0.25  # the crack spans at least this share of the element on photo 2


def crop_box(bbox_px, size_px, full_size, side=CROP_SIDE):
    """The close-up box in the full photo's pixels, centred on the crack found on the working copy."""
    (x0, y0, x1, y1), (w, h), (W, H) = bbox_px, size_px, full_size
    cx, cy = (x0 + x1) / 2 / w * W, (y0 + y1) / 2 / h * H
    s = side * min(W, H)
    left, top = min(max(0, cx - s / 2), W - s), min(max(0, cy - s / 2), H - s)
    return [int(left), int(top), int(left + s), int(top + s)]


def main(argv=None) -> None:
    import torch
    from PIL import Image, ImageOps

    from app.core import run_measured
    from element import photo2
    from element.predict import load as load_el, read_element
    from quality.scene import load as load_sc, probabilities
    from segmentation.train import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-photos", type=int, default=2500)
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    crack_m = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    el_m, sc_m = load_el(device=dev), load_sc(device=dev)
    photos = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    random.Random(0).shuffle(photos)
    want = {k: max(1, round(v * args.n / 10)) for k, v in WANT.items()}
    picks, seen = [], 0
    for p in photos[:args.max_photos]:
        if sum(want.values()) == 0:
            break
        seen += 1
        try:
            with Image.open(p) as im:
                im.draft("RGB", (1100, 1100))
                small = ImageOps.exif_transpose(im).convert("RGB")
        except OSError:
            continue
        sc = probabilities(sc_m, small, dev)
        if max(sc, key=sc.get) != "object":  # photo 2 must show an element, not a close-up or a whole building
            continue
        a = read_element(el_m, small, dev)
        if a.element not in want or want[a.element] == 0 or a.probability < 0.85:
            continue
        on = photo2.read(crack_m, small, dev)
        if not on.found or on.extent < MIN_EXTENT or on.bbox_px is None:
            continue
        with Image.open(p) as im:
            full = ImageOps.exif_transpose(im).convert("RGB")
        box = crop_box(on.bbox_px, on.size_px, full.size)
        r = run_measured(full.crop(box), crack_m, dev, sc_m)  # photo 1 through the app's own checks
        if r.statement not in ("Crack found", "Expert needed"):
            continue
        picks.append({"photo": str(p.resolve().relative_to(ROOT)).replace("\\", "/"), "crop": box, "element": a.element,
                      "extent": round(on.extent, 2), "photo1_statement": r.statement})
        want[a.element] -= 1
        print(f"  {len(picks)}: {a.element} ({a.probability:.0%}), crack over {on.extent:.0%} of it, photo 1 "
              f"{r.statement}  [{p.parent.name}/{p.name}]", flush=True)
    PICKS.parent.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps({"source": str(args.folder), "note": "local only (D63)", "cases": picks}, indent=1),
                     encoding="utf-8")
    print(f"{len(picks)} cases from {seen} photos looked at -> {PICKS}")


if __name__ == "__main__":
    main()
