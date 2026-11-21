"""Candidates for a real two-photo test set (D94): survey photos of a whole wall, column or beam, with a close-up cut
around the likeliest crack. Unlike pick_cases.py, a candidate is NOT required to pass the app's crack answer, so cases
the app misses stay in the set. A person then keeps or drops each candidate by eye, before the app is scored on them.
Local only: the photos and the lists are never committed (D63). Usage:
    py src/app/pick_test_set.py data/raw/haiti2010/Photos --n 150
Writes data/inbox_cases/test_candidates.json and contact sheets data/inbox_cases/test_sheets/sheet_NN.jpg."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.pick_cases import crop_box  # noqa: E402

OUT = ROOT / "data" / "inbox_cases" / "test_candidates.json"
SHEETS = ROOT / "data" / "inbox_cases" / "test_sheets"
MIN_ELEMENT_P = 0.7  # the element model must be fairly sure it is a wall, column or beam


def main(argv=None) -> None:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps

    from element import photo2
    from element.predict import load as load_el, read_element
    from quality.scene import load as load_sc, probabilities
    from segmentation.train import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    crack_m = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    el_m, sc_m = load_el(device=dev), load_sc(device=dev)
    photos = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    random.Random(21).shuffle(photos)
    cands = []
    for p in photos:
        if len(cands) >= args.n:
            break
        try:
            with Image.open(p) as im:
                im.draft("RGB", (1100, 1100))
                small = ImageOps.exif_transpose(im).convert("RGB")
        except OSError:
            continue
        sc = probabilities(sc_m, small, dev)
        if max(sc, key=sc.get) != "object":  # a whole element, not a close-up or a whole building
            continue
        a = read_element(el_m, small, dev)
        if a.element not in ("wall", "column", "beam") or a.probability < MIN_ELEMENT_P:
            continue
        on = photo2.read(crack_m, small, dev)  # only to place the close-up; its answer is not a filter
        if on.bbox_px is None:
            continue
        with Image.open(p) as im:
            size = ImageOps.exif_transpose(im).size
        box = crop_box(on.bbox_px, on.size_px, size)
        cands.append({"id": len(cands), "photo": str(p.resolve().relative_to(ROOT)).replace("\\", "/"), "crop": box,
                      "element": a.element, "element_p": round(a.probability, 2)})
        print(f"  {len(cands)}: {a.element} {a.probability:.0%}  {p.parent.name}/{p.name}", flush=True)
    OUT.write_text(json.dumps({"note": "candidates, not yet reviewed (D94)", "cases": cands}, indent=1), encoding="utf-8")
    SHEETS.mkdir(parents=True, exist_ok=True)
    per = 12
    for s in range(0, len(cands), per):
        sheet = Image.new("RGB", (4 * 520, 3 * 290), "black")
        d = ImageDraw.Draw(sheet)
        for j, c in enumerate(cands[s:s + per]):
            with Image.open(ROOT / c["photo"]) as im:
                full = ImageOps.exif_transpose(im).convert("RGB")
            close = full.crop(tuple(c["crop"]))
            close.thumbnail((260, 260))
            whole = full.copy()
            dw = ImageDraw.Draw(whole)
            dw.rectangle(c["crop"], outline=(255, 0, 0), width=max(4, full.width // 150))
            whole.thumbnail((250, 260))
            x, y = (j % 4) * 520, (j // 4) * 290
            sheet.paste(close, (x, y))
            sheet.paste(whole, (x + 265, y))
            d.text((x + 4, y + 268), f"#{c['id']} {c['element']}", fill="white")
        sheet.save(SHEETS / f"sheet_{s // per:02d}.jpg", quality=80)
    print(f"{len(cands)} candidates -> {OUT}; sheets in {SHEETS}")


if __name__ == "__main__":
    main()
