"""Pick real two-photo test cases for each risk tab of the wall (D79): up to 10 for Category 3, Expert needed,
Category 2 and Category 1, to see what the models and the written rules make of real photos today.

Each candidate is cut from one high-resolution survey photo as in pick_cases.py (D63): photo 2 the whole photo of the
element, photo 1 a close-up around the crack the crack model finds. It then goes through the app's own pipeline, the
same code the wall runs (inbox.process_case, SAM 3.1 when its weights are here), and is kept for the tab its level
falls in. No level is forced or edited: a tab the photos never reach stays short, and the script says so.
Writes data/inbox_cases/by_level.json (local, ignored by git); the photos are never copied into the repository.

Tabs the real photos never fill (no survey photo has a bank card, so no width is measured and the rules send them
high) can then be filled with SYNTHETIC cases, said so on the wall: photo 1 the drawn wall with a bank card and a crack
of known width (measure/synthetic.py, D61), photo 2 a real PHI-Net element photo. Their level too comes from the
app's own pipeline; they show what the rules do with a measured thin crack, not what real photos give.

Usage (Haiti 2010 survey photos, local only until their licence is confirmed):
    py src/app/pick_by_level.py data/raw/haiti2010/Photos --per-tab 10
    py src/app/pick_by_level.py --fill-synthetic --per-tab 10   # after it: fill the tabs still short
    py src/app/simulate.py --by-level        # then send them to the running app
"""

import argparse
import json
import random
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

BY_LEVEL = ROOT / "data" / "inbox_cases" / "by_level.json"
ELEMENTS = ("column", "beam", "wall")
MIN_ELEMENT_P = 0.6  # looser than pick_cases.py (0.85): an unsure element is a real case too (it gives Expert needed)
MIN_EXTENT = 0.15  # the crack spans at least this share of the photo of the element


def tally(picks: list[dict], tabs=None) -> dict[str, int]:
    from app.core import RISK_TABS
    return {t: sum(p["level_tab"] == t for p in picks) for t in (tabs or RISK_TABS)}


SYN_WIDTHS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)  # mm, the crack drawn on the synthetic wall
SYN_SHAPES = ("vertical", "steep", "horizontal", "diagonal")
SYN_ELEMENTS = ("wall", "beam", "column")


def synthetic_case(c: dict):
    """(photo 1, photo 2, meta) of a synthetic case from its parameters (stored in by_level.json)."""
    import numpy as np
    from PIL import Image

    from app import simulate
    from measure.synthetic import make_demo
    grey, _ = make_demo(c["width_mm"], seed=c["seed"], tilt_lr_deg=c["tilt"][0], tilt_ud_deg=c["tilt"][1],
                        crack_mm=simulate.SHAPES[c["shape"]])
    crack = Image.fromarray(np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2))
    # never paired with an unrelated real element photo (D91): photo 1 only, so it cannot reach a category tab
    return crack, None, {"card_placed": True,
                         "kind": f"DRAWN: bank card, {c['shape']} crack {c['width_mm']} mm (photo 1 only, D79, D91)"}


def fill_synthetic(per_tab: int, tabs=None, out: Path | None = None) -> None:
    """Fill the tabs the real photos left short with synthetic cases, each run through the app's own pipeline.
    tabs: only these (e.g. Category 2 and Category 1); out: write there instead of by_level.json."""
    import itertools
    import shutil as sh
    import tempfile as tf

    import torch

    from app import inbox, simulate
    from app.core import Wall, risk_group
    from element.parts import load_parts_model
    from element.predict import load as load_el
    from element.spalling import load as load_sp
    from quality.scene import load as load_sc
    from segmentation.train import load_checkpoint
    out = out or BY_LEVEL
    data = json.loads(BY_LEVEL.read_text(encoding="utf-8")) if BY_LEVEL.exists() and out == BY_LEVEL else {"cases": []}
    picks = data["cases"]

    def short() -> dict[str, int]:
        return tally(picks, tabs)
    if all(n >= per_tab for n in short().values()):
        print("Every tab already full:", tally(picks))
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    crack_m = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    el_m, sc_m, sp_m, sam = load_el(device=dev), load_sc(device=dev), load_sp(device=dev), load_parts_model(dev)
    tmp, wall = Path(tf.mkdtemp()), Wall()
    combos = list(itertools.product(SYN_WIDTHS, SYN_SHAPES, SYN_ELEMENTS, range(6)))
    random.Random(7).shuffle(combos)
    try:
        for i, (w, shape, element, k) in enumerate(combos):
            if all(n >= per_tab for n in short().values()):
                break
            c = {"synthetic": True, "width_mm": w, "shape": shape, "element": element, "k": k, "seed": i,
                 "tilt": [random.Random(i).uniform(-8, 8), random.Random(i + 1).uniform(-6, 6)]}
            crack, el_photo, meta = synthetic_case(c)
            case = tmp / f"s{i:05d}"
            simulate._write(case, crack, el_photo, meta)
            info = inbox.process_case(case, wall, crack_m, dev, threading.Lock(), el_m, sc_m, sp_m, sam)
            entry = wall.get(info["number"]) if info.get("number") else None
            level = (entry.stats.get("triage") or {}).get("level") if entry is not None else None
            tab = risk_group(entry.stats) if entry is not None and level is not None else None
            if tab is None or tab not in short() or short()[tab] >= per_tab:
                continue
            picks.append({**c, "level": level, "level_tab": tab, "why": entry.stats["triage"].get("reason", "")})
            print(f"  SYNTHETIC {tab:<14}: {shape} {w} mm, photo 2 a {element}  {short()}", flush=True)
            out.write_text(json.dumps({**data, "cases": picks}, indent=1), encoding="utf-8")  # as it goes
    finally:
        sh.rmtree(tmp, ignore_errors=True)
    data["cases"] = picks
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Now {short()} -> {out}")
    for t_, n in short().items():
        if n < per_tab:
            print(f"  {t_}: only {n}, even with synthetic cases; no level was forced.")


def main(argv=None) -> None:
    import torch
    from PIL import Image, ImageOps

    from app import inbox, simulate
    from app.core import RISK_TABS, Wall, risk_group
    from app.pick_cases import crop_box
    from element import photo2
    from element.parts import load_parts_model
    from element.predict import load as load_el, read_element
    from element.spalling import load as load_sp
    from quality.scene import load as load_sc, probabilities
    from segmentation.train import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, nargs="?")
    ap.add_argument("--fill-synthetic", action="store_true", help="fill the tabs still short with synthetic cases")
    ap.add_argument("--tabs", nargs="*", help='only these tabs, e.g. --tabs "Category 2" "Category 1"')
    ap.add_argument("--out", type=Path, help="write the picks here instead of by_level.json")
    ap.add_argument("--per-tab", type=int, default=10)
    ap.add_argument("--max-photos", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.fill_synthetic:
        fill_synthetic(args.per_tab, args.tabs, args.out)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    crack_m = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    el_m, sc_m, sp_m = load_el(device=dev), load_sc(device=dev), load_sp(device=dev)
    sam = load_parts_model(dev)
    photos = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    random.Random(args.seed).shuffle(photos)
    tmp = Path(tempfile.mkdtemp())
    wall = Wall()  # in memory only: nothing written for Grasshopper
    picks, seen, tried = [], 0, 0
    try:
        for p in photos[:args.max_photos]:
            if all(n >= args.per_tab for n in tally(picks).values()):
                break
            seen += 1
            try:
                with Image.open(p) as im:
                    im.draft("RGB", (1100, 1100))
                    small = ImageOps.exif_transpose(im).convert("RGB")
            except OSError:
                continue
            sc = probabilities(sc_m, small, dev)
            if max(sc, key=sc.get) != "object":  # photo 2 shows an element, not a close-up or a whole building
                continue
            a = read_element(el_m, small, dev)
            if a.element not in ELEMENTS or a.probability < MIN_ELEMENT_P:
                continue
            on = photo2.read(crack_m, small, dev)
            if not on.found or on.extent < MIN_EXTENT or on.bbox_px is None:
                continue
            with Image.open(p) as im:
                full = ImageOps.exif_transpose(im).convert("RGB")
            box = crop_box(on.bbox_px, on.size_px, full.size)
            whole = full.copy()
            whole.thumbnail((2400, 2400))
            case = tmp / f"c{seen:05d}"
            simulate._write(case, full.crop(box), whole, {"card_placed": False, "kind": "pick_by_level"})
            tried += 1
            info = inbox.process_case(case, wall, crack_m, dev, threading.Lock(), el_m, sc_m, sp_m, sam)
            entry = wall.get(info["number"]) if info.get("number") else None
            tab = risk_group(entry.stats) if entry is not None else None
            level = (entry.stats.get("triage") or {}).get("level") if entry is not None else None
            if tab is None or tally(picks)[tab] >= args.per_tab:
                continue
            if tab == "Expert needed" and level is None:
                continue  # a level the rules gave, not a case still waiting for its photo 2
            picks.append({"photo": str(p.resolve().relative_to(ROOT)).replace("\\", "/"), "crop": box,
                          "element": a.element, "level": level, "level_tab": tab,
                          "why": (entry.stats.get("triage") or {}).get("reason", "")})
            print(f"  {tab:<14} {sum(tally(picks).values()):>2}: {a.element} ({a.probability:.0%}) "
                  f"[{p.parent.name}/{p.name}]  {tally(picks)}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    BY_LEVEL.parent.mkdir(parents=True, exist_ok=True)
    BY_LEVEL.write_text(json.dumps({"source": str(args.folder), "note": "local only (D63, D79)", "cases": picks},
                                   indent=1), encoding="utf-8")
    print(f"{len(picks)} cases ({tally(picks)}) from {seen} photos looked at, {tried} run through the app -> {BY_LEVEL}")
    for t, n in tally(picks).items():
        if n < args.per_tab:
            print(f"  {t}: only {n}. The photos did not reach it more often; no level was forced.")


if __name__ == "__main__":
    main()
