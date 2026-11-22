"""Exposed steel and spalling on the close-up (D107): what SAM 3.1 finds for a few prompts, and what the spalling model
(D61, trained on whole elements) says about the close-up, on every real two-photo case. Writes contact sheets to
data/results/exposed_steel/ (local, the photos are not ours to share) and a JSON of the numbers, for review by eye.
Usage: py src/experiments/exposed_steel_eval.py"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from app import simulate  # noqa: E402

PROMPTS = ["rebar", "steel bar", "exposed rebar", "rusty steel bar"]
OUT = ROOT / "data" / "results" / "exposed_steel"
COL = {"rebar": (255, 60, 60), "steel bar": (255, 170, 0), "exposed rebar": (0, 200, 255), "rusty steel bar": (200, 80, 255)}


def main() -> None:
    import torch
    from element.parts import sam_segmenter, shared_predictor
    from element.spalling import load, probability
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp = load(device=dev)
    p = shared_predictor()
    seg = sam_segmenter(p, PROMPTS)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "inbox_cases" / "_everything.json").write_text(json.dumps({"cases": simulate.all_real_cases()}),
                                                                    encoding="utf-8")
    cases = list(simulate.field_cases()) + list(simulate.real_cases(10_000, ROOT / "data" / "inbox_cases" /
                                                                    "_everything.json")) + list(simulate.topo_cases())
    rows, tiles = [], []
    for i, (crack, element, meta) in enumerate(cases):
        im = crack.convert("RGB")
        im.thumbnail((1024, 1024))
        rgb = np.asarray(im)
        found = seg(rgb)
        best = {}
        for word, m, s in found:
            share = float(m.mean())
            if word not in best or s > best[word][1]:
                best[word] = (m, float(s), share)
        ps = probability(sp, im, dev) if sp is not None else None
        rows.append({"i": i, "kind": meta.get("kind", "")[:40], "spalling_p": None if ps is None else round(ps, 3),
                     **{w: [round(b[1], 2), round(b[2], 4)] for w, b in best.items()}})
        a = rgb.astype(float).copy()
        for w, (m, s, share) in best.items():
            if s >= 0.3:
                a[m] = 0.45 * a[m] + 0.55 * np.array(COL[w])
        t = Image.fromarray(a.astype(np.uint8))
        t.thumbnail((300, 300))
        d = ImageDraw.Draw(t)
        txt = f"#{i} sp {ps:.2f} " + " ".join(f"{w[:5]}{b[1]:.2f}" for w, b in best.items()) if ps is not None else f"#{i}"
        d.rectangle([0, 0, t.width, 14], fill=(0, 0, 0))
        d.text((3, 2), txt, fill=(255, 255, 255))
        tiles.append(t)
        print(rows[-1], flush=True)
    per = 20
    for s in range(0, len(tiles), per):
        sheet = Image.new("RGB", (5 * 300, 4 * 300), "black")
        for j, t in enumerate(tiles[s:s + per]):
            sheet.paste(t, ((j % 5) * 300, (j // 5) * 300))
        sheet.save(OUT / f"sheet_{s // per:02d}.jpg", quality=82)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
