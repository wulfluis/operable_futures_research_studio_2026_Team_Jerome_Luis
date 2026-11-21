"""Can SAM 3.1, asked for "crack" and "exposed concrete", trace whole cracks and spalling? (D91)
Scores it against the crack model v4 on the Özgenel tiles (val to choose, test to report) and on PHI-Net task 3
(spalling yes / no). Usage, from the repository root:  py src/experiments/sam_vision_eval.py
Writes data/results/sam_vision/scores.json (local). Spec: spec/features/sam_vision.md."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from element.parts import make_predictor, sam_segmenter  # noqa: E402
from segmentation.decide import CRACK, NO_CRACK, decide  # noqa: E402
from segmentation.predict import probability  # noqa: E402
from segmentation.train import load_checkpoint  # noqa: E402

SEG = ROOT / "data" / "processed" / "segmentation"
PHI3 = ROOT / "data" / "raw" / "phinet" / "task3_spalling_condition"
CONFS = (0.2, 0.3, 0.4, 0.5, 0.6)
MIN_SHARE = 0.002  # a mask must cover at least this share of the photo to count (specks are not an answer)
HIT_PX = 20  # a crack is found when at least this many of its true pixels are inside the masks


def masks(seg, img: Image.Image, word: str):
    im = img.convert("RGB")
    return [(m, p) for w, m, p in seg(np.asarray(im)) if w == word]


def union(ms, conf, shape):
    out = np.zeros(shape, bool)
    for m, p in ms:
        if p >= conf and m.sum() >= MIN_SHARE * m.size:
            out |= m
    return out


def crack_split(split, seg, unet, dev):
    rows = []
    for p in sorted((SEG / split / "images").glob("ozg_*")):
        mp = SEG / split / "masks" / p.name
        if not mp.exists():
            continue
        img = Image.open(p).convert("RGB")
        truth = np.asarray(Image.open(mp).convert("L")) > 0
        ms = masks(seg, img, "crack")
        d = decide(probability(unet, img, dev), scale=max(img.size) / 416)
        rows.append((truth, ms, d))
    out = {"tiles_with_crack": int(sum(bool(t.any()) for t, _, _ in rows)), "tiles_without": int(sum(not t.any() for t, _, _ in rows))}
    u = {"missed": int(sum(bool(t.any()) and d.statement == NO_CRACK for t, _, d in rows)),
         "false_alarms": int(sum((not t.any()) and d.statement == CRACK for t, _, d in rows)),
         "crack_pixels_covered": round(float(np.mean(np.concatenate([d.mask[t] for t, _, d in rows if t.any()]))), 3)}
    out["crack_model_v4"] = u
    for c in CONFS:
        hit = miss = fa = 0
        cover = []
        for t, ms, d in rows:
            m = union(ms, c, t.shape)
            if t.any():
                ok = int((m & t).sum()) >= HIT_PX
                hit += ok
                miss += not ok
                cover.append(m[t])
            else:
                fa += bool(m.any())
        both = []  # the crack model's line joined with SAM's
        for t, ms, d in rows:
            if t.any():
                both.append((union(ms, c, t.shape) | d.mask)[t])
        out[f"sam_conf_{c}"] = {"missed": miss, "false_alarms": fa,
                                "crack_pixels_covered": round(float(np.mean(np.concatenate(cover))), 3),
                                "crack_pixels_covered_with_crack_model": round(float(np.mean(np.concatenate(both))), 3)}
    return out


def spalling(seg):
    x = np.load(PHI3 / "X_test.npy", mmap_mode="r")
    y = np.load(PHI3 / "y_test.npy").astype(int)
    found = {c: {0: 0, 1: 0} for c in CONFS}
    for i in range(len(y)):
        img = Image.fromarray(np.asarray(x[i])).resize((672, 672), Image.BICUBIC)
        ms = masks(seg, img, "exposed concrete")
        for c in CONFS:
            found[c][int(y[i])] += bool(union(ms, c, (672, 672)).any())
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return {"photos_with_spalling": n1, "photos_without": n0,
            **{f"conf_{c}": {"with_spalling_lit": f"{found[c][1]} of {n1}", "without_spalling_lit": f"{found[c][0]} of {n0}"}
               for c in CONFS}}


def main():
    t0 = time.time()
    dev = "cuda"
    seg = sam_segmenter(make_predictor(dev, conf=min(CONFS)), ["crack", "exposed concrete"])
    unet = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    res = {"version": "sam-vision-eval-0.1", "min_mask_share": MIN_SHARE, "hit_px": HIT_PX}
    for split in ("val", "test"):
        res[f"ozgenel_{split}"] = crack_split(split, seg, unet, dev)
        print(split, json.dumps(res[f"ozgenel_{split}"], indent=1), flush=True)
    res["phinet_spalling_test"] = spalling(seg)
    print(json.dumps(res["phinet_spalling_test"], indent=1), flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out = ROOT / "data" / "results" / "sam_vision"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("done", res["minutes"], "min")


if __name__ == "__main__":
    main()
