"""SAM 3.1 as an extension of the crack model (D91): only SAM's crack lines that connect to the crack the model found
are added, so the model still decides whether there is a crack. And three more spalling prompts on PHI-Net task 3.
Usage: py src/experiments/sam_extend_eval.py  -> data/results/sam_vision/extend_scores.json (local)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from element.parts import make_predictor, sam_segmenter  # noqa: E402
from segmentation.decide import CRACK, NO_CRACK, decide  # noqa: E402
from segmentation.predict import probability  # noqa: E402
from segmentation.train import load_checkpoint  # noqa: E402

SEG = ROOT / "data" / "processed" / "segmentation"
PHI3 = ROOT / "data" / "raw" / "phinet" / "task3_spalling_condition"
CONFS = (0.3, 0.4, 0.5)
JOIN_SHARE = 0.02  # SAM lines this close (share of the long side) to the model's crack count as connected
SPALL_WORDS = ("rebar", "broken concrete", "spalled concrete")
MIN_SHARE = 0.002


def extend(model_mask: np.ndarray, sam: np.ndarray) -> np.ndarray:
    """SAM's crack pixels in pieces that touch the model's crack (grown by JOIN_SHARE), joined to it."""
    if not model_mask.any() or not sam.any():
        return model_mask
    near = ndimage.binary_dilation(model_mask, iterations=max(1, round(JOIN_SHARE * max(model_mask.shape))))
    lab, n = ndimage.label(sam, structure=np.ones((3, 3)))
    keep = np.unique(lab[near & sam])
    keep = keep[keep > 0]
    return model_mask | np.isin(lab, keep)


def main():
    t0 = time.time()
    dev = "cuda"
    seg = sam_segmenter(make_predictor(dev, conf=min(CONFS)), ["crack", *SPALL_WORDS])
    unet = load_checkpoint(ROOT / "models" / "unet_crack.pt", dev)
    res = {"join_share": JOIN_SHARE}
    for split in ("val", "test"):
        tp = {c: 0 for c in CONFS}
        fp = {c: 0 for c in CONFS}
        cover = {c: [] for c in CONFS}
        base_cover, base_fp, base_tp = [], 0, 0
        missed = fa = 0
        for p in sorted((SEG / split / "images").glob("ozg_*")):
            mp = SEG / split / "masks" / p.name
            if not mp.exists():
                continue
            img = Image.open(p).convert("RGB")
            truth = np.asarray(Image.open(mp).convert("L")) > 0
            d = decide(probability(unet, img, dev), scale=max(img.size) / 416)
            found = d.statement != NO_CRACK
            missed += truth.any() and not found
            fa += (not truth.any()) and d.statement == CRACK
            tol = ndimage.binary_dilation(truth, iterations=6)  # SAM's lines are thicker than the true crack
            out = [(w, m, s) for w, m, s in seg(np.asarray(img)) if w == "crack"]
            base = d.mask if found else np.zeros_like(truth)
            base_tp += int((base & tol).sum())
            base_fp += int((base & ~tol).sum())
            if truth.any():
                base_cover.append(base[truth])
            for c in CONFS:
                sam = np.zeros_like(truth)
                for _, m, s in out:
                    if s >= c and m.sum() >= MIN_SHARE * m.size:
                        sam |= m
                e = extend(base, sam)
                tp[c] += int((e & tol).sum())
                fp[c] += int((e & ~tol).sum())
                if truth.any():
                    cover[c].append(e[truth])
        r = {"crack_model_v4": {"missed": int(missed), "false_alarms": int(fa),
                                "crack_pixels_covered": round(float(np.mean(np.concatenate(base_cover))), 3),
                                "pixels_off_the_crack": round(base_fp / max(1, base_tp + base_fp), 3)}}
        for c in CONFS:
            r[f"extended_by_sam_conf_{c}"] = {
                "missed": int(missed), "false_alarms": int(fa),  # unchanged: the model still decides
                "crack_pixels_covered": round(float(np.mean(np.concatenate(cover[c]))), 3),
                "pixels_off_the_crack": round(fp[c] / max(1, tp[c] + fp[c]), 3)}
        res[f"ozgenel_{split}"] = r
        print(split, json.dumps(r, indent=1), flush=True)
    x = np.load(PHI3 / "X_test.npy", mmap_mode="r")
    y = np.load(PHI3 / "y_test.npy").astype(int)
    lit = {(w, c): [0, 0] for w in SPALL_WORDS for c in CONFS}
    for i in range(len(y)):
        img = np.asarray(Image.fromarray(np.asarray(x[i])).resize((672, 672), Image.BICUBIC))
        out = seg(img)
        for w in SPALL_WORDS:
            for c in CONFS:
                lit[(w, c)][int(y[i])] += any(ww == w and s >= c and m.sum() >= MIN_SHARE * m.size for ww, m, s in out)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    res["phinet_spalling_test"] = {f"{w} @ {c}": {"with_spalling": f"{v[1]} of {n1}", "without": f"{v[0]} of {n0}"}
                                   for (w, c), v in lit.items()}
    print(json.dumps(res["phinet_spalling_test"], indent=1), flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (ROOT / "data" / "results" / "sam_vision" / "extend_scores.json").write_text(json.dumps(res, indent=1),
                                                                                  encoding="utf-8")
    print("done", res["minutes"], "min")


if __name__ == "__main__":
    main()
