"""Spalling model, training and test: ConvNeXt-tiny on PHI-Net task 3 (spalling / no spalling). Spec:
spec/features/photo2.md (D61). Usage, from the repository root:
    py src/experiments/spalling_model.py data/raw/phinet/task3_spalling_condition --epochs 8
Weights to models/spalling_convnext_tiny_phinet3.pt, numbers to data/processed/spalling_model.json (not committed).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from element.spalling import CLASSES, MODEL_FILE, choose_spall_min
from experiments.element_baseline import (calibrate, device_name, fit_temperature, predict_resnet, split_train_val,
                                          train_resnet)
from experiments.phinet_prepare import class_names

ROOT = Path(__file__).resolve().parents[2]


def check_order(readme: str) -> None:
    names = {k: v.lower() for k, v in class_names(readme).items()}
    assert "non" in names[0] and "spall" in names[1] and "non" not in names[1], names


def report(p_spall: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    said = p_spall >= threshold
    sp, no = y == 1, y == 0
    return {"n": int(len(y)), "threshold": threshold,
            "spalling_caught": round(float(np.mean(said[sp])), 3),
            "false_alarms_on_no_spalling": round(float(np.mean(said[no])), 3),
            "accuracy": round(float(np.mean(said == sp)), 3),
            "confusion": {"spalling": {"seen": int(np.sum(said & sp)), "not_seen": int(np.sum(~said & sp))},
                          "no_spalling": {"seen": int(np.sum(said & no)), "not_seen": int(np.sum(~said & no))}}}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "spalling_model.json")
    p.add_argument("--weights", type=Path, default=MODEL_FILE)
    args = p.parse_args(argv)
    check_order((args.folder / "README.txt").read_text(encoding="utf-8", errors="replace"))
    data = {}
    for split in ("train", "test"):
        x = np.load(args.folder / f"X_{split}.npy", mmap_mode="r")
        y = np.load(args.folder / f"y_{split}.npy").astype(int)
        idx = np.arange(len(y))
        if args.limit and args.limit < len(y):
            idx = np.sort(np.random.default_rng(0).choice(idx, args.limit, replace=False))
        data[split] = (x, y, idx)
    dev = device_name(args.device)
    x, y, idx = data["train"]
    xt, yt, it = data["test"]
    t0 = time.time()
    model, _ = train_resnet(x, y, idx, dev, epochs=args.epochs, arch="convnext_tiny", improved=True,
                            n_classes=len(CLASSES))
    val, _ = split_train_val(idx)
    vp, tp = predict_resnet(model, x, val, dev, flip=True), predict_resnet(model, xt, it, dev, flip=True)
    temperature = fit_temperature(vp, y[val])
    vp, tp = calibrate(vp, temperature), calibrate(tp, temperature)
    threshold = choose_spall_min(vp[:, 1], y[val] == 1)
    results = {"epochs": args.epochs, "seconds": round(time.time() - t0), "temperature": temperature,
               "spall_min": threshold, "validation": report(vp[:, 1], y[val], threshold),
               "test": report(tp[:, 1], yt[it], threshold)}
    import torch
    args.weights.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
