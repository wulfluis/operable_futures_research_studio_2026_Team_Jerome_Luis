"""Scene check, training and test: ConvNeXt-tiny on PHI-Net task 1 (object / close-up / structure). Spec:
spec/features/scene_gate.md (D54). Usage, from the repository root:
    py src/experiments/scene_gate.py data/raw/phinet/task1_scene_level --epochs 8
Weights to models/scene_convnext_tiny_phinet1.pt, numbers to data/processed/scene_gate.json (both not committed).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.element_baseline import (calibrate, device_name, fit_temperature, predict_resnet, split_train_val,
                                          train_resnet)
from experiments.phinet_prepare import class_names
from quality.scene import MODEL_FILE, SCENES, choose_close_up_min

ROOT = Path(__file__).resolve().parents[2]
PHINET_WORD = {"object": "object", "close_up": "pixel", "structure": "structural"}


def check_order(readme: str) -> None:
    """Our SCENES order must be PHI-Net's numbering (0 object, 1 pixel, 2 structural)."""
    names = {k: v.lower() for k, v in class_names(readme).items()}
    for k, s in enumerate(SCENES):
        assert PHINET_WORD[s] in names[k], f"PHI-Net class {k} is {names[k]!r}, expected {PHINET_WORD[s]}"


def gate_report(prob: np.ndarray, y: np.ndarray, close_up_min: float) -> dict:
    cu = SCENES.index("close_up")
    top = prob.argmax(1)
    passed = prob[:, cu] >= close_up_min

    def share(mask):
        return round(float(np.mean(passed[mask])), 3) if mask.any() else None

    return {"n": int(len(y)), "top1_accuracy": round(float(np.mean(top == y)), 3),
            "confusion_top1": {SCENES[t]: {SCENES[p]: int(np.sum((y == t) & (top == p))) for p in range(len(SCENES))}
                               for t in range(len(SCENES))},
            "close_up_min": close_up_min,
            "close_ups_sent_to_retake": round(1 - share(y == cu), 3),
            "objects_let_through_as_close_up": share(y == SCENES.index("object")),
            "structures_let_through_as_close_up": share(y == SCENES.index("structure"))}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--limit", type=int, help="a random subset of train and test, for a quick run")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "scene_gate.json")
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
    print(f"device {dev}; train {len(idx)}, test {len(it)}", flush=True)
    t0 = time.time()
    model, _ = train_resnet(x, y, idx, dev, epochs=args.epochs, arch="convnext_tiny", improved=True,
                            n_classes=len(SCENES))
    val, _ = split_train_val(idx)
    vp, tp = predict_resnet(model, x, val, dev, flip=True), predict_resnet(model, xt, it, dev, flip=True)
    temperature = fit_temperature(vp, y[val])
    vp, tp = calibrate(vp, temperature), calibrate(tp, temperature)
    cu = SCENES.index("close_up")
    close_up_min = choose_close_up_min(vp[:, cu], y[val] == cu)
    results = {"data": str(args.folder), "epochs": args.epochs, "seconds": round(time.time() - t0),
               "temperature": temperature, "close_up_min": close_up_min,
               "validation": gate_report(vp, y[val], close_up_min), "test": gate_report(tp, yt[it], close_up_min)}
    import torch
    args.weights.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), args.weights)
    results["weights"] = str(args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("temperature", "close_up_min", "validation", "test")}, indent=1))
    return results


if __name__ == "__main__":
    main()
