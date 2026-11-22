"""Is the close-up check too strict? (D110) Candidate rules for a crack photo without a bank card, scored on both sides:
our real crack close-ups (they should pass) and PHI-Net's held-out task 1 test photos (whole buildings must not pass;
close-ups should). Usage: py src/experiments/scene_relax_eval.py
Writes data/results/scene_relax.json (local; the photos stay local)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image  # noqa: E402

from quality.scene import CLOSE_UP_MIN, HALVES_NEEDED, SCENES, halves_of, load, probabilities  # noqa: E402

PHI = ROOT / "data" / "raw" / "phinet" / "task1_scene_level"


def rules(p: dict, halves: list[float]) -> dict:
    best = max(p, key=p.get)
    h_ok = sum(h >= CLOSE_UP_MIN for h in halves) >= HALVES_NEEDED
    return {
        "A today: close-up at least 10 %": p["close_up"] >= CLOSE_UP_MIN,
        "B or 2 of 4 halves read as close-up, not a building": p["close_up"] >= CLOSE_UP_MIN or (best != "structure" and h_ok),
        "C anything but a building": best != "structure" or p["close_up"] >= CLOSE_UP_MIN,
        "D B, and a building only under 50 %": p["close_up"] >= CLOSE_UP_MIN or (p["structure"] < 0.5 and h_ok),
    }


def score(model, images, dev) -> list[dict]:
    out = []
    for im in images:
        p = probabilities(model, im, dev)
        halves = [probabilities(model, h, dev)["close_up"] for h in halves_of(im)]
        out.append(rules(p, halves))
    return out


def share(rows, key):
    return round(float(np.mean([r[key] for r in rows])), 3) if rows else None


def main() -> None:
    import torch
    from app import simulate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(device=dev)
    (ROOT / "data" / "inbox_cases" / "_everything.json").write_text(
        json.dumps({"cases": simulate.all_real_cases()}), encoding="utf-8")
    ours = [c for c, _, _ in list(simulate.field_cases()) + list(simulate.real_cases(
        10_000, ROOT / "data" / "inbox_cases" / "_everything.json")) + list(simulate.topo_cases())]
    res = {"ours (real crack close-ups, should pass)": score(model, ours, dev)}
    X, y = np.load(PHI / "X_test.npy", mmap_mode="r"), np.load(PHI / "y_test.npy")
    y = y.ravel().astype(int)
    rng = np.random.default_rng(54)
    for k, s in enumerate(SCENES):
        idx = rng.permutation(np.flatnonzero(y == k))[:400]
        imgs = [Image.fromarray(np.asarray(X[i], dtype=np.uint8)) for i in idx]
        res[f"PHI-Net test {s}"] = score(model, imgs, dev)
    keys = list(res["ours (real crack close-ups, should pass)"][0])
    table = {k: {g: share(rows, k) for g, rows in res.items()} | {"n": {g: len(rows) for g, rows in res.items()}}
             for k in keys}
    (ROOT / "data" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "results" / "scene_relax.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for k, v in table.items():
        print(k, {g: v[g] for g in res})


if __name__ == "__main__":
    main()
