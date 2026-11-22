"""Our opening and sliding (D99, src/measure/kinematics.py) against the known displacements of the published toy
cracks of Pantoja-Rosero et al. 2022 [48] (Zenodo 6632071, CC BY 4.0). Each toy crack is a binary mask made by moving
one side of a drawn crack by a known rigid movement (H_params: rotation, x and y shift in pixels). Usage:
    py src/experiments/kinematics_eval.py
Writes data/results/kinematics/toy_eval.json (local). The dataset stays in data/raw/epfl_kinematics (not committed)."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from element.photo2 import trace_network, _length  # noqa: E402
from measure.kinematics import from_path  # noqa: E402

DATA = ROOT / "data" / "raw" / "epfl_kinematics"
OUT = ROOT / "data" / "results" / "kinematics"


def truth(path: np.ndarray, shift: np.ndarray) -> tuple[float, float]:
    """The true opening and |sliding| along the crack's main direction (rotation ignored: it is small in the toys)."""
    t = path[-1] - path[0]
    t = t / np.hypot(*t)
    n = np.array([-t[1], t[0]])
    return abs(float(shift @ n)), abs(float(shift @ t))


def main() -> None:
    import hashlib
    rows, seen = [], set()
    for png in sorted(DATA.glob("toy_examples*/*.png")):
        hp = png.with_name(png.stem + "_H_params.npy")
        if not hp.exists():
            continue
        h = np.load(hp, allow_pickle=True).astype(float).ravel()
        mask = np.asarray(Image.open(png).convert("L")) > 127
        key = hashlib.md5(mask.tobytes() + h.tobytes()).hexdigest()  # the folders repeat the same cracks
        if key in seen:
            continue
        seen.add(key)
        nets = trace_network(mask, 20)
        if not nets:
            continue
        path = max(nets, key=_length)
        k = from_path(mask, path)
        op, sl = truth(path, h[1:3])
        rows.append({"case": f"{png.parent.name}/{png.stem}", "rotation_rad": round(h[0], 4), "true_opening": round(op, 2),
                     "true_sliding": round(sl, 2), "opening": k.opening_median and round(k.opening_median, 2),
                     "sliding": k.sliding_median and round(k.sliding_median, 2), "reading": k.reading,
                     "stations": len(k.stations)})
        print(rows[-1], flush=True)
    ok = [r for r in rows if r["opening"] is not None]
    rd = [r for r in ok if r["sliding"] is not None]
    summary = {
        "cases": len(rows), "opening_read": len(ok), "sliding_read": len(rd),
        "opening_mean_abs_error_px": round(float(np.mean([abs(r["opening"] - r["true_opening"]) for r in ok])), 2) if ok else None,
        "opening_bias_px": round(float(np.mean([r["opening"] - r["true_opening"] for r in ok])), 2) if ok else None,
        "true_opening_median_px": round(float(np.median([r["true_opening"] for r in ok])), 2) if ok else None,
        "reading_right": sum((r["true_sliding"] / max(r["true_opening"], 1e-9) < 0.3) == (r["reading"] == "mostly opening")
                             for r in rd),
        "sliding_mean_abs_error_px": round(float(np.mean([abs(r["sliding"] - r["true_sliding"]) for r in rd])), 2) if rd else None,
    }
    print(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toy_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
