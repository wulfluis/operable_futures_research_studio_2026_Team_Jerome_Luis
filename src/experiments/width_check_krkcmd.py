"""Our width rule against 19,098 crack widths measured by hand (D108): the krkCMd data of Jakubowski and Tomczak [113]
(Zenodo 14568863, CC BY 4.0). Each row is a brightness profile across a crack (501 samples of a 6400 dpi scan:
0.00397 mm per sample) with the width an operator read on it (MANwidth, micrometres), and the authors' own two
methods for comparison (DLMwidth: their deep-learning metasensor; AEDwidth: their analytic edge detector).

Our rule, as in src/measure/crack.py (profile_width, grey case): the darkest point of the crack, the wall's brightness
on each side, and the two edges where the brightness is half-way back to the wall (half-depth). In the app the wall
level is read just outside the crack model's mask; here, without a mask, it is the median of the brighter half of the
profile on each side of the darkest point (an assumption, said so in the note).

Then the same profiles are averaged down to coarser pixels, as a phone sees the crack (0.02, 0.05 and 0.1 mm per
pixel), and measured again: how the error grows as the crack gets fewer pixels across ([46]: 3 pixels minimum, 6 to
8 best). Usage, from the repository root:
    py src/experiments/width_check_krkcmd.py
Writes data/results/width_check/krkcmd.json (local). The data stays in data/raw/krkcmd (not committed)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "raw" / "krkcmd" / "krkCMd_table.csv"
OUT = ROOT / "data" / "results" / "width_check"
MM_PER_SAMPLE = 25.4 / 6400  # the scans' 6400 dpi
PIXEL_SIZES_MM = (MM_PER_SAMPLE, 0.02, 0.05, 0.1)


def half_depth_width(g: np.ndarray, depth: float = 0.5) -> float | None:
    """Width in samples between the two half-depth crossings around the darkest point; sub-sample by interpolation.
    None when the crack is not darker than the wall (healed, filled)."""
    g = np.asarray(g, float)
    n = len(g)
    if n < 5:
        return None
    ic = int(np.argmin(g))
    left, right = g[:ic], g[ic + 1:]
    if len(left) < 2 or len(right) < 2:
        return None
    bl = float(np.median(np.sort(left)[len(left) // 2:]))
    br = float(np.median(np.sort(right)[len(right) // 2:]))
    c = float(g[ic])
    if min(bl, br) - c < 5:  # no contrast: the app's MIN_CONTRAST idea
        return None

    def cross(direction, level):
        i = ic
        while 0 < i < n - 1:
            j = i + direction
            if g[j] >= level:
                return i + direction * (level - g[i]) / max(g[j] - g[i], 1e-9)
            i = j
        return None
    a, b = cross(-1, bl - depth * (bl - c)), cross(1, br - depth * (br - c))  # depth 0.5: half-depth
    if a is None or b is None:
        return None
    return float(b - a)


def coarse(g: np.ndarray, factor: float, offset: float) -> np.ndarray:
    """The profile seen with pixels `factor` samples wide (box average), starting at `offset` samples: a phone pixel
    falls anywhere on the crack."""
    k = max(1, int(round(factor)))
    start = int(round(offset * k)) % k
    g = g[start:]
    m = len(g) // k
    return g[: m * k].reshape(m, k).mean(axis=1)


def main() -> None:
    import pandas as pd
    d = pd.read_csv(TABLE)
    cols = [c for c in d.columns if c.startswith("x")]
    prof = d[cols].to_numpy(float)
    man = d["MANwidth"].to_numpy(float) / 1000  # mm
    dlm = d["DLMwidth"].to_numpy(float) / 1000
    aed = d["AEDwidth"].to_numpy(float) / 1000
    rng = np.random.default_rng(26)
    keep = man > 0  # healed cracks (width 0 by hand) are left out: there is no crack to measure
    out = {"rows": int(len(d)), "rows_with_a_crack": int(keep.sum()),
           "hand_width_mm": {"median": round(float(np.median(man[keep])), 3),
                             "p10": round(float(np.percentile(man[keep], 10)), 3),
                             "p90": round(float(np.percentile(man[keep], 90)), 3)},
           "authors": {}, "ours": {}}
    for name, v in (("deep learning metasensor (DLM)", dlm), ("analytic edge detector (AED)", aed)):
        e = v[keep] - man[keep]
        out["authors"][name] = {"mean_abs_error_mm": round(float(np.mean(np.abs(e))), 4),
                                "bias_mm": round(float(np.mean(e)), 4),
                                "within_10_percent": round(float(np.mean(np.abs(e) <= 0.1 * man[keep])), 3)}
    for px in PIXEL_SIZES_MM:
        factor = px / MM_PER_SAMPLE
        w = np.full(len(d), np.nan)
        for i in np.flatnonzero(keep):
            g = prof[i] if factor <= 1.01 else coarse(prof[i], factor, rng.random())
            r = half_depth_width(g)
            if r is not None:
                w[i] = r * (MM_PER_SAMPLE if factor <= 1.01 else px)
        ok = keep & ~np.isnan(w)
        e = w[ok] - man[ok]
        across = man[ok] / px
        bands = {}
        for lo, hi in ((0, 3), (3, 6), (6, 1e9)):
            s = (across >= lo) & (across < hi)
            if s.sum():
                bands[f"{lo}-{'more' if hi > 1e8 else hi} px across"] = {
                    "n": int(s.sum()), "mean_abs_error_mm": round(float(np.mean(np.abs(e[s]))), 4),
                    "bias_mm": round(float(np.mean(e[s])), 4)}
        out["ours"][f"{px:.4f} mm per pixel"] = {
            "measured": int(ok.sum()), "not_read": int(keep.sum() - ok.sum()),
            "mean_abs_error_mm": round(float(np.mean(np.abs(e))), 4), "bias_mm": round(float(np.mean(e)), 4),
            "within_one_pixel": round(float(np.mean(np.abs(e) <= px)), 3),
            "within_10_percent": round(float(np.mean(np.abs(e) <= 0.1 * man[ok])), 3), "by_pixels_across": bands}
        print(f"{px:.4f} mm/px", json.dumps(out["ours"][f"{px:.4f} mm per pixel"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "krkcmd.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("rows", "rows_with_a_crack", "hand_width_mm", "authors")}, indent=1))


def tune(depths=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)) -> dict:
    """The edge level (share of the crack's depth below the wall) chosen on half the specimen images and checked on
    the other half (split by series and image, so no crack is on both sides); at the scan and at 0.05 mm per pixel."""
    import pandas as pd
    d = pd.read_csv(TABLE)
    cols = [c for c in d.columns if c.startswith("x")]
    prof = d[cols].to_numpy(float)
    man = d["MANwidth"].to_numpy(float) / 1000
    groups = (d["Series"].astype(str) + "|" + d["Image"].astype(str)).to_numpy()
    uniq = sorted(set(groups))
    rng = np.random.default_rng(26)
    pick = set(rng.permutation(uniq)[: len(uniq) // 2])
    choose = np.array([g in pick for g in groups]) & (man > 0)
    check = ~np.array([g in pick for g in groups]) & (man > 0)
    res = {"groups": len(uniq), "choose_rows": int(choose.sum()), "check_rows": int(check.sum()), "by_depth": {}}
    offsets = rng.random(len(d))
    for px in (MM_PER_SAMPLE, 0.05):
        factor = px / MM_PER_SAMPLE
        for dep in depths:
            w = np.full(len(d), np.nan)
            for i in np.flatnonzero(man > 0):
                g = prof[i] if factor <= 1.01 else coarse(prof[i], factor, offsets[i])
                r = half_depth_width(g, dep)
                if r is not None:
                    w[i] = r * (MM_PER_SAMPLE if factor <= 1.01 else px)
            row = {}
            for name, sel in (("choose", choose), ("check", check)):
                ok = sel & ~np.isnan(w)
                e = w[ok] - man[ok]
                row[name] = {"mean_abs_error_mm": round(float(np.mean(np.abs(e))), 4), "bias_mm": round(float(np.mean(e)), 4),
                             "within_10_percent": round(float(np.mean(np.abs(e) <= 0.1 * man[ok])), 3), "read": int(ok.sum())}
            res["by_depth"][f"{px:.4f} | {dep}"] = row
            print(f"{px:.4f} mm/px depth {dep}: {row}", flush=True)
    (OUT / "krkcmd_tune.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    sys.exit(tune() and None if "--tune" in sys.argv else main())
