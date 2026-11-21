"""Train the autoencoder on sound concrete only (D89), set its thresholds on held-out sound tiles, and evaluate it.
Spec: spec/features/anomaly_ae.md. Usage, from the repository root:
    py src/anomaly/train.py            # cache the photos once, train, thresholds, evaluation
    py src/anomaly/train.py --epochs 2 # a quick run
Writes models/anomaly_ae.pt and models/anomaly_ae.json (never committed: GitHub Releases, D39)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from anomaly.model import AE, THRESHOLDS, TILE, WEIGHTS, heatmap, unusual  # noqa: E402

RAW = ROOT / "data" / "raw"
SEG = ROOT / "data" / "processed" / "segmentation"
CACHE = ROOT / "data" / "processed" / "anomaly"
SIDE = 128  # training images: short side scaled to this, centre square
HOLD_EVERY = 20  # every 20th photo of each source is held out (5 %) for the thresholds
PIXEL_PCT, TILE_PCT = 99.0, 95.0  # illustrative until calibrated
MAX_FALSE_PHOTOS = 0.02  # at most 2 % of held-out sound photos may show a defect area (illustrative)
MIN_HIT_PX = 10  # a crack counts as lit when at least this many of its pixels are unusual
FLOOR_PCTS = (99.0, 99.5, 99.9)  # candidate floors: percentiles of held-out sound pixel errors
RATIOS = (2.0, 3.0, 6.0)  # candidate "times the surface around it"


def _square(im: Image.Image) -> np.ndarray:
    im = im.convert("RGB")
    k = SIDE / min(im.size)
    im = im.resize((max(SIDE, round(im.width * k)), max(SIDE, round(im.height * k))), Image.BILINEAR)
    x0, y0 = (im.width - SIDE) // 2, (im.height - SIDE) // 2
    return np.asarray(im.crop((x0, y0, x0 + SIDE, y0 + SIDE)), dtype=np.uint8)


def sources():
    """(name, list of loaders) for every sound-concrete source of the spec."""
    out = {}
    out["metu_negative"] = [lambda p=p: Image.open(p) for p in sorted((RAW / "metu" / "Negative").glob("*.jpg"))]
    out["sdnet_uncracked_walls"] = [lambda p=p: Image.open(p) for p in sorted((RAW / "sdnet2018" / "W" / "UW").glob("*.jpg"))]
    t2 = RAW / "phinet" / "task2_damage_state"
    if (t2 / "X_train.npy").exists():
        x = np.load(t2 / "X_train.npy", mmap_mode="r")
        y = np.load(t2 / "y_train.npy").astype(int)
        out["phinet_undamaged"] = [lambda i=i: Image.fromarray(np.asarray(x[i])) for i in np.flatnonzero(y == 1)]
    ozg = []
    for p in sorted((SEG / "train" / "images").glob("ozg_*")):
        m = SEG / "train" / "masks" / p.name
        if m.exists() and not np.asarray(Image.open(m)).any():
            ozg.append(lambda p=p: Image.open(p))
    out["ozgenel_crack_free_tiles"] = ozg
    return out


def build_cache() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "train.npy").exists():
        return json.loads((CACHE / "counts.json").read_text(encoding="utf-8"))
    train, held, counts = [], [], {}
    for name, loaders in sources().items():
        n_train = n_held = 0
        for i, load in enumerate(loaders):
            try:
                a = _square(load())
            except OSError:
                continue
            if i % HOLD_EVERY == HOLD_EVERY - 1:
                held.append(a)
                n_held += 1
            else:
                train.append(a)
                n_train += 1
        counts[name] = {"train": n_train, "held_out": n_held}
        print(f"  {name}: {n_train} train, {n_held} held out", flush=True)
    np.save(CACHE / "train.npy", np.stack(train))
    np.save(CACHE / "held.npy", np.stack(held))
    (CACHE / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts


def _batch(arr, idx, rng, dev):
    ys, xs = rng.integers(0, SIDE - TILE + 1, len(idx)), rng.integers(0, SIDE - TILE + 1, len(idx))
    t = np.stack([arr[i, y:y + TILE, x:x + TILE] for i, y, x in zip(idx, ys, xs)])
    flip = rng.random(len(idx)) < 0.5
    t[flip] = t[flip, :, ::-1]
    flip = rng.random(len(idx)) < 0.5
    t[flip] = t[flip, ::-1]
    return torch.from_numpy(np.ascontiguousarray(t)).permute(0, 3, 1, 2).float().div(255).to(dev)


def train(epochs: int, dev: str, seed: int = 0) -> AE:
    arr = np.load(CACHE / "train.npy", mmap_mode="r")
    arr = np.asarray(arr)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    m = AE().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        m.train()
        order, total = rng.permutation(len(arr)), 0.0
        for i in range(0, len(order), 256):
            x = _batch(arr, order[i:i + 256], rng, dev)
            loss = ((m(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(x)
        sched.step()
        print(f"  epoch {ep + 1}/{epochs}: mean squared error {total / len(order):.5f}", flush=True)
    return m.eval()


@torch.no_grad()
def tile_errors(m: AE, arr: np.ndarray, dev: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixel errors and tile mean errors on fixed 64 px crops (four corners and the centre) of each image."""
    c = SIDE - TILE
    spots = [(0, 0), (0, c), (c, 0), (c, c), (c // 2, c // 2)]
    pix, tiles = [], []
    for i in range(0, len(arr), 128):
        part = np.stack([a[y:y + TILE, x:x + TILE] for a in arr[i:i + 128] for y, x in spots])
        t = torch.from_numpy(part).permute(0, 3, 1, 2).float().div(255).to(dev)
        e = ((m(t) - t) ** 2).mean(1).cpu().numpy()
        pix.append(e[:, ::4, ::4].ravel())  # every 4th pixel is enough for a percentile
        tiles.append(e.mean(axis=(1, 2)))
    return np.concatenate(pix), np.concatenate(tiles)


def _ozg_tiles(split: str):
    for p in sorted((SEG / split / "images").glob("ozg_*")):
        mp = SEG / split / "masks" / p.name
        if mp.exists():
            yield Image.open(p).convert("RGB"), np.asarray(Image.open(mp).convert("L")) > 0


@torch.no_grad()
def calibrate(m: AE, dev: str) -> dict:
    """The floor and the ratio of model.unusual(), judged on what the viewer shows (areas, after the area rules of
    src/anomaly/regions.py): at most MAX_FALSE_PHOTOS of held-out sound photos show any area, and among those the pair
    that lights up the most cracks on the Özgenel validation tiles (the test tiles stay unseen)."""
    from anomaly import regions
    held = np.load(CACHE / "held.npy")
    H = [(a, *heatmap(m, Image.fromarray(a), dev, side=None)) for a in held]
    allpix = np.concatenate([h.ravel() for _, h, _ in H])
    val = []
    for img, mk in _ozg_tiles("val"):
        h, _ = heatmap(m, img, dev)
        val.append((h, np.asarray(Image.fromarray(mk).resize((h.shape[1], h.shape[0]), Image.NEAREST))))
    best = None
    for fp in FLOOR_PCTS:
        floor = float(np.percentile(allpix, fp))
        for ratio in RATIOS:
            thr = {"pixel_floor": floor, "pixel_ratio": ratio}
            false = float(np.mean([bool(regions.find(a, unusual(h, thr))) for a, h, _ in H]))
            if false > MAX_FALSE_PHOTOS:
                continue
            cr = [int((unusual(h, thr) & mk).sum()) >= MIN_HIT_PX for h, mk in val if mk.any()]
            hit = float(np.mean(cr)) if cr else 0.0
            if best is None or hit > best["val_crack_tiles_lit"]:
                best = {"pixel_floor": floor, "pixel_floor_percentile": fp, "pixel_ratio": ratio,
                        "held_sound_photos_with_an_area": round(false, 4), "val_crack_tiles_lit": round(hit, 3)}
    if best is None:
        raise SystemExit("No floor and ratio keep the false areas under the limit: see the experiment note.")
    best["tile_threshold"] = float(np.percentile([t.mean() for _, _, t in H], TILE_PCT))
    best["tile_percentile"] = TILE_PCT
    return best


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def evaluate(m: AE, thr: dict, dev: str) -> dict:
    """Özgenel test tiles (never seen in training or calibration): does the heatmap rank crack pixels above sound
    ones, how many cracks does it light up, and how many crack-free tiles show a defect area (false alarms)."""
    from anomaly import regions
    s_all, l_all, lit, false_area, score_crack, score_sound = [], [], [], [], [], []
    for img, mk in _ozg_tiles("test"):
        heat, tiles = heatmap(m, img, dev)
        mask = np.asarray(Image.fromarray(mk).resize((heat.shape[1], heat.shape[0]), Image.NEAREST))
        u = unusual(heat, thr)
        s_all.append(heat[::2, ::2].ravel())
        l_all.append(mask[::2, ::2].ravel())
        score = float(tiles.mean()) / thr["tile_threshold"]
        if mask.any():
            lit.append(int((u & mask).sum()) >= MIN_HIT_PX)
            score_crack.append(score)
        else:
            false_area.append(bool(regions.find(np.asarray(img.resize((heat.shape[1], heat.shape[0]))), u)))
            score_sound.append(score)
    s, lab = np.concatenate(s_all), np.concatenate(l_all)
    return {"ozgenel_test_tiles": len(lit) + len(false_area),
            "pixel_auroc_crack_vs_sound": round(auroc(s, lab), 3),
            "crack_tiles_lit": f"{sum(lit)} of {len(lit)}",
            "crack_free_tiles_with_a_defect_area": f"{sum(false_area)} of {len(false_area)}",
            "tiles_with_crack_score_above_1": round(float(np.mean(np.array(score_crack) > 1)), 3),
            "tiles_without_crack_score_above_1": round(float(np.mean(np.array(score_sound) > 1)), 3)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--recalibrate", action="store_true", help="keep the trained weights; thresholds and evaluation only")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print("Caching the sound-concrete photos (once)...", flush=True)
    counts = build_cache()
    if args.recalibrate:
        m = AE().to(dev)
        m.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True))
        m.eval()
        epochs = json.loads(THRESHOLDS.read_text(encoding="utf-8")).get("epochs") if THRESHOLDS.exists() else None
    else:
        print(f"Training on {dev}...", flush=True)
        m = train(args.epochs, dev)
        epochs = args.epochs
        WEIGHTS.parent.mkdir(exist_ok=True)
        torch.save(m.state_dict(), WEIGHTS)
    print("Setting the thresholds on held-out sound photos and the validation tiles...", flush=True)
    thr = {"version": "anomaly-ae-0.2 (illustrative thresholds)", "trained": time.strftime("%Y-%m-%d"),
           "epochs": epochs, "sources": counts, **calibrate(m, dev)}
    print("Evaluating on the Özgenel test tiles...", flush=True)
    thr["evaluation"] = evaluate(m, thr, dev)
    THRESHOLDS.write_text(json.dumps(thr, indent=1), encoding="utf-8")
    print(json.dumps(thr["evaluation"], indent=1))
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {WEIGHTS}, {THRESHOLDS}")


if __name__ == "__main__":
    main()
