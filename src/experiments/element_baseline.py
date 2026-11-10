"""Element model, first test: Grounding DINO without training against a ResNet-18 trained on PHI-Net task 6.

Spec: spec/features/element_model.md (D44). Usage, from the repository root:
    python src/experiments/element_baseline.py data/raw/phinet/task6_component_type --method both
    python src/experiments/element_baseline.py data/raw/phinet/task6_component_type --method zero-shot --limit 100
The folder is the one written by src/experiments/phinet_prepare.py (X_train.npy, y_train.npy, X_test.npy,
y_test.npy, README.txt). Results go to data/processed/element_baseline.json; the trained weights to models/
(both ignored by git). Thresholds are illustrative.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.phinet_prepare import class_names
from triage.matrix import CONFIDENCE_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["beam", "column", "wall", "others"]  # our order; PHI-Net task 6 numbers them 0 beam, 1 column, 2 others, 3 wall
PROMPT_WORDS = ["column", "beam", "wall"]  # 'others' cannot be asked for
GD_MODEL = "IDEA-Research/grounding-dino-tiny"
GD_BOX_THRESHOLD = 0.15  # keep weak boxes so every word gets a score
GD_SCORE_THRESHOLDS = (0.25, 0.35, 0.45)  # reported side by side; illustrative
UNSURE = "unsure"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


# ---------- data ----------

def phinet_names(readme: str) -> dict[int, str]:
    """PHI-Net class number -> our class name, from the task README ('0: Beam' ...)."""
    out = {}
    for k, name in class_names(readme).items():
        n = name.strip().lower()
        out[k] = next((c for c in CLASSES if c.rstrip("s") in n), "others")
    return out


def load_task6(folder: Path, limit: int | None = None, seed: int = 0):
    names = phinet_names((folder / "README.txt").read_text(encoding="utf-8", errors="replace"))
    data = {}
    for split in ("train", "test"):
        x = np.load(folder / f"X_{split}.npy", mmap_mode="r")
        y = np.array([CLASSES.index(names[int(v)]) for v in np.load(folder / f"y_{split}.npy")])
        idx = np.arange(len(y))
        if limit and limit < len(y):
            idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
        data[split] = (x, y, idx)
    return data


# ---------- metrics ----------

def report(truth: list[str], pred: list[str]) -> dict:
    """Confusion with 'unsure' as a column; accuracy on confident answers; coverage; recall per class."""
    cols = CLASSES + [UNSURE]
    conf = {t: {p: 0 for p in cols} for t in CLASSES}
    for t, p in zip(truth, pred):
        conf[t][p] += 1
    n = len(truth)
    confident = [(t, p) for t, p in zip(truth, pred) if p != UNSURE]
    correct = sum(t == p for t, p in confident)

    def share(a, b):
        return round(a / b, 3) if b else None

    col = conf["column"]
    others = conf["others"]
    return {"n": n, "coverage": share(len(confident), n), "accuracy_confident": share(correct, len(confident)),
            "recall": {c: share(conf[c][c], sum(conf[c].values())) for c in CLASSES},
            "column_confidently_wrong": share(sum(v for p, v in col.items() if p not in ("column", UNSURE)),
                                              sum(col.values())),
            "others_called_element": share(sum(others[p] for p in ("beam", "column", "wall")),
                                           sum(others.values())),
            "confusion": conf}


def decide(scores: dict[str, float], threshold: float) -> str:
    """Best class if its score reaches the threshold, else unsure (rule 2)."""
    if not scores:
        return UNSURE
    best = max(scores, key=scores.get)
    return best if scores[best] >= threshold else UNSURE


def markdown(name: str, r: dict) -> str:
    rows = [f"**{name}**: n = {r['n']}, coverage {r['coverage']}, accuracy on confident answers "
            f"{r['accuracy_confident']}, column confidently wrong {r['column_confidently_wrong']}, others called an "
            f"element {r['others_called_element']}", "", "| True \\ predicted | " + " | ".join(CLASSES + [UNSURE]) + " |",
            "|---" * (len(CLASSES) + 2) + "|"]
    rows += [f"| {t} | " + " | ".join(str(r["confusion"][t][p]) for p in CLASSES + [UNSURE]) + " |" for t in CLASSES]
    return "\n".join(rows)


# ---------- A: Grounding DINO, no training ----------

def device_name(choice: str = "auto") -> str:
    import torch
    if choice != "auto":
        return choice
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def zero_shot_scores(images, device: str, batch: int = 8) -> list[dict[str, float]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(GD_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL).to(device).eval()
    text = ". ".join(f"a {w}" for w in PROMPT_WORDS) + "."
    out = []
    for i in range(0, len(images), batch):
        ims = [Image.fromarray(np.asarray(im)) for im in images[i:i + batch]]
        inputs = processor(images=ims, text=[text] * len(ims), return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs)
        sizes = [im.size[::-1] for im in ims]
        try:
            post = processor.post_process_grounded_object_detection(
                res, inputs.input_ids, threshold=GD_BOX_THRESHOLD, text_threshold=GD_BOX_THRESHOLD, target_sizes=sizes)
        except TypeError:  # older transformers
            post = processor.post_process_grounded_object_detection(
                res, inputs.input_ids, box_threshold=GD_BOX_THRESHOLD, text_threshold=GD_BOX_THRESHOLD,
                target_sizes=sizes)
        for r in post:
            labels = r.get("text_labels", r.get("labels"))
            s = {}
            for lab, sc in zip(labels, r["scores"].tolist()):
                word = next((w for w in PROMPT_WORDS if w in str(lab).lower()), None)
                if word:
                    s[word] = max(s.get(word, 0.0), sc)
            out.append(s)
        print(f"  zero-shot {min(i + batch, len(images))}/{len(images)}", end="\r", flush=True)
    print()
    return out


# ---------- B: ResNet-18, trained ----------

def _tensor(x: np.ndarray):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x)).float().div(255).permute(0, 3, 1, 2)
    return (t - torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)) / torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)


def train_resnet(x, y, idx, device: str, epochs: int = 8, batch: int = 64, seed: int = 0, log=print):
    """Fine-tune ResNet-18; 10 % of train picks the best epoch. Augmentation: horizontal flip and small crops only
    (no vertical flip or rotation: a beam turned upright looks like a column)."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet18_Weights, resnet18

    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_val = max(1, len(idx) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), dtype=torch.float32, device=device)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for i in range(0, len(tr), batch):
            b = np.sort(tr[i:i + batch])
            xb = _tensor(np.asarray(x[b]))
            if rng.random() < 0.5:
                xb = xb.flip(3)
            k = int(rng.integers(0, 17))  # small random crop, resized back
            if k:
                o = int(rng.integers(0, k + 1))
                xb = F.interpolate(xb[:, :, o:o + 224 - k, o:o + 224 - k], size=(224, 224), mode="bilinear",
                                   align_corners=False)
            loss = F.cross_entropy(model(xb.to(device)), torch.from_numpy(y[b]).to(device), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = float(np.mean(predict_resnet(model, x, val, device).argmax(1) == y[val]))
        log(f"  epoch {ep + 1}/{epochs}: validation accuracy {acc:.3f} ({time.time() - t0:.0f} s)")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def predict_resnet(model, x, idx, device: str, batch: int = 128) -> np.ndarray:
    """Probabilities in the order of idx. The images are read in sorted order (the files are memory-mapped) and the
    rows put back, so a shuffled idx, as in the validation split, still lines up with its labels (bug until 10 Nov:
    the validation accuracy compared sorted predictions with unsorted labels, so the best epoch was picked at random)."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            part = np.asarray(idx[i:i + batch])
            order = np.argsort(part)
            prob = torch.softmax(model(_tensor(np.asarray(x[part[order]])).to(device)), 1).cpu().numpy()
            back = np.empty_like(prob)
            back[order] = prob
            out.append(back)
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ---------- main ----------

def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--method", choices=["zero-shot", "trained", "both"], default="both")
    p.add_argument("--limit", type=int, help="test (and train) on a random subset of this size, for a quick run")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "element_baseline.json")
    args = p.parse_args(argv)
    try:  # python.org Python on macOS ships without root certificates: model downloads would fail
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    data = load_task6(args.folder, args.limit)
    xt, yt, it = data["test"]
    truth = [CLASSES[v] for v in yt[it]]
    dev = device_name(args.device)
    print(f"device {dev}; test images {len(it)}; train images {len(data['train'][2])}")
    results = {"data": str(args.folder), "limit": args.limit, "device": dev}

    if args.method in ("zero-shot", "both"):
        t0 = time.time()
        scores = zero_shot_scores(xt[np.sort(it)], dev)
        results["zero_shot"] = {"model": GD_MODEL, "seconds": round(time.time() - t0),
                                "by_threshold": {str(th): report(truth, [decide(s, th) for s in scores])
                                                 for th in GD_SCORE_THRESHOLDS}}
        for th in GD_SCORE_THRESHOLDS:
            print(markdown(f"A. Grounding DINO, threshold {th}", results["zero_shot"]["by_threshold"][str(th)]), "\n")

    if args.method in ("trained", "both"):
        import torch
        x, y, idx = data["train"]
        t0 = time.time()
        model, val_acc = train_resnet(x, y, idx, dev, epochs=args.epochs)
        prob = predict_resnet(model, xt, it, dev)
        pred = [CLASSES[k] if pr[k] >= CONFIDENCE_THRESHOLD else UNSURE for pr, k in zip(prob, prob.argmax(1))]
        results["trained"] = {"model": "resnet18 ImageNet, fine-tuned", "epochs": args.epochs,
                              "validation_accuracy": round(val_acc, 3), "seconds": round(time.time() - t0),
                              "threshold": CONFIDENCE_THRESHOLD, "report": report(truth, pred),
                              "report_no_threshold": report(truth, [CLASSES[k] for k in prob.argmax(1)])}
        print(markdown(f"B. ResNet-18 trained, threshold {CONFIDENCE_THRESHOLD}", results["trained"]["report"]))
        weights = ROOT / "models" / "element_resnet18_phinet6.pt"
        weights.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), weights)
        results["trained"]["weights"] = str(weights.relative_to(ROOT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"written {args.out}")
    return results


if __name__ == "__main__":
    main()
