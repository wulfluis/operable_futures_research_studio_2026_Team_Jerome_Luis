"""Train the crack U-Net on the prepared public sets. Spec: spec/features/segmentation.md.

Usage, from the repository root (after src/segmentation/prepare.py):
    py src/segmentation/train.py data/processed/segmentation models/unet_crack.pt
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, as the encoder was pre-trained
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """H x W x 3 uint8 -> 3 x H x W normalised float."""
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def load_checkpoint(path: Path, device: str = "cpu") -> torch.nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    return model.to(device).eval()


class CrackSet(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False, limit: int | None = None,
                 exclude: tuple[str, ...] = (), repeat: dict[str, int] | None = None):
        """exclude: file-name prefixes left out; repeat: prefix -> times each such photo is shown per epoch."""
        files = [f for f in sorted((root / split / "images").glob("*.png"))[:limit] if not f.name.startswith(exclude)]
        repeat = repeat or {}
        self.files = [f for f in files for _ in range(next((n for p, n in repeat.items() if f.name.startswith(p)), 1))]
        self.masks = root / split / "masks"
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = np.array(Image.open(self.files[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks / self.files[i].name)) > 0
        if self.augment:
            if self.rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if self.rng.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = int(self.rng.integers(4))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            gain, shift = self.rng.uniform(0.75, 1.25), self.rng.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)
        return to_tensor(img), torch.from_numpy(mask[None].astype(np.float32).copy())


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


@torch.no_grad()
def val_f1(model, loader, device) -> float:
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(x.to(device, memory_format=torch.channels_last)) > 0
        y = y.to(device) > 0.5
        tp += (pred & y).sum().item(); fp += (pred & ~y).sum().item(); fn += (~pred & y).sum().item()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--log", type=Path, default=Path("data/results/segmentation/train_log.csv"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="images per split, for a quick check")
    ap.add_argument("--exclude", nargs="*", default=[], help="file-name prefixes to leave out, e.g. metu_ sdnet_")
    ap.add_argument("--repeat", nargs="*", default=[], help="prefix=N: show those photos N times per epoch")
    args = ap.parse_args(argv)
    exclude = tuple(args.exclude)
    repeat = {p: int(n) for p, n in (r.split("=") for r in args.repeat)}

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = DataLoader(CrackSet(args.data, "train", augment=True, limit=args.limit, exclude=exclude, repeat=repeat),
                       args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val = DataLoader(CrackSet(args.data, "val", limit=args.limit), args.batch, num_workers=args.workers,
                     persistent_workers=args.workers > 0)
    model = build_model().to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train), pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_f1", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            start, total = time.time(), 0.0
            for x, y in train:
                x, y = x.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(x)
                loss = bce(logits.float(), y) + dice_loss(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
            f1 = val_f1(model, val, device)
            seconds = time.time() - start
            log.writerow([epoch, round(total / len(train), 4), round(f1, 4), round(seconds)])
            f.flush()
            print(f"epoch {epoch:2d}  loss {total / len(train):.4f}  val F1 {f1:.4f}  {seconds:.0f}s", flush=True)
            if f1 > best:
                best = f1
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": f1, "size": 416}, args.out)
    print(f"best val F1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
