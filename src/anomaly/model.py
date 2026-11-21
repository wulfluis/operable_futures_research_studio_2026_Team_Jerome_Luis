"""The autoencoder and its heatmap (D89). Spec: spec/features/anomaly_ae.md.

Trained only on sound concrete, it redraws any tile as sound concrete; where the photo differs, the squared error
between the photo and the redraw lights up (as in the team's weld hackathon model)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

TILE = 64  # px, the autoencoder's input (colour, for rust)
CODE = 64  # numbers in the code: narrow, so it can only redraw what sound concrete looks like (assumption)
WORK_SIDE = 768  # px, the photo's long side for the heatmap
STEP = 16  # px between tiles (overlapping, so every pixel is seen by several tiles)
ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "anomaly_ae.pt"
THRESHOLDS = ROOT / "models" / "anomaly_ae.json"


class AE(nn.Module):
    def __init__(self, code: int = CODE):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))  # noqa: E731
        d = lambda i, o: nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())  # noqa: E731
        self.enc = nn.Sequential(c(3, 32), c(32, 64), c(64, 128), c(128, 128), nn.Flatten(), nn.Linear(128 * 16, code))
        self.dec_in = nn.Linear(code, 128 * 16)
        self.dec = nn.Sequential(d(128, 128), d(128, 64), d(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.dec_in(self.enc(x)).view(-1, 128, 4, 4))


def work_image(img: Image.Image, side: int = WORK_SIDE) -> Image.Image:
    im = img.convert("RGB")
    k = side / max(im.size)
    return im.resize((max(TILE, round(im.width * k)), max(TILE, round(im.height * k))), Image.BILINEAR)


def _starts(n: int) -> list[int]:
    s = list(range(0, max(1, n - TILE + 1), STEP))
    if s[-1] != n - TILE:
        s.append(n - TILE)  # the last tile reaches the edge, so every pixel is covered
    return s


LOCAL_PX = 41  # px at WORK_SIDE: the surface around a pixel, for "stands out from its surroundings"


def unusual(heat: np.ndarray, thr: dict) -> np.ndarray:
    """Pixels that look different from sound concrete: above a small floor AND several times the error of the surface
    around them. Relative, so a crack on smooth plaster (small errors everywhere) is seen, and rough concrete (large
    errors everywhere) is not flagged all over. Both values are set on held-out sound concrete (src/anomaly/train.py)."""
    from scipy import ndimage
    around = ndimage.uniform_filter(heat, LOCAL_PX)
    return (heat > thr["pixel_floor"]) & (heat > thr["pixel_ratio"] * (around + 1e-6))


@torch.no_grad()
def heatmap(model: AE, img: Image.Image, device: str = "cpu", batch: int = 256,
            side: int | None = WORK_SIDE) -> tuple[np.ndarray, np.ndarray]:
    """(per-pixel error, per-tile mean errors) for the photo at `side` (None: as it is). Error = squared difference
    between the photo and the redraw, averaged over colour and over every tile that covers the pixel."""
    x = np.asarray(work_image(img, side) if side else img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = x.shape
    boxes = [(y, xx) for y in _starts(h) for xx in _starts(w)]
    acc, cnt, tile_err = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), []
    model.eval()
    for i in range(0, len(boxes), batch):
        part = boxes[i:i + batch]
        t = torch.from_numpy(np.stack([x[y:y + TILE, xx:xx + TILE] for y, xx in part])).permute(0, 3, 1, 2).to(device)
        e = ((model(t) - t) ** 2).mean(1).cpu().numpy()  # (n, 64, 64)
        for (y, xx), ee in zip(part, e):
            acc[y:y + TILE, xx:xx + TILE] += ee
            cnt[y:y + TILE, xx:xx + TILE] += 1
        tile_err.extend(e.mean(axis=(1, 2)).tolist())
    return acc / np.maximum(cnt, 1), np.asarray(tile_err)


_lock = threading.Lock()
_cache: dict = {}


def load(device: str | None = None) -> tuple[AE, dict, str] | None:
    """The trained autoencoder and its thresholds, loaded once; None when the files are missing (then nothing is
    shown, the app works as before)."""
    with _lock:
        if "m" not in _cache:
            if not (WEIGHTS.exists() and THRESHOLDS.exists()):
                _cache["m"] = None
            else:
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                m = AE()
                m.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True))
                _cache["m"] = (m.to(dev).eval(), json.loads(THRESHOLDS.read_text(encoding="utf-8")), dev)
        return _cache["m"]
