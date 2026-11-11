"""Spalling or exposed bars on the photo of the whole element (D61). Spec: spec/features/photo2.md.
The model reads the photo; the written rule (threshold chosen on validation) decides "seen" or not."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "spalling_convnext_tiny_phinet3.pt"
ARCH = "convnext_tiny"
CLASSES = ["no_spalling", "spalling"]  # PHI-Net task 3 order: 0 non-spalling, 1 spalling
TEMPERATURE = 1.0  # fitted on the validation images (D61, data/processed/spalling_model.json, 11 Nov)
SPALL_MIN = 0.4  # chosen on the validation images: at least 90 % of spalling caught (D61); illustrative
CATCH_TARGET = 0.90
SIZE = 224
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load(path: Path = MODEL_FILE, device: str = "cpu"):
    if not Path(path).exists():
        return None
    import torch
    from experiments.element_baseline import build_model
    m = build_model(ARCH, pretrained=False, n_classes=len(CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def calibrate(prob: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=-1, keepdims=True)


def probability(model, photo: Image.Image, device: str = "cpu") -> float:
    """Calibrated probability of spalling, the mean of the photo and its mirror image."""
    import torch
    img = ImageOps.exif_transpose(photo).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        p = (torch.softmax(model(t), 1) + torch.softmax(model(t.flip(3)), 1)) / 2
    return float(calibrate(p.cpu().numpy(), TEMPERATURE)[0, 1])


def seen(model, photo: Image.Image, device: str = "cpu") -> tuple[bool | None, float | None]:
    """(spalling seen, probability); (None, None) without the model: not known, nothing raised by it."""
    if model is None:
        return None, None
    p = probability(model, photo, device)
    return p >= SPALL_MIN, p


def choose_spall_min(p_spall: np.ndarray, is_spall: np.ndarray, target: float = CATCH_TARGET,
                     candidates=tuple(round(0.05 * k, 2) for k in range(1, 20))) -> float:
    """The highest threshold that still catches at least `target` of the spalling photos. Validation only."""
    real = p_spall[is_spall]
    ok = [c for c in candidates if real.size and np.mean(real >= c) >= target]
    return max(ok) if ok else min(candidates)
