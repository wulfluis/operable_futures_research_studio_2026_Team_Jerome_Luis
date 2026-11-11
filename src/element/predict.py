"""Element model in the app: which element a photo of the whole element shows (column, beam, wall, others).
Spec: spec/features/app_classification.md (D53); model and calibration: spec/features/element_model.md (D52).

The model only suggests: the "doubt means column" rule (src/triage/element_rule.py) turns its probabilities into an
answer, and a field worker confirms it. Residents are never asked (D25).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from triage.element_rule import ElementAnswer, element_answer

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "element_convnext_tiny_phinet6_v2.pt"
ARCH = "convnext_tiny"
CLASSES = ["beam", "column", "wall", "others"]  # the order the model was trained with (element_baseline.CLASSES)
TEMPERATURE = 0.76  # fitted on the validation images (D52, data/processed/element_calibrated.json)
COLUMN_DOUBT = 0.5  # chosen on the validation images (D52)
SIZE = 224  # PHI-Net image size; the whole photo is squeezed to it (assumption: PHI-Net resized whole photos)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
TO_MATRIX = {"column": "column", "beam": "beam", "wall": "load_bearing_wall"}  # others/unsure: no matrix element


def load(path: Path = MODEL_FILE, device: str = "cpu"):
    """The trained network, or None when the file is not on this laptop (the field worker then chooses)."""
    if not Path(path).exists():
        return None
    import torch
    from experiments.element_baseline import build_model
    model = build_model(ARCH, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def calibrate(prob: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=-1, keepdims=True)


def probabilities(model, photo: Image.Image, device: str = "cpu") -> dict[str, float]:
    """Calibrated probabilities per class, the mean of the photo and its mirror image (as tested)."""
    import torch
    img = ImageOps.exif_transpose(photo).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        p = (torch.softmax(model(t), 1) + torch.softmax(model(t.flip(3)), 1)) / 2
    return dict(zip(CLASSES, map(float, calibrate(p.cpu().numpy()[0]))))


def read_element(model, photo: Image.Image, device: str = "cpu") -> ElementAnswer:
    """The element answer for one photo; no model gives 'unsure' with the reason."""
    if model is None:
        return ElementAnswer("unsure", False, 0.0, "The element model is not installed on this laptop: a field "
                                                   "worker chooses the element.")
    return element_answer(probabilities(model, photo, device), COLUMN_DOUBT)


def matrix_element(answer: ElementAnswer | None) -> str | None:
    """The risk matrix's element key for a model answer; None when the model gives none."""
    return TO_MATRIX.get(answer.element) if answer else None
