"""Scene check: is the photo a close-up (P), a whole element (O) or a whole building (S)? Spec: spec/features/scene_gate.md
(D54). The model reads the scale; written rules decide what happens: a crack photo that is not a close-up is a Retake,
and an element photo must not be a close-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "models" / "scene_convnext_tiny_phinet1.pt"
ARCH = "convnext_tiny"
SCENES = ["object", "close_up", "structure"]  # PHI-Net task 1 order: 0 object (O), 1 pixel (P), 2 structural (S)
NAMES = {"object": "a whole element", "close_up": "a close-up of a surface", "structure": "a whole building"}
TEMPERATURE = 0.76  # fitted on the validation images (D54, data/processed/scene_gate.json, 11 Nov)
CLOSE_UP_MIN = 0.1  # chosen on the validation images (D54): at most 5 % of close-ups sent to retake; illustrative
CLOSE_UP_RETAKE_TARGET = 0.05  # at most this share of real close-ups sent to retake (illustrative)
SIZE = 224
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
RETAKE_WIDE = ("This looks like {seen}, not a close-up of the wall: hold the phone 20 to 40 cm from the crack so the "
               "wall fills the photo.")
RETAKE_CLOSE = "This is a close-up: step back so the whole column, beam or wall is in the photo."


@dataclass
class SceneAnswer:
    scene: str | None  # the most likely scene, or None without a model
    close_up: float | None  # calibrated probability of a close-up
    ok: bool
    reason: str

    def to_dict(self) -> dict:
        return {"scene": self.scene, "close_up_probability": None if self.close_up is None else round(self.close_up, 3),
                "ok": self.ok, "reason": self.reason, "close_up_min": CLOSE_UP_MIN}


def load(path: Path = MODEL_FILE, device: str = "cpu"):
    """The scene network, or None when the file is not on this laptop (then no scene check, as before D54)."""
    if not Path(path).exists():
        return None
    import torch
    from experiments.element_baseline import build_model
    m = build_model(ARCH, pretrained=False, n_classes=len(SCENES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def calibrate(prob: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=-1, keepdims=True)


def probabilities(model, photo: Image.Image, device: str = "cpu") -> dict[str, float]:
    import torch
    img = ImageOps.exif_transpose(photo).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    with torch.no_grad():
        p = (torch.softmax(model(t), 1) + torch.softmax(model(t.flip(3)), 1)) / 2
    return dict(zip(SCENES, map(float, calibrate(p.cpu().numpy()[0], TEMPERATURE))))


def crack_photo_check(probs: dict[str, float] | None, close_up_min: float = CLOSE_UP_MIN) -> SceneAnswer:
    """A crack photo must be a close-up; otherwise Retake (rule 2: an unusable photo means retake)."""
    if probs is None:
        return SceneAnswer(None, None, True, "No scene check (model not installed).")
    best = max(probs, key=probs.get)
    if probs["close_up"] >= close_up_min:
        return SceneAnswer(best, probs["close_up"], True, f"A close-up ({probs['close_up']:.0%}).")
    seen = NAMES["structure" if probs["structure"] >= probs["object"] else "object"]
    return SceneAnswer(best, probs["close_up"], False, RETAKE_WIDE.format(seen=seen))


def element_photo_check(probs: dict[str, float] | None) -> SceneAnswer:
    """An element photo must not be a close-up: a close-up cannot show which element it is."""
    if probs is None:
        return SceneAnswer(None, None, True, "No scene check (model not installed).")
    best = max(probs, key=probs.get)
    if best == "close_up":
        return SceneAnswer(best, probs["close_up"], False, RETAKE_CLOSE)
    return SceneAnswer(best, probs["close_up"], True, f"Shows {NAMES[best]} ({probs[best]:.0%}).")


def choose_close_up_min(p_close_up: np.ndarray, is_close_up: np.ndarray, target: float = CLOSE_UP_RETAKE_TARGET,
                        candidates=tuple(round(0.05 * k, 2) for k in range(1, 20))) -> float:
    """The highest threshold at which at most `target` of the real close-ups are sent to retake. Validation only."""
    real = p_close_up[is_close_up]
    ok = [c for c in candidates if real.size and np.mean(real < c) <= target]
    return max(ok) if ok else min(candidates)


def check(model, photo: Image.Image, device: str = "cpu", kind: str = "crack") -> SceneAnswer:
    probs = None if model is None else probabilities(model, photo, device)
    return crack_photo_check(probs) if kind == "crack" else element_photo_check(probs)
