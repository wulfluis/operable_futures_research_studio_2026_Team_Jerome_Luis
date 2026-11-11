"""Which part of the building the crack lies on, from SAM 3.1's masks of the element photo (D62).

SAM 3.1 (reading list [81][82]) is asked for each part by name and returns a mask per thing found. The crack's line,
already carried into the element photo by the two photos' link (D58, D59), is laid on the masks: the part that holds
most of it is the answer. This is a suggestion shown in the app; it does not change the level, the element the level
uses, the position or the angle (rule 5: the level comes from the written rules).

Without the weights (`models/sam3.1/sam3.1_multiplex.pt`, gated, SAM License, never in git) the app runs as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "sam3.1" / "sam3.1_multiplex.pt"
IMGSZ = 644  # SAM 3 itself uses 1008; 1008 ran out of memory on the M4 with 16 GB (MPS), 11 Nov 2026

# The parts and the words SAM 3.1 is asked with: Luis's list of 11 Nov 2026 (the parts that matter for cracks).
# Column, wall, beam and slab are risk-matrix elements (risk_matrix_v3.json); balcony, stairs, window and expansion
# joint are not (yet), so they are context only.
PARTS = {
    "column": ["column", "concrete column"],
    "wall": ["concrete wall"],
    "beam": ["beam", "concrete beam"],
    "slab": ["ceiling slab", "concrete ceiling"],
    "balcony": ["balcony"],
    "stairs": ["staircase", "stairs"],
    "window": ["window"],
    "expansion joint": ["expansion joint"],
}
PART_NAMES = {"column": "Column", "wall": "Wall", "beam": "Beam", "slab": "Slab", "balcony": "Balcony",
              "stairs": "Stairs", "window": "Window", "expansion joint": "Expansion joint"}
COLOURS = {"column": (230, 25, 75), "wall": (60, 180, 75), "beam": (255, 225, 25), "slab": (0, 130, 200),
           "balcony": (245, 130, 48), "stairs": (145, 30, 180), "window": (70, 240, 240),
           "expansion joint": (240, 50, 230)}
WALL_NOTE = "load-bearing or infill: not known from the photo"

FIND_CONF = 0.25  # SAM's own score down to which things are found
MIN_CONF = 0.5  # shown and used only from here (illustrative): a mirror was read as a window at 0.41 (11 Nov)
SAME_THING_IOU = 0.6  # two masks of one part overlapping this much are one thing (illustrative)
ON_SHARE = 0.5  # the crack lies on a part when at least this share of its line is inside the part (illustrative)
BORDER_SHARE = 0.25  # two parts each holding at least this share: the crack runs along their border (illustrative)
LINE_WIDTH_SHARE = 0.004  # the crack line is drawn this thick (share of the photo's long side), at least 3 px

Segmenter = Callable[[np.ndarray], list]  # RGB array -> [(word, mask bool HxW, confidence)]


@dataclass
class Parts:
    on: str | None  # the part holding most of the crack, or None
    reason: str
    confidence: float | None = None  # SAM's score of that part
    share_of_crack: float | None = None
    border_with: str | None = None  # a second part when the crack runs along the border of two
    found: dict = field(default_factory=dict)  # part -> number of things found (score >= MIN_CONF)
    seconds: float | None = None
    preview: Image.Image | None = field(default=None, repr=False)  # the element photo with the masks; never shared

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 3)  # noqa: E731
        return {"on": self.on, "name": PART_NAMES.get(self.on) if self.on else None, "confidence": r(self.confidence),
                "share_of_crack": r(self.share_of_crack), "border_with": self.border_with,
                "border_name": PART_NAMES.get(self.border_with) if self.border_with else None,
                "found": self.found, "reason": self.reason,
                "note": WALL_NOTE if "wall" in (self.on, self.border_with) else None,
                "seconds": None if self.seconds is None else round(self.seconds, 1),
                "source": "SAM 3.1 masks of the element photo (D62), a suggestion; the level does not use it"}


def as_parts(found: list) -> list[tuple[str, np.ndarray, float]]:
    """Each word's finds renamed to its part, best first; of two masks of the same thing, one is kept."""
    word_to_part = {w: p for p, words in PARTS.items() for w in words}
    kept: list[tuple[str, np.ndarray, float]] = []
    for word, mask, conf in sorted(found, key=lambda f: -f[2]):
        part = word_to_part.get(word, word)
        same = any(p == part and (m & mask).sum() > SAME_THING_IOU * (m | mask).sum() for p, m, _ in kept)
        if not same:
            kept.append((part, mask, float(conf)))
    return kept


def crack_mask(crack: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The crack's line (n, 2 x, y) drawn into an empty mask of the photo's size."""
    m = np.zeros(shape, np.uint8)
    width = max(3, round(LINE_WIDTH_SHARE * max(shape)))
    cv2.polylines(m, [np.round(crack).astype(np.int32).reshape(-1, 1, 2)], False, 1, width)
    return m.astype(bool)


def lies_on(things: list[tuple[str, np.ndarray, float]], crack: np.ndarray | None, shape: tuple[int, int],
            seconds: float | None = None) -> Parts:
    """The part the crack lies on. things: (part, mask, score) with score >= MIN_CONF; crack: its line in the
    element photo's pixels, or None when the two photos were not linked."""
    found: dict = {}
    for p, _, _ in things:
        found[p] = found.get(p, 0) + 1
    listed = ", ".join(f"{PART_NAMES[p].lower()} ({n})" for p, n in found.items()) or "no part of the list"
    if crack is None or len(crack) < 2:
        return Parts(None, f"The crack could not be placed in the element photo (the two photos were not linked), so "
                           f"it is not known which part it lies on. Found in the photo: {listed}.", found=found,
                     seconds=seconds)
    line = crack_mask(crack, shape)
    n = max(1, int(line.sum()))
    union: dict[str, np.ndarray] = {}
    score: dict[str, float] = {}
    for p, m, c in things:  # a part found several times (two walls): all its things together
        union[p] = m if p not in union else union[p] | m
        score[p] = max(c, score.get(p, 0.0))
    share = {p: float((line & m).sum()) / n for p, m in union.items()}
    ranked = sorted(share, key=lambda p: -share[p])
    if not ranked or share[ranked[0]] < BORDER_SHARE:
        return Parts(None, f"The crack lies on none of the parts SAM 3.1 found (found: {listed}).", found=found,
                     seconds=seconds)
    first = ranked[0]
    second = next((p for p in ranked[1:] if share[p] >= BORDER_SHARE), None)
    if second is not None:
        return Parts(first, f"The crack runs along the border of the {PART_NAMES[first].lower()} "
                            f"({share[first]:.0%} of its line) and the {PART_NAMES[second].lower()} "
                            f"({share[second]:.0%}).", score[first], share[first], second, found, seconds)
    if share[first] < ON_SHARE:
        return Parts(None, f"Only {share[first]:.0%} of the crack's line lies on the {PART_NAMES[first].lower()}: "
                           "not clear which part it lies on.", found=found, seconds=seconds)
    return Parts(first, f"{share[first]:.0%} of the crack's line lies on the {PART_NAMES[first].lower()} "
                        f"(SAM 3.1, score {score[first]:.2f}).", score[first], share[first], None, found, seconds)


def find_parts(rgb: np.ndarray, crack: np.ndarray | None, segment: Segmenter) -> Parts:
    """SAM 3.1 on the element photo (rgb, the size the crack's line is in) and the part the crack lies on."""
    t = time.perf_counter()
    things = [f for f in as_parts(segment(rgb)) if f[2] >= MIN_CONF]
    out = lies_on(things, crack, rgb.shape[:2], time.perf_counter() - t)
    out.preview = preview(rgb, things, crack, out.on)
    return out


def preview(rgb: np.ndarray, things: list, crack: np.ndarray | None, on: str | None) -> Image.Image:
    """The element photo, small, with each part's mask in its colour, the crack in red with a ring."""
    base = np.asarray(rgb[..., :3], np.float32)
    out = base.copy()
    for p, m, _ in sorted(things, key=lambda f: -f[1].sum()):  # big first, so small parts stay visible on top
        colour = np.array(COLOURS.get(p, (200, 200, 200)), np.float32)
        a = 0.55 if p == on else 0.35
        out[m] = (1 - a) * base[m] + a * colour
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    k = min(1.0, 900 / max(img.size))
    img = img.resize((round(img.width * k), round(img.height * k)))
    d = ImageDraw.Draw(img)
    lw = max(2, img.width // 250)
    if crack is not None and len(crack) >= 2:
        c = crack * k
        d.line([tuple(p) for p in c], fill=(230, 30, 30), width=lw + 1)
        mid = c.mean(axis=0)
        r = max(12.0, 0.6 * float(np.linalg.norm(c.max(axis=0) - c.min(axis=0))))
        d.ellipse([mid[0] - r, mid[1] - r, mid[0] + r, mid[1] + r], outline=(230, 30, 30), width=lw)
    return img


def _clip_tokenizer_only() -> None:
    """SAM 3 in Ultralytics needs only `clip.simple_tokenizer` (PyPI `openai-clip` 1.0.1). The package's own
    __init__ imports `pkg_resources`, which recent setuptools no longer ships, so load the tokenizer module alone,
    and give it the call that Ultralytics' own CLIP fork has (as `clip.tokenize`)."""
    import importlib.util
    import sys
    import types

    if "clip" in sys.modules:
        return
    spec = importlib.util.find_spec("clip")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Install the tokenizer: python3 -m pip install --no-deps openai-clip==1.0.1 ftfy wcwidth")
    pkg = types.ModuleType("clip")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["clip"] = pkg
    import clip.simple_tokenizer as st
    import torch

    def tokenize(self, texts, context_length: int = 77):
        texts = [texts] if isinstance(texts, str) else list(texts)
        sot, eot = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        out = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [sot] + self.encode(t) + [eot]
            if len(ids) > context_length:
                ids = ids[:context_length - 1] + [eot]
            out[i, :len(ids)] = torch.tensor(ids)
        return out

    st.SimpleTokenizer.__call__ = tokenize


def make_predictor(device: str, conf: float = FIND_CONF):
    """Ultralytics' SAM 3 text predictor with the SAM 3.1 weights."""
    _clip_tokenizer_only()
    from ultralytics.models.sam import SAM3SemanticPredictor
    return SAM3SemanticPredictor(overrides=dict(conf=conf, task="segment", mode="predict", model=str(WEIGHTS),
                                                half=False, imgsz=IMGSZ, device=device, save=False, verbose=False))


def sam_segmenter(predictor, words: list[str] | None = None) -> Segmenter:
    """A segmenter from the predictor: every word of PARTS (or the given words) in one call."""
    words = words or [w for ws in PARTS.values() for w in ws]

    def segment(rgb: np.ndarray) -> list:
        h, w = rgb.shape[:2]
        predictor.set_image(np.ascontiguousarray(rgb[..., :3][..., ::-1]))  # Ultralytics expects BGR
        r = predictor(text=words)[0]
        if r.masks is None or not len(r.masks):
            return []
        out = []
        for m, c, p in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            if m.shape != (h, w):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) > 127
            out.append((words[c] if c < len(words) else str(c), m, float(p)))
        return out
    return segment


def load_parts_model(device: str | None = None) -> Segmenter | None:
    """The SAM 3.1 segmenter, or None when the weights or the packages are missing (the app then runs as before).
    The laptop's graphics chip (mps) when there is one: on the processor alone it is several times slower."""
    if not WEIGHTS.exists():
        return None
    try:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return sam_segmenter(make_predictor(device))
    except ImportError:
        return None
