"""Exposed steel on the close-up (D107): SAM 3.1 asked for bars. Spec: spec/features/exposed_steel.md.
Shown as possible; when seen, the traced crack widths are not used (they may follow the steel) and the type is
spalling with exposed bar. Thresholds illustrative, chosen by eye on 100 real close-ups.
D112: exposed steel makes the case Category 3 at once (written rule, src/triage/matrix.py). D115: the same SAM call
asks for what may come before the crack: roots and pipes, shown only (water was tried and left out: it marked plain
wall). Their words and thresholds are untested on real photos (assumption)."""

from __future__ import annotations

import numpy as np
from PIL import Image

PROMPTS = ["rebar", "exposed rebar", "rusty steel bar"]  # "steel bar" also finds railings and gates: not asked
MIN_SCORE = 0.5
MIN_SHARE = 0.002  # of the photo
SIDE = 1024
LABEL = "Possible exposed steel (SAM 3.1): the concrete cover is gone here"
# D115: what can come before the crack, asked in the same call; shown only
# one word each: SAM's time grows per word. Water is left out (26 Nov 2026): on real photos it marked plain wall areas
CAUSES = {"roots": ["plant roots"], "pipe": ["pipe"]}
CAUSE_LABELS = {"roots": "Possible roots (SAM 3.1)",
                "pipe": "Possible pipe (SAM 3.1)"}
CAUSE_MIN_SCORE = 0.5  # illustrative, untested on real photos
CAUSE_MIN_SHARE = 0.002  # of the photo


def decide(sam_out: list) -> dict:
    """From SAM's answers [(prompt, mask, score)]: {seen, score, prompt, mask}. Only the three bar prompts count."""
    best = None
    for word, m, s in sam_out:
        if word not in PROMPTS or s < MIN_SCORE or m.mean() < MIN_SHARE:
            continue
        if best is None or s > best[2]:
            best = (word, m, float(s))
    if best is None:
        return {"seen": False, "score": None, "prompt": None, "mask": None}
    union = np.zeros_like(best[1], dtype=bool)
    for word, m, s in sam_out:
        if word in PROMPTS and s >= MIN_SCORE and m.mean() >= MIN_SHARE:
            union |= m
    return {"seen": True, "score": round(best[2], 2), "prompt": best[0], "mask": union}


def decide_causes(sam_out: list) -> list[dict]:
    """From SAM's answers [(prompt, mask, score)]: one entry per cause seen, {kind, score, mask}, best score first."""
    word_to = {w: k for k, ws in CAUSES.items() for w in ws}
    best: dict[str, tuple] = {}
    for word, m, s in sam_out:
        kind = word_to.get(word)
        if kind is None or s < CAUSE_MIN_SCORE or m.mean() < CAUSE_MIN_SHARE:
            continue
        if kind not in best:
            best[kind] = (float(s), m.copy())
        else:
            best[kind] = (max(best[kind][0], float(s)), best[kind][1] | m)
    return [{"kind": k, "score": round(v[0], 2), "mask": v[1]} for k, v in sorted(best.items(), key=lambda kv: -kv[1][0])]


def _outlines(mask: np.ndarray, kind: str, label: str, size: tuple[int, int]) -> list[dict]:
    import cv2
    sw, sh = size
    out = []
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:4]:
        if cv2.contourArea(c) < 20:
            continue
        c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
        out.append({"kind": kind, "label": label, "share": round(float(mask.mean()), 4),
                    "centre": [round(float(c[:, 0].mean()) / sw, 5), round(float(c[:, 1].mean()) / sh, 5)],
                    "outline": [[round(float(x) / sw, 5), round(float(y) / sh, 5)] for x, y in c]})
    return out


def find(photo: Image.Image, keep_mask: bool = False) -> dict | None:
    """Exposed steel on the photo, with its outline as shares of the photo for the crack viewer, and the possible
    causes before the crack (D115); None without SAM. keep_mask: also return the mask at the size SAM read (the element
    photo, D112)."""
    from element.parts import sam_segmenter, shared_predictor
    p = shared_predictor()
    if p is None:
        return None
    im = photo.convert("RGB")
    k = min(1.0, SIDE / max(im.size))
    small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR) if k < 1 else im
    sam_out = sam_segmenter(p, PROMPTS + [w for ws in CAUSES.values() for w in ws])(np.asarray(small))
    out = decide(sam_out)
    if out["seen"]:
        out["defects"] = _outlines(out["mask"], "steel", LABEL, small.size)
    if not keep_mask:
        out.pop("mask", None)
    out["causes"] = []
    for c in decide_causes(sam_out):
        out["causes"].append({"kind": c["kind"], "score": c["score"], "label": CAUSE_LABELS[c["kind"]],
                              "outlines": _outlines(c["mask"], c["kind"], CAUSE_LABELS[c["kind"]], small.size)})
    return out
