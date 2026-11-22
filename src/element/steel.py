"""Exposed steel on the close-up (D107): SAM 3.1 asked for bars. Spec: spec/features/exposed_steel.md.
Shown as possible; when seen, the traced crack widths are not used (they may follow the steel) and the type is
spalling with exposed bar. Thresholds illustrative, chosen by eye on 100 real close-ups."""

from __future__ import annotations

import numpy as np
from PIL import Image

PROMPTS = ["rebar", "exposed rebar", "rusty steel bar"]  # "steel bar" also finds railings and gates: not asked
MIN_SCORE = 0.5
MIN_SHARE = 0.002  # of the photo
SIDE = 1024
LABEL = "Possible exposed steel (SAM 3.1): the concrete cover is gone here"


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


def find(photo: Image.Image) -> dict | None:
    """Exposed steel on the photo, with its outline as shares of the photo for the crack viewer; None without SAM."""
    import cv2
    from element.parts import sam_segmenter, shared_predictor
    p = shared_predictor()
    if p is None:
        return None
    im = photo.convert("RGB")
    k = min(1.0, SIDE / max(im.size))
    small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR) if k < 1 else im
    out = decide(sam_segmenter(p, PROMPTS)(np.asarray(small)))
    if out["seen"]:
        sw, sh = small.size
        cs, _ = cv2.findContours(out["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out["defects"] = []
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:4]:
            if cv2.contourArea(c) < 20:
                continue
            c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
            out["defects"].append({"kind": "steel", "label": LABEL, "share": round(float(out["mask"].mean()), 4),
                                   "centre": [round(float(c[:, 0].mean()) / sw, 5), round(float(c[:, 1].mean()) / sh, 5)],
                                   "outline": [[round(float(x) / sw, 5), round(float(y) / sh, 5)] for x, y in c]})
    out.pop("mask", None)
    return out
