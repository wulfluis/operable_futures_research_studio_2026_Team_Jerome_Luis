"""A few short questions after the photo, one at a time, like the "20 questions" game (D120). Spec:
spec/features/questions.md.

Most ask where in the building the crack sits; a few ask about the warning signs the risk matrix already knows
(src/triage/risk_matrix_v3.json). A "yes" to a sign raises the level as that sign does; an answer never lowers it,
and "I don't know" is always there and counts as nothing. Every answer is kept with the result as training data.

The knock: the resident knocks on the wall next to the crack and records it. The app reads the sound's ringing and
pitch, but does not yet say hollow or solid: there are no knocks labelled by engineers to learn it from. Shown
only; the recording itself is not kept, only its numbers."""

from __future__ import annotations

import io
import wave

import numpy as np

DONT_KNOW = "dont_know"

# key, question, options (value, words), the matrix sign a "yes" sets (None: where the crack is, context only)
QUESTIONS = [
    {"key": "side", "text": "Where do you see the crack?",
     "options": [("inside", "Inside"), ("outside", "Outside"), ("both", "Both sides of the wall")], "sign": None},
    {"key": "floor", "text": "On which floor is it?",
     "options": [("ground", "Ground floor"), ("middle", "A middle floor"), ("top", "Top floor"),
                 ("below", "Below the ground")], "sign": None},
    {"key": "floors_above", "text": "How many floors are above it?",
     "options": [("0", "None"), ("1", "1"), ("2", "2"), ("3+", "3 or more")], "sign": None},
    {"key": "on", "text": "What is the crack on?",
     "options": [("wall", "A wall"), ("column", "A column"), ("beam", "A beam"), ("ceiling", "The ceiling")],
     "sign": None},
    {"key": "near_opening", "text": "Does it start at the corner of a window or door?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": None},
    {"key": "near_joint", "text": "Is it where a wall meets the floor, the ceiling or a column?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": None},
    {"key": "sticking", "text": "Do doors or windows near it stick, or no longer close?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": "sticking_openings"},
    {"key": "earthquake", "text": "Was there an earthquake since you first saw the crack?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": "recent_earthquake"},
    {"key": "added_floors", "text": "Were floors added on top of the building?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": "added_floors"},
    {"key": "water", "text": "Is the wall damp there, or does water drip near it?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": "damp"},
    {"key": "sagging", "text": "Does a beam, the floor or the ceiling above sag?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": "sagging"},
    {"key": "changed", "text": "Has it become longer or wider since you first saw it?",
     "options": [("yes", "Yes"), ("no", "No")], "sign": None},  # the matrix's growth sign needs two measured photos
]
BY_KEY = {q["key"]: q for q in QUESTIONS}


def signs(answers: dict) -> tuple[str, ...]:
    """The matrix signs the answers set: a "yes" to a sign question. Nothing else; never a lower level."""
    return tuple(BY_KEY[k]["sign"] for k, v in answers.items()
                 if k in BY_KEY and BY_KEY[k]["sign"] and v == "yes")


def next_question(answers: dict) -> dict | None:
    """The first question not answered yet (skipped counts as answered: "I don't know"), or None when done."""
    return next((q for q in QUESTIONS if q["key"] not in answers), None)


def words(answers: dict) -> list[tuple[str, str]]:
    """(question, answer in words) for every answer given, in the questions' order."""
    out = []
    for q in QUESTIONS:
        v = answers.get(q["key"])
        if v is None:
            continue
        out.append((q["text"], "I don't know" if v == DONT_KNOW else dict(q["options"]).get(v, v)))
    return out


# ---- the knock ----------------------------------------------------------------------------------------------------

def _samples(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(float)
    if width == 1:
        x -= 128
    x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x
    return x / (np.abs(x).max() or 1.0), rate


def knock(data: bytes) -> dict | None:
    """The loudest knock in a WAV recording: its pitch (strongest frequency, Hz), brightness (spectral centroid, Hz)
    and how long it rings (ms until 20 dB below its peak), and how many knocks were heard. None: no knock heard.
    No hollow or solid answer: not learned yet (no labelled knocks)."""
    try:
        x, rate = _samples(data)
    except (wave.Error, EOFError, KeyError, ValueError):
        return None
    if len(x) < rate // 20:
        return None
    win = max(1, rate // 1000)  # 1 ms envelope
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))
    peak = int(np.argmax(env))
    floor = float(np.median(env))
    if env[peak] < 4 * max(floor, 1e-4):  # nothing stands out from the background
        return None
    after = env[peak:]
    below = np.flatnonzero(after < env[peak] * 0.1)  # -20 dB
    ring_ms = (below[0] if len(below) else len(after)) / rate * 1000
    seg = x[peak: peak + rate // 10]  # 100 ms after the knock
    if len(seg) < 64:
        return None
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / rate)
    band = (f > 80) & (f < min(8000, rate / 2))
    pitch = float(f[band][np.argmax(spec[band])]) if band.any() else None
    centroid = float((f[band] * spec[band]).sum() / max(spec[band].sum(), 1e-12)) if band.any() else None
    loud = env > max(env[peak] * 0.3, 4 * floor)
    starts = np.flatnonzero(np.diff(loud.astype(int)) == 1)
    knocks = int(len(starts[np.diff(np.r_[-rate, starts]) > rate // 10])) or 1  # knocks at least 0.1 s apart
    return {"pitch_hz": None if pitch is None else round(pitch), "brightness_hz": None if centroid is None else
            round(centroid), "ring_ms": round(ring_ms, 1), "knocks": knocks, "reading": None,
            "note": "Hollow or solid is not read yet: the app needs knocks labelled by engineers to learn it."}


# ---- how sure the app is today: measured numbers only (README, spec/decisions.md) ---------------------------------

TODAY = [
    ("Finding the crack", "misses 3 of 76 real cases (crack model v5, D100)"),
    ("Recognising the element", "answers 67 % of 579 test photos, right on 86 % of those (README)"),
    ("Width with a bank card", "about 0.06 mm too wide on 14,198 hand-measured lab cracks (D108)"),
    ("Crack type and level", "not yet checked against engineers: 0 engineer-rated cases so far"),
    ("Hollow or solid from a knock", "not learned yet: 0 labelled knocks"),
]
