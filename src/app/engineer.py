"""The crack read the way a crack engineer reads it (D99): how it moved, how sure each width is, what a crack along a
bar means, and what this rests on. From E. Corres Sojo's thesis [46] and the method it uses [48]. Shown only: nothing
here changes a level (rule 5); the thesis gives no width limits, the levels stay in src/triage/.
Spec: spec/features/crack_kinematics.md ("Read like an engineer")."""

from __future__ import annotations

SOURCES = {
    "thesis": "E. Corres Sojo, Contributions to rebar-to-concrete interaction and its structural implications for "
              "design and monitoring applications, PhD thesis 10772, EPFL, 2024 [46]",
    "method": "B. G. Pantoja-Rosero et al., Determining crack kinematics from imaged crack patterns, Construction and "
              "Building Materials 343, 2022 [48]",
}


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "–"
    return (f"{v:.2f}" if unit == "mm" and v < 10 else f"{v:.1f}") + f" {unit}"


def notes(d: dict) -> list[dict]:
    """The engineer's reading of one result (its to_dict): a list of {title, text, source}, in reading order."""
    out = []
    ks = [k for k in (d.get("kinematics") or []) if k and k.get("opening_median") is not None]
    if ks:
        k = ks[0]
        u = k["unit"]
        slide = ("the sliding could not be read: the crack's faces are too smooth to match"
                 if k.get("sliding_median") is None else f"sliding {_fmt(k['sliding_median'], u)} along it")
        meaning = {"mostly opening": "The faces mostly moved apart.",
                   "opening and sliding": "The faces moved apart and also slid along each other: an engineer looks "
                                          "at this for shear.",
                   "mostly sliding": "The faces mostly slid along each other: an engineer looks at this for shear."
                   }.get(k.get("reading"), "")
        out.append({"title": "How the crack moved",
                    "text": f"Opening {_fmt(k['opening_median'], u)} across the crack, {slide} (median along the main "
                            f"crack). {meaning} Read from the crack's two faces, which were one surface: the bumps of "
                            "one match the hollows of the other. One photo shows the movement since the crack formed, "
                            "not whether it is still moving.",
                    "source": "method"})
    c = d.get("crack") or {}
    if d.get("measured") and c.get("width_uncertainty_mm") is not None:
        out.append({"title": "How sure the width is",
                    "text": f"Every width is ± {c['width_uncertainty_mm']:.2f} mm: one pixel either way, as photo "
                            "widths matched a crack microscope on a 50-year-old crack. Dust and worn crack edges make "
                            "it larger: wipe the crack before the photo.",
                    "source": "thesis"})
    elif d.get("statement") in ("Crack found", "Expert needed"):
        out.append({"title": "How sure the width is",
                    "text": "Without a bank card the width is in the photo's pixels, ± 1 pixel. A card next to the "
                            "crack turns it into millimetres.",
                    "source": "thesis"})
    pattern = ((d.get("triage") or {}).get("pattern") or (d.get("pattern_suggestion") or {}).get("pattern"))
    if pattern == "along_bar":
        out.append({"title": "A crack along a steel bar",
                    "text": "A crack running along a bar weakens the bar's grip on the concrete (bond); the wider the "
                            "crack compared with the bar, the weaker the grip. With rust, the bar is also losing "
                            "steel. A photo shows the bar only where it is exposed.",
                    "source": "thesis"})
    if d.get("growth"):
        g = d["growth"]
        out.append({"title": "Since the last visit", "text": g.get("reason", ""), "source": "thesis"})
    out.append({"title": "What this rests on",
                "text": "The movement, the width's error and the growth rule follow a crack engineer's published "
                        "work. The level does not: it comes from our written risk rules, illustrative until an "
                        "engineer calibrates them, and an engineer confirms every Category 3.",
                "source": "thesis"})
    return out


def html(d: dict) -> str:
    """The notes as the result window's panel body."""
    rows = "".join(f'<p class="d-line"><b>{n["title"]}.</b> {n["text"]}</p>' for n in notes(d))
    refs = "".join(f'<li>{SOURCES[k]}</li>' for k in ("thesis", "method"))
    return rows + f'<p class="d-note">Based on:</p><ul class="d-note">{refs}</ul><p class="d-note">Shown only: the ' \
                  'level does not use it.</p>'
