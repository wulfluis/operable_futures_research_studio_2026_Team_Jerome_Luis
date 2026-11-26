"""What to do now, beside the level (D121). Spec: spec/features/advice.md.

Written, plain steps a resident can take at once, for the level and for what the photos and the answers showed.
General precautions, not an engineer's assessment: illustrative until reviewed by engineers. Never "safe", never a
clearance; nothing here changes a level (rule 5)."""

from __future__ import annotations

ALWAYS = [
    "Do not fill or paint over the crack before it has been checked: the engineer needs to see it as it is.",
    "Draw a short pencil line across each end of the crack and write the date next to it, so you can see if it grows.",
]
BY_LEVEL = {
    "Category 1": [
        "Photograph the same crack again in about 6 months, with the card, from the same distance.",
    ],
    "Category 2": [
        "Keep heavy things away from the cracked area and from the floor above it: a full fridge, water tanks, "
        "stacked goods, full bookshelves.",
        "Do not drill into or hang heavy things on the cracked wall, column or beam.",
        "Photograph the same crack again in about 1 month, with the card, from the same distance.",
    ],
    "Category 3": [
        "Move heavy things away from the area now: a full fridge, water tanks, stacked goods, full bookshelves.",
        "Do not sleep or sit right under or next to the cracked element until an engineer has seen it.",
        "Do not remove walls, add floors or start building work on this part of the house.",
    ],
    "Expert needed": [
        "Until the engineer has looked, keep heavy things away from the cracked area and from the floor above it.",
        "Do not drill into or hang heavy things on the cracked element.",
    ],
    "Retake": [],
}
BY_SIGN = {
    "steel": ["Do not knock off loose concrete around the steel yourself, and keep people from standing under it.",
              "Keep water away from the steel: it rusts faster when wet."],
    "water": ["Find and stop the water: a leaking pipe, the roof, a gutter or water standing against the wall."],
    "pipe": ["Check whether the pipe near the crack leaks."],
    "roots": ["Roots near the wall can move the ground: tell the engineer about the tree or plant."],
    "column": ["A column carries the floors above: do not lean or stack loads against it."],
    "sagging": ["Keep people out from under the sagging part until an engineer has seen it."],
    "sticking": ["Doors or windows that start to stick can mean the building is moving: tell the engineer."],
}
LEAVE = ("Leave the room and call for help at once if the crack grows quickly, you hear cracking sounds, pieces fall, "
         "or the floor or ceiling starts to sag.")
NOTE = "General precautions, not an engineer's assessment (illustrative until reviewed by engineers)."


def advice(level: str | None, *, steel: bool = False, causes: tuple[str, ...] = (), element: str | None = None,
           answers: dict | None = None) -> list[str]:
    """The steps for this result, most urgent first, without repeats. level None: no level yet."""
    a = answers or {}
    out: list[str] = []
    if steel:
        out += BY_SIGN["steel"]
    if level in ("Category 3", "Expert needed") or steel or a.get("sagging") == "yes":
        out.append(LEAVE)
    if a.get("sagging") == "yes":
        out += BY_SIGN["sagging"]
    out += BY_LEVEL.get(level or "", BY_LEVEL["Expert needed"] if level is None else [])
    if "water" in causes or a.get("water") == "yes":
        out += BY_SIGN["water"]
    if "pipe" in causes:
        out += BY_SIGN["pipe"]
    if "roots" in causes:
        out += BY_SIGN["roots"]
    if element == "column" or a.get("on") == "column":
        out += BY_SIGN["column"]
    if a.get("sticking") == "yes":
        out += BY_SIGN["sticking"]
    if level != "Retake":
        out += ALWAYS
    return list(dict.fromkeys(out))
