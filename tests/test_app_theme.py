"""The app's look (D82): the level colours stay fixed, are never green, and Category 1 does not borrow the accent."""
import colorsys

from app.theme import THEME_CSS, TOKENS, TONES, label, state


def _hue(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def test_no_level_is_green():  # green reads as "safe" (rule 1)
    for level, kind in TONES.items():
        colour = TOKENS[kind]
        if colour.startswith("#"):
            assert not 75 <= _hue(colour) <= 165, (level, colour)


def test_category_1_has_its_own_colour():
    assert TOKENS[TONES["Category 1"]] != TOKENS["accent"]


def test_pills_escape_text():
    assert "<b>" not in label("<b>x</b>") and "<b>" not in state("<b>x</b>")


def test_motion_stops_for_reduced_motion():
    assert "prefers-reduced-motion" in THEME_CSS
