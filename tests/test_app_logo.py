import xml.etree.ElementTree as ET

from app.logo import LOGO_CSS, logo_oo, logo_title


def test_the_two_os_of_look_become_the_mark():
    out = logo_title()
    assert out.startswith('<span class="logo" role="img" aria-label="Second Look">Second L<svg')
    assert out.endswith("</svg>k</span>")  # "Second L" + mark + "k"; the page sets the capitals
    assert "<Look" not in logo_title("Second <Look>")  # escaped


def test_mark_is_valid_svg_with_two_frames_and_one_red_crack():
    root = ET.fromstring(logo_oo())
    groups = root.findall("{http://www.w3.org/2000/svg}g") or list(root)
    assert root.get("aria-hidden") == "true" and len(groups) == 2
    assert sum(p.get("class") == "oo-crack" for p in root.iter() if p.tag.endswith("path")) == 1


def test_animation_plays_once_like_the_deck_cover():
    assert "infinite" not in LOGO_CSS  # slides apart, the crack draws, then it stays
    assert "@keyframes oo-crack" in LOGO_CSS and "animation: oo-a 1s" in LOGO_CSS
