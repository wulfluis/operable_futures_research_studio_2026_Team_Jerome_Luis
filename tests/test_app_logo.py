import xml.etree.ElementTree as ET

from app.logo import LOGO_CSS, logo_oo, logo_title


def test_the_two_os_of_look_become_the_mark():
    out = logo_title()
    assert out.startswith('<span class="logo" role="img" aria-label="Second Look">Second L<svg')
    assert out.endswith("</svg>k</span>")  # "Second L" + mark + "k"; the page sets the capitals
    assert "<Look" not in logo_title("Second <Look>")  # escaped


def test_mark_is_valid_svg_with_two_frames_and_no_crack():
    root = ET.fromstring(logo_oo())
    groups = root.findall("{http://www.w3.org/2000/svg}g") or list(root)
    assert root.get("aria-hidden") == "true" and len(groups) == 2
    assert not any(p.get("class") == "oo-crack" for p in root.iter())  # no crack in the logo since D76


def test_the_logo_stands_still():
    assert "animation" not in LOGO_CSS and "@keyframes" not in LOGO_CSS  # GitHub's look: nothing moves (D76)
