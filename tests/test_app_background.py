import re
import xml.etree.ElementTree as ET

from app.background import (H, W, background_css, crack_lines, crack_svg, crack_underline_css,
                            crack_underline_svg, flash_times, model_line)


def test_crack_is_valid_svg_and_always_the_same():
    root = ET.fromstring(crack_svg(flash=False))
    assert root.tag.endswith("svg") and len(root) == len(crack_lines()) > 5
    assert crack_svg() == crack_svg()


def test_red_flash_runs_down_the_crack_and_into_each_branch_after_it():
    root = ET.fromstring(crack_svg())
    assert len(root) == 2 * len(crack_lines())  # one red flash over each white line
    assert all(a.get("attributeName") == "stroke-dashoffset" for a in root.iter("{http://www.w3.org/2000/svg}animate"))  # it runs, no flicker
    times = flash_times(crack_lines())
    assert times[0][0] == 0 and all(start > 0 and travel > 0 for start, travel in times[1:])
    for animate in root.iter("{http://www.w3.org/2000/svg}animate"):
        assert 0 < float(animate.get("keyTimes").split(";")[1]) < 1  # the flash ends before the cycle restarts


def test_main_crack_runs_from_top_to_bottom():
    main = crack_lines()[0][0]
    assert main[0][1] < 0 and main[-1][1] > H
    assert all(-W < x < 2 * W for x, _ in main)


def test_css_embeds_the_crack_on_a_dark_page():
    css = background_css()
    assert "data:image/svg+xml;base64," in css and "#0b0b0b" in css


def test_one_cracked_line_runs_under_the_buttons():
    root = ET.fromstring(crack_underline_svg())
    assert root.get("preserveAspectRatio") == "none" and len(root) == 1 + 1600 // 160 + 4  # line, hairlines, branches
    pts = [[tuple(map(float, p.split(","))) for p in path.get("d")[1:].split(" L")] for path in root]
    line, branches = pts[0], pts[-4:]
    assert line[0][0] < 10 and line[-1][0] > 1590  # one continuous line across the full width
    offsets = []
    for i, b in enumerate(branches):  # each rises toward its own button, inside that button's quarter
        assert b[-1][1] < 25 < 60 < b[0][1] and all(400 * i < x < 400 * (i + 1) for x, _ in b)
        offsets.append(b[0][0] - (400 * i + 200))
    assert all(abs(o) > 20 for o in offsets) and len({round(abs(o)) for o in offsets}) == 4  # off-centre, unequal
    css = crack_underline_css(".rule", 4, ".row:has(.b{n}:hover)")
    positions = re.findall(r"\.rule > div:nth-child\((\d)\) \{ background-position: ([\d.]+)% 0; \}", css)
    assert [(int(n), round(float(p), 2)) for n, p in positions] == [(1, 0), (2, 33.33), (3, 66.67), (4, 100)]
    assert ".row:has(.b3:hover) .rule > div:nth-child(3) { background-image:" in css  # that piece turns red


def test_model_line_stands_still_and_fills_the_bottom_from_edge_to_edge():
    out = model_line("U-Net <v3>")
    assert 'aria-label="U-Net &lt;v3&gt;">U-NET &lt;V3&gt;</div>' in out  # escaped, one line
    assert "bottom:" in out and "left:" in out and "right:" in out and "text-align-last: justify" in out
    assert "animation" not in out  # it does not move or flicker
    assert "background: linear-gradient(to bottom, #0b0b0b00, #0b0b0bf2" in out  # a band hides the crack under it


def test_crack_fades_in_from_the_dark_at_the_top():
    css = background_css()
    assert "linear-gradient(to bottom, #0b0b0b 0, #0b0b0b 2vh, #0b0b0b00 14vh), url(" in css  # fade over the crack
    assert "background-size: 100% 100%, cover;" in css
