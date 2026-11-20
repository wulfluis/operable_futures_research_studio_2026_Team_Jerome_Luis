"""The Second Look word mark, as on the cover of the midterm deck: the two O's of "Look" are scan frames, side by side
(D38). Since 12 Nov (D76): no crack in it and nothing moves. Since 18 Nov (D82): one O in white, one in the scan
cyan, on the dark concrete page.
Spec: spec/features/class_demo.md."""

import html

FRAME = "M5 18 L5 5 L18 5 M32 5 L45 5 L45 18 M45 32 L45 45 L32 45 M18 45 L5 45 L5 32"  # four corners of a scan frame
GREY, INK = "#22d3ee", "#eef1f5"  # GREY is the second frame: the scan cyan since D82

# Sizes in em of the title, from the deck cover (176 px type, 264 x 136 px mark): the frames span the capital height.
LOGO_CSS = f"""
.logo-oo {{ display: inline-block; width: 1.5em; height: 0.773em; vertical-align: -0.065em; margin-right: -0.05em; }}
.logo-oo .oo-frame {{ fill: none; stroke-width: 10; }}  /* the letters' own weight: one font across the name */
.logo-oo .oo-a {{ transform: translate(13px, 10px) scale(2.28); }}
.logo-oo .oo-b {{ transform: translate(139px, 10px) scale(2.28); }}
"""


def logo_oo() -> str:
    """The two O's: one in white, one in cyan, side by side."""
    return (f'<svg class="logo-oo" viewBox="0 0 264 136" aria-hidden="true">'
            f'<g class="oo-b"><path class="oo-frame" stroke="{INK}" d="{FRAME}"/></g>'
            f'<g class="oo-a"><path class="oo-frame" stroke="{GREY}" d="{FRAME}"/></g></svg>')


def logo_title(name: str = "Second Look") -> str:
    """The name with its last "oo" replaced by the mark; screen readers get the plain name."""
    head, oo, tail = name.rpartition("oo")
    if not oo:
        return html.escape(name)
    return (f'<span class="logo" role="img" aria-label="{html.escape(name)}">'
            f'{html.escape(head)}{logo_oo()}{html.escape(tail)}</span>')
