"""The Second Look word mark, as on the cover of the midterm deck: the two O's of "Look" are scan frames. On load the
second O slides out of the first and a red crack draws through it (CSS only, plays once). Spec: spec/features/class_demo.md."""

import html

FRAME = "M5 18 L5 5 L18 5 M32 5 L45 5 L45 18 M45 32 L45 45 L32 45 M18 45 L5 45 L5 32"  # four corners of a scan frame
CRACK = "M24 6 L28 18 L21 26 L29 34 L24 44"
RED, GREY, WHITE = "#d62728", "#6f6f6f", "#ffffff"

# Sizes in em of the title, from the deck cover (176 px type, 264 x 136 px mark): the frames span the capital height.
LOGO_CSS = f"""
.logo-oo {{ display: inline-block; width: 1.5em; height: 0.773em; vertical-align: -0.065em; margin-right: -0.05em; }}
.logo-oo .oo-frame {{ fill: none; stroke-width: 10; }}
.logo-oo .oo-a {{ transform: translate(13px, 10px) scale(2.28); animation: oo-a 1s cubic-bezier(.7, 0, .2, 1) .8s both; }}
.logo-oo .oo-b {{ transform: translate(139px, 10px) scale(2.28); animation: oo-b 1s cubic-bezier(.7, 0, .2, 1) .8s both; }}
.logo-oo .oo-crack {{ fill: none; stroke: {RED}; stroke-width: 4; stroke-linejoin: round; stroke-dasharray: 48;
                      stroke-dashoffset: 48; animation: oo-crack .7s ease-out 1.9s forwards; }}
@keyframes oo-a {{ from {{ transform: translate(76px, 10px) scale(2.28); }} }}
@keyframes oo-b {{ from {{ transform: translate(76px, 10px) scale(2.28); }} }}
@keyframes oo-crack {{ to {{ stroke-dashoffset: 0; }} }}
"""


def logo_oo() -> str:
    """The two O's: the cracked white one drawn first, the grey one over it while they still overlap."""
    return (f'<svg class="logo-oo" viewBox="0 0 264 136" aria-hidden="true">'
            f'<g class="oo-b"><path class="oo-frame" stroke="{WHITE}" d="{FRAME}"/><path class="oo-crack" d="{CRACK}"/></g>'
            f'<g class="oo-a"><path class="oo-frame" stroke="{GREY}" d="{FRAME}"/></g></svg>')


def logo_title(name: str = "Second Look") -> str:
    """The name with its last "oo" replaced by the mark; screen readers get the plain name."""
    head, oo, tail = name.rpartition("oo")
    if not oo:
        return html.escape(name)
    return (f'<span class="logo" role="img" aria-label="{html.escape(name)}">'
            f'{html.escape(head)}{logo_oo()}{html.escape(tail)}</span>')
