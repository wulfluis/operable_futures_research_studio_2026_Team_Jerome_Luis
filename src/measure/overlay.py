"""Overlay figure for one measurement: the photo with the reference and search box, and the crack close up."""

import numpy as np


def draw_overlay(rgb, reference_px, roi, result, title, out, show: bool = False, extra_lines=()):
    """Write the figure to `out` (a path or a binary file object, PNG)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.imshow(rgb)
    if reference_px is not None:
        ref = np.asarray(reference_px, dtype=float)
        if len(ref) == 4:
            ref = np.vstack([ref, ref[:1]])
        a1.plot(ref[:, 0], ref[:, 1], "-o", color="#1baf7a", lw=2, ms=4, label="reference")
    x0, y0, x1, y1 = roi
    a1.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", color="#3987e5", lw=1.5, label="search box")
    a1.legend(loc="lower left", fontsize=9)
    a1.set_title("Photo, reference and search box", loc="left", fontsize=11, weight="bold")
    a1.set_axis_off()

    if result is not None and result.found and result.mask is not None:
        ys, xs = np.nonzero(result.mask)
        m = 40
        cy0, cy1 = max(0, ys.min() - m), min(rgb.shape[0], ys.max() + m)
        cx0, cx1 = max(0, xs.min() - m), min(rgb.shape[1], xs.max() + m)
        ext = (cx0 - 0.5, cx1 - 0.5, cy1 - 0.5, cy0 - 0.5)
        a2.imshow(rgb[cy0:cy1, cx0:cx1], extent=ext)
        layer = np.zeros((cy1 - cy0, cx1 - cx0, 4))
        layer[result.mask[cy0:cy1, cx0:cx1]] = (0.92, 0.41, 0.2, 0.45)
        a2.imshow(layer, extent=ext)
        if result.main_path_px is not None:
            a2.plot(result.main_path_px[:, 0], result.main_path_px[:, 1], color="#184f95", lw=1.2, label="length")
        if result.widest_edges_px is not None:
            e = result.widest_edges_px
            a2.plot(e[:, 0], e[:, 1], color="#d62728", lw=2.5, label="widest point")
            a2.annotate(f"{result.width_max_mm:.2f} mm", e[1], xytext=(12, 12), textcoords="offset points",
                        color="#d62728", fontsize=11, weight="bold",
                        bbox={"boxstyle": "round", "fc": "white", "ec": "#d62728", "alpha": 0.9})
        a2.legend(loc="lower left", fontsize=9)
    else:
        a2.text(0.5, 0.5, "No crack measured", ha="center", va="center", fontsize=13, transform=a2.transAxes)
    a2.set_title("Crack pixels (orange), centre line and widest point", loc="left", fontsize=11, weight="bold")
    a2.set_axis_off()

    lines = [title]
    if result is not None:
        d = result.to_dict()
        if d["width_max_mm"] is not None:
            lines.append(f"Width (widest): {d['width_max_mm']:.2f} mm +- {d['width_uncertainty_mm']:.2f} mm   "
                         f"median {d['width_median_mm']:.2f} mm   {d['px_across_at_widest']:.1f} px across   "
                         f"pixel {d['mm_per_px']:.3f} mm")
        if d["length_main_mm"] is not None:
            lines.append(f"Length: {d['length_main_mm']:.1f} mm along the main path, "
                         f"{d['length_total_mm']:.1f} mm with all branches")
    lines += list(extra_lines)
    lines.append("A measurement, not a safety assessment. Thresholds illustrative until calibrated.")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=9.5, va="bottom", family="monospace")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.06 + 0.025 * len(lines), wspace=0.04)
    fig.savefig(out, dpi=150, format="png")
    if show:
        plt.show()
    plt.close(fig)
