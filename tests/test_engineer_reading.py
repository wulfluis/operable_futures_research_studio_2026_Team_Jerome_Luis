"""D99: the crack read like an engineer: growth beyond the error only, the notes panel, the viewer's movement data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import engineer  # noqa: E402
from app.crack_view import kin_data  # noqa: E402
from measure.growth import compare  # noqa: E402
from measure.kinematics import from_path  # noqa: E402
from test_kinematics import crack  # noqa: E402


def _r(w, u=0.1, measured=True):
    return {"measured": measured, "crack": {"width_max_mm": w, "width_uncertainty_mm": u, "length_main_mm": 300}}


def test_growth_counts_only_beyond_the_error():
    assert compare(_r(1.0), _r(1.3))["grown"] is True  # +0.3 > 0.14
    small = compare(_r(1.0), _r(1.1))  # +0.1 < 0.14
    assert small["grown"] is False and "No change beyond" in small["reason"]


def test_growth_needs_a_card_both_times():
    g = compare(_r(1.0, measured=False), _r(1.5))
    assert g["grown"] is None and "bank card" in g["reason"]


def test_growth_without_the_earlier_result():
    assert compare(None, _r(1.0), 7)["grown"] is None


def test_a_narrower_crack_is_not_called_closed():
    assert "Retake both" in compare(_r(1.5), _r(1.0))["reason"]


def test_the_engineer_panel_reads_the_movement_and_names_its_sources():
    d = {"statement": "Crack found", "measured": True, "crack": {"width_uncertainty_mm": 0.08},
         "kinematics": [{"unit": "mm", "opening_median": 0.4, "sliding_median": 0.3, "reading": "opening and sliding"}],
         "triage": {"pattern": "along_bar"}}
    titles = [n["title"] for n in engineer.notes(d)]
    assert titles[:3] == ["How the crack moved", "How sure the width is", "A crack along a steel bar"]
    html = engineer.html(d)
    assert "shear" in html and "Corres Sojo" in html and "Pantoja-Rosero" in html and "level does not use it" in html


def test_the_panel_never_says_safe():
    d = {"statement": "Crack found", "kinematics": [{"unit": "px", "opening_median": 5, "sliding_median": None,
                                                     "reading": "opening only (sliding not readable)"}]}
    assert "safe" not in engineer.html(d).lower()


def test_the_faces_from_measured_edges_give_the_same_movement():
    mask, path = crack(6, 3)
    xs = path[:, 0].astype(int)
    cols = [np.nonzero(mask[:, x])[0] for x in xs]
    edges = np.array([[[x, c.min() - 0.5], [x, c.max() + 0.5]] for x, c in zip(xs, cols) if len(c)], float)
    k = from_path(None, path, 6.0, edges_px=edges.reshape(-1, 2))
    assert k.opening_median == pytest.approx(6, abs=1.2) and k.sliding_median == pytest.approx(3, abs=1.2)


def test_the_viewer_gets_arrows_in_crack_order():
    mask, path = crack(6, 3)
    k = from_path(mask, path)
    data = kin_data([None, k], 600, 200)
    assert data[0] is None and data[1]["st"] and len(data[1]["st"][0]) == 6
    assert kin_data(None, 600, 200) is None
