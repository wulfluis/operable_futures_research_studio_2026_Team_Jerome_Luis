"""What looks different from sound concrete (D89): spec/features/anomaly_ae.md. Drawn images only."""
import io
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from anomaly import model as ae
from anomaly import predict, regions
from app.crack_view import view_data


def _wall(h=300, w=400):
    rng = np.random.default_rng(0)
    return np.clip(180 + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)


def _box(shape, y0, y1, x0, x1):
    m = np.zeros(shape[:2], bool)
    m[y0:y1, x0:x1] = True
    return m


def test_an_orange_brown_patch_is_possible_rust():
    rgb = _wall()
    rgb[100:150, 100:160] = (170, 95, 40)
    found = regions.find(rgb, _box(rgb.shape, 100, 150, 100, 160))
    assert len(found) == 1 and found[0]["kind"] == "rust" and found[0]["label"] == "Possible rust stain"


def test_a_darker_grey_patch_is_a_dark_patch():
    rgb = _wall()
    rgb[100:150, 100:160] = (100, 100, 100)
    assert regions.find(rgb, _box(rgb.shape, 100, 150, 100, 160))[0]["kind"] == "dark"


def test_anything_else_is_another_unusual_area():
    rgb = _wall()
    rgb[100:150, 100:160] = (200, 200, 205)
    assert regions.find(rgb, _box(rgb.shape, 100, 150, 100, 160))[0]["kind"] == "other"


def test_crack_areas_the_card_and_specks_are_left_out():
    rgb = _wall()
    unusual = _box(rgb.shape, 50, 250, 50, 53) | _box(rgb.shape, 20, 60, 300, 380) | _box(rgb.shape, 200, 203, 200, 203)
    crack = _box(rgb.shape, 50, 250, 51, 52)  # the traced crack explains the long thin line
    card = _box(rgb.shape, 10, 70, 290, 390)  # the bank card
    assert regions.find(rgb, unusual, crack, card) == []


def test_the_heatmap_covers_the_whole_photo():
    torch.manual_seed(0)
    m = ae.AE().eval()
    heat, tiles = ae.heatmap(m, Image.new("RGB", (500, 300), (180, 180, 180)), "cpu")
    w, h = ae.work_image(Image.new("RGB", (500, 300))).size
    assert heat.shape == (h, w) and np.isfinite(heat).all() and (heat > 0).all() and len(tiles) > 1


def test_the_heat_picture_is_transparent_where_nothing_is_unusual():
    heat = np.zeros((40, 40), np.float32)
    heat[10:20, 10:20] = 1.0
    im = Image.open(io.BytesIO(predict.heat_png(heat, 0.1)))
    a = np.asarray(im)[..., 3]
    assert a[0, 0] == 0 and a[15, 15] > 0


def test_no_model_means_no_layers(monkeypatch):
    monkeypatch.setattr(ae, "load", lambda device=None: None)
    assert predict.analyse_photo(Image.new("RGB", (100, 100))) is None


def test_the_viewer_gets_the_defects_and_the_heatmap():
    n = 30
    prof = np.column_stack([np.full(n, 500.0), np.linspace(100, 900, n)])
    crack = SimpleNamespace(found=True, profile_px=prof, profile_mm=np.full(n, 0.4), profile_edges_px=None,
                            main_path_px=prof.copy(), px_across=4.0, mm_per_px=0.1, width_uncertainty_mm=0.05,
                            length_main_mm=80.0)
    an = {"score": 1.3, "heat_png": b"\x89PNG", "defects": [
        {"kind": "rust", "label": "Possible rust stain", "centre": [0.2, 0.3], "outline": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]}]}
    r = SimpleNamespace(crack=crack, measured=True, pixel_crack=None, card=None, photo=Image.new("RGB", (1000, 1000)),
                        anomaly=an)
    d = view_data(r)
    assert d["surface_score"] is None  # computed, but not shown until calibrated at the app's scale (D89)
    assert d["heat"].startswith("data:image/png;base64,")
    assert d["defects"][0]["kind"] == "rust" and d["defects"][0]["c"] == [0.2, 0.3]
