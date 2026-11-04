import numpy as np
import pytest
import torch
from PIL import Image

from segmentation.predict import draw_overlay, run, working_image


class FakeModel(torch.nn.Module):
    """Stands in for the U-Net: 'sees' a crack wherever the photo is dark."""

    def forward(self, x):
        return (x.mean(1, keepdim=True) < -1.0).float() * 20 - 10


def _wall(crack=True, size=(1200, 900), blur=False):
    rng = np.random.default_rng(0)
    a = rng.integers(110, 190, (size[1], size[0], 3)).astype(np.uint8)
    if crack:
        a[size[1] // 2 - 2:size[1] // 2 + 2, 100:1100] = 10
    if blur:
        a[:] = 150
    return Image.fromarray(a)


def test_working_image_is_padded_to_32_and_keeps_the_shape():
    img, (w, h) = working_image(_wall(size=(1200, 900)))
    assert (w, h) == (832, 624) and img.size[0] % 32 == 0 and img.size[1] % 32 == 0


def test_dark_line_is_a_crack_with_a_red_overlay():
    r = run(_wall(), FakeModel())
    assert r.statement == "Crack found" and r.confidence > 0.9
    assert "scale card" in r.reason
    assert r.overlay.size[0] == 832 and r.overlay.size[1] > 624


def test_plain_wall_is_no_crack():
    assert run(_wall(crack=False), FakeModel()).statement == "No crack found"


def test_flat_photo_is_retake_before_the_model():
    r = run(_wall(blur=True), FakeModel())
    assert r.statement == "Retake" and "blurred" in r.reason and r.confidence is None


def test_overlay_draws_crack_pixels_red():
    img = Image.new("RGB", (100, 100), "white")
    mask = np.zeros((100, 100), bool); mask[50, :] = True
    out = np.array(draw_overlay(img, mask, "Crack found", "x"))
    assert out[50, 10, 0] > 200 and out[50, 10, 1] < 120 and out[10, 10].tolist() == [255, 255, 255]
