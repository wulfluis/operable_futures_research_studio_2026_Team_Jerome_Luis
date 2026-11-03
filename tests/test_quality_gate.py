import numpy as np
import pytest
from PIL import Image, ImageFilter

from quality.gate import MAX_BLUR, MIN_SHARPNESS, blur, check, sharpness, to_grey

rng = np.random.default_rng(0)
TEXTURE = rng.integers(60, 200, size=(128, 128)).astype(np.float64)  # synthetic, sharp, mid-grey


def test_sharp_mid_grey_with_scale_card_passes():
    result = check(TEXTURE, scale_visible=True)
    assert result.image_ok and result.failed == []


def test_blur_lowers_sharpness_and_fails():
    blurred = np.asarray(Image.fromarray(TEXTURE.astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), dtype=float)
    assert sharpness(blurred) < sharpness(TEXTURE)
    assert "blurred" in check(blurred, scale_visible=True).failed


def test_flat_image_has_zero_sharpness():
    assert sharpness(np.full((32, 32), 128.0)) == 0
    assert blur(np.full((32, 32), 128.0)) == 1.0  # no detail at all counts as fully blurred


def _plaster(grain: float, mottle: float, depth: float = 80) -> np.ndarray:
    """A synthetic wall: soft mottling plus fine grain, with one sharp hairline crack."""
    r = np.random.default_rng(0)
    m = np.asarray(Image.fromarray(r.integers(0, 255, (400, 400)).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                   dtype=float)
    wall = 170 + (m - m.mean()) * mottle / m.std() + r.normal(0, grain, (400, 400))
    wall[200, 40:360] -= depth
    return np.clip(wall, 0, 255).astype(np.uint8).astype(float)


def _soften(grey: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(grey.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), dtype=float)


@pytest.mark.parametrize("grain, mottle, depth, which", [(1.0, 0, 60, "Laplacian variance"), (1.0, 8, 80, "blur effect")],
                         ids=["smooth_wall", "mottled_wall"])
def test_sharp_wall_failed_by_one_measure_is_not_blurred(grain, mottle, depth, which):
    """Each measure alone sends some sharp walls to retake (D12); "blurred" needs both to agree (D33)."""
    wall = _plaster(grain, mottle, depth)
    alone = sharpness(wall) < MIN_SHARPNESS if which == "Laplacian variance" else blur(wall) > MAX_BLUR
    assert alone  # this measure alone would call the sharp wall blurred
    assert "blurred" not in check(wall, scale_visible=True).failed


def test_mottled_wall_out_of_focus_is_blurred():
    assert "blurred" in check(_soften(_plaster(1.0, 8)), scale_visible=True).failed


@pytest.mark.parametrize("level, reason", [(10, "too dark"), (250, "too bright")])
def test_exposure(level, reason):
    img = np.clip(TEXTURE - TEXTURE.mean() + level, 0, 255)
    assert reason in check(img, scale_visible=True).failed


@pytest.mark.parametrize("scale", [None, False])
def test_unconfirmed_scale_card_fails(scale):
    result = check(TEXTURE, scale_visible=scale)
    assert not result.image_ok and "scale card not confirmed" in result.failed


def test_large_sharp_photo_is_not_called_blurred():
    """A sharp scene photographed at phone resolution must not fail just for being large (decision D8)."""
    small = Image.fromarray(TEXTURE.astype(np.uint8))
    phone = np.asarray(small.resize((3000, 3000), Image.Resampling.BICUBIC), dtype=np.float64)
    assert sharpness(phone) < 100  # raw measure collapses at high resolution
    assert "blurred" not in check(phone, scale_visible=True).failed


def test_small_images_are_measured_as_they_are():
    assert check(TEXTURE, scale_visible=True).sharpness == sharpness(TEXTURE)


def test_to_grey_reads_a_file(tmp_path):
    path = tmp_path / "x.png"
    Image.fromarray(TEXTURE.astype(np.uint8)).convert("RGB").save(path)
    assert to_grey(path).shape == (128, 128)
