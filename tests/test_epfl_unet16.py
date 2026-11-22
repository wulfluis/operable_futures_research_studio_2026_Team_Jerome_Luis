"""D104: the EPFL crack model's layout loads its weights by name, and the adapter reads distance < 2 px as crack."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from segmentation.epfl_unet16 import CRACK_DISTANCE, Adapter, UNet16, is_epfl  # noqa: E402


def test_the_layout_has_the_weight_names_of_the_published_model():
    keys = UNet16().state_dict().keys()
    for k in ("encoder.0.weight", "encoder.28.weight", "conv5.4.weight", "center.block.1.conv.weight",
              "dec5.block.2.conv.weight", "dec1.conv.weight", "final.weight"):
        assert k in keys
    assert UNet16().state_dict()["dec2.block.1.conv.weight"].shape == (128, 192, 3, 3)
    assert is_epfl(UNet16().state_dict()) and not is_epfl({"model": {}})


def test_the_output_keeps_the_photo_size():
    with torch.no_grad():
        assert UNet16()(torch.zeros(1, 3, 64, 96)).shape == (1, 1, 64, 96)


def test_the_adapter_calls_crack_only_under_two_pixels_of_distance():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[[[0.5, 1.9, 2.1, 20.0]]]])
    p = torch.sigmoid(Adapter(Fixed())(None))[0, 0, 0]
    assert CRACK_DISTANCE == 2.0 and (p > 0.5).tolist() == [True, True, False, False]
