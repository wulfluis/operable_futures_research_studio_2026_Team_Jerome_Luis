"""The EPFL crack model of Pantoja-Rosero et al. 2022 [108] ("TOPO-Loss"), as a candidate crack finder (D104).

Their trained weights (Zenodo 6769028, CC BY 4.0, folder models/mse+topo: their best) belong to a U-Net with a VGG16
encoder in the layout of TernausNet's UNet16 (Iglovikov and Shvets 2018, MIT licence). This file is our own code for
that layout, so the weights load by name; the authors' code is GPL-3.0 and is not copied.

The model does not give a crack probability. Per pixel it gives the distance to the nearest crack centre line, in
pixels, capped at 20 (it was trained on such distance maps); the authors call a pixel crack when that distance is
under 2. To fit our pipeline (sigmoid, then 0.5 and 0.25 thresholds), `Adapter` turns the distance d into a logit
SLOPE * (CRACK_DISTANCE - d): above 0.5 exactly when d < 2, as the authors read it. Assumption: the slope only sets how
far the faint 0.25 threshold reaches (d < about 2.4). Input: the same ImageNet normalisation as our model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CRACK_DISTANCE = 2.0  # px: the authors' crack/no-crack line on the distance map
SLOPE = 3.0  # logit per pixel of distance (assumption, see above)


class _ConvRelu(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class _Up(nn.Module):
    """Double the size, then two 3x3 convolutions (indices 1 and 2, as the weights name them)."""

    def __init__(self, cin: int, mid: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(nn.Identity(), _ConvRelu(cin, mid), _ConvRelu(mid, cout))

    def forward(self, x):
        return self.block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


def _vgg16_features() -> nn.Sequential:
    """VGG16's convolutional part, with the layer indices torchvision uses (convs at 0, 2, 5, 7, 10, ...)."""
    layers, cin = [], 3
    for v in (64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"):
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(cin, v, 3, padding=1), nn.ReLU(inplace=True)]
            cin = v
    return nn.Sequential(*layers)


class UNet16(nn.Module):
    """U-Net with a VGG16 encoder, 32 base filters, bilinear upsampling: the layout of the EPFL weights."""

    def __init__(self, nf: int = 32):
        super().__init__()
        enc = _vgg16_features()
        self.encoder = enc
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(enc[0], enc[1], enc[2], enc[3])
        self.conv2 = nn.Sequential(enc[5], enc[6], enc[7], enc[8])
        self.conv3 = nn.Sequential(enc[10], enc[11], enc[12], enc[13], enc[14], enc[15])
        self.conv4 = nn.Sequential(enc[17], enc[18], enc[19], enc[20], enc[21], enc[22])
        self.conv5 = nn.Sequential(enc[24], enc[25], enc[26], enc[27], enc[28], enc[29])
        self.center = _Up(512, nf * 16, nf * 8)
        self.dec5 = _Up(512 + nf * 8, nf * 16, nf * 8)
        self.dec4 = _Up(512 + nf * 8, nf * 16, nf * 8)
        self.dec3 = _Up(256 + nf * 8, nf * 8, nf * 2)
        self.dec2 = _Up(128 + nf * 2, nf * 4, nf)
        self.dec1 = _ConvRelu(64 + nf, nf)
        self.final = nn.Conv2d(nf, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool(c1))
        c3 = self.conv3(self.pool(c2))
        c4 = self.conv4(self.pool(c3))
        c5 = self.conv5(self.pool(c4))
        center = self.center(self.pool(c5))
        d5 = self.dec5(torch.cat([center, c5], 1))
        d4 = self.dec4(torch.cat([d5, c4], 1))
        d3 = self.dec3(torch.cat([d4, c3], 1))
        d2 = self.dec2(torch.cat([d3, c2], 1))
        d1 = self.dec1(torch.cat([d2, c1], 1))
        return self.final(d1)  # the distance to the nearest crack centre line, px


class Adapter(nn.Module):
    """The distance model as our pipeline expects it: a logit, above 0 exactly where the distance is under 2 px."""

    def __init__(self, net: UNet16):
        super().__init__()
        self.net = net

    def forward(self, x):
        return SLOPE * (CRACK_DISTANCE - self.net(x))


def is_epfl(state: dict) -> bool:
    return "encoder.0.weight" in state and "dec1.conv.weight" in state and "model" not in state


def load(path: Path, device: str = "cpu") -> nn.Module:
    net = UNet16()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return Adapter(net).to(device).eval()
