# Models

**Download the model in use:** [unet_crack.pt](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/download/model-v4/unet_crack.pt) (release [model-v4](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/tag/model-v4), SHA-256 `a13c1348402ca6c3b3a1509bd5b1ca539641c971bb83326f3382e8b53b8d7a2c`) and put it in this folder.

**Also download** (release [models-element-scene-v1](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/tag/models-element-scene-v1)), into this folder:
- [element_convnext_tiny_phinet6_v2.pt](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/download/models-element-scene-v1/element_convnext_tiny_phinet6_v2.pt), SHA-256 `2322d9e14e843df25fff2031b1fb290d7fc9b831563df74ce7031c42440814b5`
- [scene_convnext_tiny_phinet1.pt](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/download/models-element-scene-v1/scene_convnext_tiny_phinet1.pt), SHA-256 `67301a79531c231a15eb930f7341613c9436d4f69257103aa2b8f4ce276de23a`
Without them the app still runs, without the element and scene checks. Both are trained on PHI-Net (CC BY-NC-SA 4.0): non-commercial, research and teaching only.

Weights are attached to GitHub Releases, never committed to the repository (D39). Note: one training set (crack-seg [33]) has an unclear licence (Public Domain on its page, AGPL-3.0 in its file header); the release notes say so.

## In use
| File | Model | Trained on | Test result | Details |
|---|---|---|---|---|
| `unet_crack.pt` (v4, about 98 MB) | U-Net, ResNet-34 encoder pre-trained on ImageNet; marks crack pixels | 12,477 photos: 8,082 public crack close-ups (CrackSeg9k [25], crack-seg [33]), 4,236 tiles of 353 building-concrete photos with crack masks (Özgenel [49]) and 159 brick walls without cracks (Darbhanga [30]); 25 rounds on a laptop GPU | Building concrete (588 unseen tiles): 17 cracks missed, 5 false alarms, pixel F1 0.85 (v3: 30, 21, 0.78). Public close-ups (906 unseen photos): 18 missed, 6 false alarms, pixel F1 0.64 (v3: 12, 6, 0.63) | `docs/experiments/2026-10-29_unet_first_training.md`, D31, D32, D48 |

| `element_convnext_tiny_phinet6_v2.pt` (about 106 MB) | ConvNeXt-tiny; element from a photo of the whole element: beam, column, wall, other | PHI-Net task 6 [53][55], 4,755 photos | 579 test photos: 66.8 % answered, 86.3 % of those right, column confidently called something else 4.9 % | `docs/experiments/2026-11-10_element_resnet_phinet6.md`, D52 |
| `scene_convnext_tiny_phinet1.pt` (about 106 MB) | ConvNeXt-tiny; scene check: close-up, whole element, whole building | PHI-Net task 1 [53][55], 24,309 photos | 2,997 test photos: 93.4 % right, no whole building let through, 4.0 % of close-ups sent to retake | `docs/experiments/2026-11-11_scene_gate_phinet1.md`, D54 |

Kept for comparison: `unet_crack_v1.pt`, `unet_crack_v2.pt`, `unet_crack_v3.pt` (release model-v3); `unet_crack_v4.pt` is the same as `unet_crack.pt`.
Known limit: it only knows close-ups; whole-building photos are read as cracks (28 of 30).
The app loads `unet_crack.pt` (`src/app/second_look.py`); it runs on a GPU or, slower, on the CPU.

## Planned (pipeline v3, not trained yet)
Material model (concrete or not), a slab class for the element model, crack-pattern model.
