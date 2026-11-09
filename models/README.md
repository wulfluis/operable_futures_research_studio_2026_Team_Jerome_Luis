# Models

**Download the model in use:** [unet_crack.pt](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/download/model-v4/unet_crack.pt) (release [model-v4](https://github.com/iaac-maai/operable_futures_research_studio_2026_Team_Jerome_Luis/releases/tag/model-v4), SHA-256 `a13c1348402ca6c3b3a1509bd5b1ca539641c971bb83326f3382e8b53b8d7a2c`) and put it in this folder.

Weights are attached to GitHub Releases, never committed to the repository (D39). Note: one training set (crack-seg [33]) has an unclear licence (Public Domain on its page, AGPL-3.0 in its file header); the release notes say so.

## In use
| File | Model | Trained on | Test result | Details |
|---|---|---|---|---|
| `unet_crack.pt` (v4, about 98 MB) | U-Net, ResNet-34 encoder pre-trained on ImageNet; marks crack pixels | 12,477 photos: 8,082 public crack close-ups (CrackSeg9k [25], crack-seg [33]), 4,236 tiles of 353 building-concrete photos with crack masks (Özgenel [49]) and 159 brick walls without cracks (Darbhanga [30]); 25 rounds on a laptop GPU | Building concrete (588 unseen tiles): 17 cracks missed, 5 false alarms, pixel F1 0.85 (v3: 30, 21, 0.78). Public close-ups (906 unseen photos): 18 missed, 6 false alarms, pixel F1 0.64 (v3: 12, 6, 0.63) | `docs/experiments/2026-10-29_unet_first_training.md`, D31, D32, D48 |

Kept for comparison: `unet_crack_v1.pt`, `unet_crack_v2.pt`, `unet_crack_v3.pt` (release model-v3); `unet_crack_v4.pt` is the same as `unet_crack.pt`.
Known limit: it only knows close-ups; whole-building photos are read as cracks (28 of 30).
The app loads `unet_crack.pt` (`src/app/second_look.py`); it runs on a GPU or, slower, on the CPU.

## Planned (pipeline v3, not trained yet)
Material model (concrete or not), element model (column, beam, slab, wall; pre-trained on PHI-Net task 6), crack-pattern model, and a "close-up of a surface?" check before the crack model.
