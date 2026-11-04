# Models

Trained weights (`*.pt`, `*.pth`) are not committed: this repository is public, and one training set (crack-seg [33]) has an unclear licence (Public Domain on its page, AGPL-3.0 in its file header). Ask the team for the file and put it in this folder.

## In use
| File | Model | Trained on | Test result | Details |
|---|---|---|---|---|
| `unet_crack.pt` (v3, about 98 MB) | U-Net, ResNet-34 encoder pre-trained on ImageNet; marks crack pixels | 8,241 photos: 8,082 public crack close-ups (CrackSeg9k [25], crack-seg [33]) and 159 brick walls without cracks (Darbhanga [30]); 25 rounds on a laptop GPU | On 1,203 unseen public photos: 538 of 550 cracks found, 7 false alarms on 653 photos without a crack, pixel F1 0.63 | `docs/experiments/2026-10-29_unet_first_training.md`, D31, D32 |

Kept for comparison: `unet_crack_v1.pt`, `unet_crack_v2.pt` (`unet_crack_v3.pt` is the same as `unet_crack.pt`).
Known limit: it only knows close-ups; whole-building photos are read as cracks (28 of 30).
The app loads `unet_crack.pt` (`src/app/second_look.py`); it runs on a GPU or, slower, on the CPU.

## Planned (pipeline v3, not trained yet)
Material model (concrete or not), element model (column, beam, slab, wall; pre-trained on PHI-Net task 6), crack-pattern model, and a "close-up of a surface?" check before the crack model.
