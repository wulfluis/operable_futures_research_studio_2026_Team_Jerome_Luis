# Scene check on PHI-Net task 1: close-up, whole element or whole building? (11 Nov 2026)

Experiment of D54 (`spec/features/scene_gate.md`, `src/experiments/scene_gate.py`). Why: the crack model reads frames, blinds, wires and corners as cracks in photos that are not close-ups (app tests of 10 and 11 Nov).

## Setup
ConvNeXt-tiny pre-trained on ImageNet, trained as the element model's second run (D52), 8 rounds, on the laptop GPU (17 minutes). PHI-Net task 1 [53][55]: 24,309 training images (10 % of them to pick the best round, the temperature and the threshold), 2,997 test images used once (object 962, close-up 965, structure 1,070). CC BY-NC-SA 4.0, research only (D6). Numbers: `data/processed/scene_gate.json`, log `data/results/scene/train.log` (local).

## Result
Best validation accuracy 0.941 (round 5). Temperature 0.76. `CLOSE_UP_MIN` chosen on validation: **0.10** (the highest value that sends at most 5 % of close-ups to retake).

Test images, top answer:

| True \ read as | whole element | close-up | whole building |
|---|---|---|---|
| whole element (962) | 877 | 29 | 56 |
| close-up (965) | 82 | 882 | 1 |
| whole building (1,070) | 30 | 0 | 1,040 |

Accuracy 93.4 %. With the rule at 0.10 (test):
- **Whole buildings let through as a close-up: 0 %** (0 of 1,070).
- Whole elements let through as a close-up: 6.4 %.
- Real close-ups sent to retake: 4.0 %.

## Checked on other photos (not PHI-Net, no training on them)
| Photos | Pass as a close-up |
|---|---|
| Building-concrete crack close-ups, Özgenel test tiles [49] | 588 of 588 (100 %) |
| crack-seg test close-ups [33] | 111 of 112 (99 %) |
| CrackSeg9k test close-ups [25] | 577 of 609 (95 %) |
| Brick wall close-ups, Darbhanga test [30] | 19 of 22 (86 %) |
| Haiti 2010 photos, 200 at random (mostly buildings and rooms, some close-ups), local only | 5 of 200 (2 %) |

## What it means
The crack model no longer sees most photos it cannot read: room views and whole buildings get "Retake: hold the phone 20 to 40 cm from the crack". Real crack close-ups pass, brick a little less often (a needless retake costs one more photo). Not solved: a close-up of a window frame, a socket or tile joints is still a close-up; that needs the crack model to learn them (D54, step 3). Weights: `models/scene_convnext_tiny_phinet1.pt` (local, not committed).
