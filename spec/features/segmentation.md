# Feature: crack segmentation (`src/segmentation/`)

Status: day 1 of the first model (29 Oct): data preparation done (9,441 pairs: crack-seg 3,717 / 199 / 112, CrackSeg9k 4,365 / 439 / 609 train / val / test). Day 2 (29 Oct): trained and tested once: on 721 unseen photos, 95.8 % of cracks found, 3.1 % missed, 2.9 % false alarms (`docs/experiments/2026-10-29_unet_first_training.md`). 30 Oct: no-crack photos added; model v3 (test: 538 of 550 cracks found, 12 missed, 7 false alarms out of 653 no-crack photos; D32). Box [6] of the pipeline story (figure 01d), scope D30. 9 Nov: model v4 in use, Özgenel building-concrete masks added (on building concrete 17 cracks missed and 5 false alarms against 30 and 21 for v3; on the old public close-ups 18 missed against 12; D48).

## Purpose
Mark which pixels of a usable photo are crack. The mask feeds the width measurement (mm, from the scale card) and then the written risk matrix. The model never gives a level itself.

## Model (`src/segmentation/train.py`)
U-Net with a ResNet-34 encoder pre-trained on ImageNet (`segmentation-models-pytorch`), one output channel (crack / not crack). No Ultralytics code (AGPL-3.0, D9).
Pre-trained on public crack sets now; fine-tuned on our own engineer-labelled photos later (learn loop, box [9]).

- Training: train split only; random flips, 90° turns and brightness/contrast changes; loss = binary cross-entropy + Dice (cracks are about 1 to 2 % of the pixels, Dice keeps them from being ignored); AdamW, cosine learning rate, 25 epochs, batch 16, mixed precision on the GPU (illustrative).
- The model kept is the epoch with the best crack F1 on the val split. The test split is used once, at the end, for the reported score.
- Output: `models/unet_crack.pt` and `data/results/segmentation/` (training log, test scores, overlays). Neither is committed.

## Crack or no crack (`src/segmentation/decide.py`, written rule)
The model gives every pixel a crack probability. A written rule turns that map into a statement:

| Statement | Rule (illustrative until calibrated) |
|---|---|
| Crack found | the largest connected group of pixels above 0.5 has at least 150 pixels (at 416 px) |
| No crack found | no group reaches 50 pixels |
| Expert needed | in between: the model is unsure |

Confidence = mean probability of the pixels in the largest group. "No crack found" is never a safety statement (rule 1). The thresholds are checked on the val split, never on the test split.

### Straight edges are not cracks (10 Nov, D51)
The model reads the line between a wall and a ceiling, a wall corner or a frame as a crack (a thin dark line). Before the rule above, each group of crack pixels is searched for an edge (illustrative until calibrated):

1. **Its main straight line.** Over every direction in 1° steps, the narrow band (8 px wide at the working size) holding the most of the group's pixels is found (a Hough transform); a straight line is fitted to those pixels and refitted until it stops moving. Pixels within 1.5 % of the line's length (at least 4 px) belong to it. A corner stays straight under perspective; a crack wanders out of such a band.
2. **That line is an edge** when it holds at least half of the group's pixels, is at least 40 % of the photo's short side long, and reaches a side of the photo (within 3 % of the long side). One side is enough: a wall-ceiling line often ends at the room corner inside the photo.
3. **What is left of the group** is split into pieces. A piece with at least 85 % of its pixels on its own straight line (the short corner where two walls meet below the ceiling) goes with the edge. A piece that wanders (a crack leaving the corner) stays a crack.

- The edge pixels are taken out; everything else in the photo is judged by the rule above as before. So an edge plus a crack is still "Crack found", with only the crack drawn (team, 10 Nov).
- An ignored edge is not drawn in red and is never measured.
- The reason says so: "A long straight line running across the photo was read as a corner or edge and ignored; a crack running exactly along an edge is not checked."
- Known risk, accepted by the team (D51): a real crack that is very straight and reaches a side of the photo, such as a crack running exactly along a wall-ceiling joint, is ignored; so is a short straight crack hanging from an edge.
- First version (same day) asked for two sides of the photo and a straight whole group; the first app test (a wall-ceiling line from the top of the photo to the room corner, with a short bend down) was still "Crack found". Checked again on the red line of that result's screenshot: "No crack found", edge ignored.
- Not handled: an edge the model marks in several separate pieces, each shorter than 40 % of the short side.

### Dark patches are not cracks (11 Nov, D54)
After the edge rule, a group of at least 50 pixels (at 416 px) whose length is at most 3 times its width (`BLOB_MAX_ELONGATION`, illustrative; width from the distance transform, 95th percentile, so one wide spot does not turn a long crack into a patch) is not counted. If the only thing left is a large patch (at least the crack size), the answer is Expert needed with "a dark patch, not a line: it could be a stain, an object or spalling", and the patch is drawn; it is never No crack found. A wide crack is still a crack: width alone never makes a patch.

## Scores (test split, `docs/experiments/`)
- Pixels: precision, recall, F1 and IoU for the crack class, per dataset.
- Photos: crack / no crack / Expert needed against the truth (crack if the mask has any crack pixel), as a confusion table (`src/triage/evaluate.py` style). The worst error is a missed crack.

## Data preparation (`src/segmentation/prepare.py`)
Sources (licences and checks in `docs/data_map.md`):

| Set | Used | Masks | Split |
|---|---|---|---|
| CrackSeg9k Vol1 [25] | 5,413 image + mask pairs (Vol2 not downloaded) | grey masks with soft edges (0 to 255); crack = value above 127 (assumption) | own split, grouped by source photo (below) |
| crack-seg [33] | 4,029 images | YOLO polygons filled into masks | kept as published (train 3,717, val 200, test 112) |

- Every image and mask is resized to 416 x 416 px (crack-seg's own size, a multiple of 32 as the U-Net needs). Images: bilinear. Masks: resized as grey, then thresholded, so thin cracks are not lost.
- CrackSeg9k's published train/test lists **leak**: tiles cut from the same source photo sit in both (Rissbilder: 116 of the source photos; DeepCrack: 37; GAPS384: 40). A test score on them would be too optimistic. We split by **source photo** instead: 80 / 10 / 10 % train / val / test, decided by a hash of the source-photo key, so the split is fixed and repeatable.
- crack-seg's train set holds up to 6 augmented copies of one photo (`<id>.rf.<hash>`); no id crosses splits, so its split is kept.
- Images with an empty mask stay: CrackSeg9k's 1,411 "noncrack" images teach what is not a crack. The one crack-seg image with an empty label file is dropped (likely a missing annotation, assumption).
- Output: `data/processed/segmentation/{train,val,test}/{images,masks}/` as PNG (mask 0 or 255), plus `index.csv` (file, set, split, source-photo group, crack fraction). Never committed (`data/` is ignored).

## Extra no-crack photos (`src/segmentation/negatives.py`, model v2, 30 Oct)
Model v1 marked any thin dark line as crack: on 30 full-scene building photos it said "Crack found" 30 times (`spec/features/class_demo.md`). v2 adds close-ups of surfaces with no crack, with an empty mask, so the model learns lines that are not cracks:

| Set (all CC BY 4.0) | Taken | Why |
|---|---|---|
| METU negatives [36] | 3,000 random of 20,000 | concrete of campus buildings: stains, pores, formwork |
| SDNET2018 uncracked walls (W/UW) [22] | 1,500 random of 14,287 | concrete walls; tiles of one photo kept in one split (`<photo>-<tile>`) |
| Darbhanga, images without a crack box [30] | all of them (train and val) | brick masonry: mortar joints are the classic false alarm; brick infill is common in informal buildings |

- Same 416 px, same split by hash (METU has no photo ids, so neighbouring tiles may sit in different splits; acceptable for no-crack photos, assumption).
- SDNET2018's "uncracked" tiles are known to contain a few missed cracks; kept to 1,500 so the noise stays small (assumption).
- v1 is kept as `models/unet_crack_v1.pt`. A new model replaces it only if, on the same test photos, it misses no more cracks than v1 (worst error first) and has fewer false alarms.
- Result (`docs/experiments/2026-10-29_unet_first_training.md`): v2 (all three sets) missed more cracks, mostly asphalt road tiles, and was rejected. **v3** trains with the Darbhanga brick photos only, each shown 5 times per epoch (`train.py --exclude metu_ sdnet_ --repeat darb_=5`); METU and SDNET stay in val and test as no-crack checks. v3 passed on the test photos (12 missed against 17, 7 false alarms against 21, 0 of 22 brick) and is in use; on the val photos it missed 5 against 1 for v1, so the pilot photos decide next.

## Building concrete with crack masks (`src/segmentation/ozgenel.py`, model v4, 9 Nov)
The reference thesis [46] trained its segmenter on the Özgenel Concrete Crack Segmentation Dataset [49]: 458 high-resolution photos of concrete on METU campus buildings with black-and-white crack masks (CC BY 4.0; SHA-256 given by Mendeley and checked on download). Concrete of buildings, with crack masks, is what our training data lacks (`docs/data_map.md`). Not in CrackSeg9k.
- **Tiles.** A photo is far larger than the U-Net's 416 px, and shrinking it whole would make its cracks vanish. Each photo is cut into square tiles of one third of its short side, each resized to 416 px (illustrative: the close-up scale of the other sets). The photos are 2448 x 3264 or 4032 x 3024: both give a 4 x 3 grid of 12 tiles, 5,496 in all (train 4,236, val 672, test 588; 2,661 with crack). Cracks, a median 21 px wide in the masks, end up about 11 px wide. Every tile is kept, with or without crack.
- **Rotation.** 194 of the 458 photos carry an EXIF rotation tag (179 turned 90°, 15 turned 180°) that their masks do not have: the masks are drawn on the photo as it is shown. Each photo is turned by its tag before tiling; a photo and mask that still differ in size stop the run (never stretched). The first v4 run (9 Nov) stretched them instead, so 42 % of the masks sat beside their cracks and both models were scored against wrong answers; that run was discarded. Check: on all 458 pairs the masked pixels are now clearly darker than the rest (before the fix, 191 were not).
- **Masks.** Resized as grey with area averaging, then a pixel is crack when at least a quarter of it was crack (illustrative: the 0.5 threshold of CrackSeg9k would erase cracks one or two pixels wide when a tile shrinks to 0.4 of its size).
- **Split by photo**, by the same hash as the other sets: every tile of a photo stays in one split. File prefix `ozg_`.
- **Leak to avoid.** The METU classification tiles [36] are cut from these same 458 photos and sit in our val and test sets as no-crack checks (they cannot be traced to their photo). So for the v3 / v4 comparison the METU tiles are **left out of the test**; SDNET2018 and Darbhanga stay as no-crack checks.
- **Training** as v3 (`--exclude metu_ sdnet_ --repeat darb_=5`), with the Özgenel tiles added. Saved as `models/unet_crack_v4.pt`; `models/unet_crack.pt` is not touched.
- **Replacement rule** (as v1 to v3): on the same test photos without METU, v4 replaces v3 only if it misses no more cracks and raises no more false alarms. Also reported: both models on the Özgenel test tiles.

## Limits
- Mostly pavement, concrete surfaces and walls; few columns and beams; no scale in any image. Off-concept for informal buildings (D14), used only to pre-train crack pixels.
- crack-seg polygons are drawn wider than the crack (checked on a sample, 29 Oct); CrackSeg9k masks are thin. Both teach where the crack is; width in mm must not be calibrated on crack-seg masks.
- Thresholds (127, 416 px, 80/10/10) are illustrative until checked against engineer-labelled photos.

## One photo in, one statement out (`src/segmentation/predict.py`)
Quality gate first (blurred, too dark or too bright: Retake; a missing card does not stop crack or no crack, it makes the width unknown; with a bank card the width is measured, `spec/features/app_measure.md`, D46). The photo's long side is scaled to 832 px (illustrative) and padded to a multiple of 32; the crack-rule thresholds grow with the photo size. Output: the statement, a one-sentence reason with the confidence, and the photo with the crack in red.

## Video (`src/viz/segmentation_video.py`)
1920 x 1080 MP4 of test photos from every outcome, misses and false alarms included: photo, crack fades in red, statement, and what the dataset says.

## Tests
`tests/test_segmentation_prepare.py`: polygon to mask, soft mask threshold, source-photo keys, split is repeatable and never splits a group. `test_segmentation_decide.py`: the crack rule and pixel scores. `test_segmentation_predict.py`: retake, crack, no crack and the overlay with a stand-in model. `test_segmentation_evaluate.py`: outcomes and a full run with an untrained model. `test_segmentation_video.py`: frames, wrapping, photo choice.
