# First model: crack or no crack with a U-Net (29 Oct 2026)

Box [6] of the pipeline story (figure 01d), scope D30, decision D31.
Spec: `spec/features/segmentation.md`. Code: `src/segmentation/` (prepare, train, decide, predict, evaluate), `src/viz/segmentation_video.py`.

## What was run
- Data: 9,441 public photos with crack masks, CrackSeg9k Vol1 [25] and crack-seg [33], at 416 x 416 px. CrackSeg9k re-split by source photo, because its published lists put tiles of one photo in both train and test (D31).
- Model: U-Net, ResNet-34 encoder pre-trained on ImageNet, 25 epochs on an RTX 5080 Laptop (about 1 hour). Kept: epoch 23, the best crack F1 on the val photos (0.69).
- Crack or no crack: the written rule in `src/segmentation/decide.py` (largest crack line at least 150 px: crack found; under 50 px: no crack found; in between: Expert needed). Thresholds set before training and checked on the val photos (638 photos: 1 missed crack, 3 false alarms); unchanged for the test.
- Test: run once on 721 held-out photos, never used for training or for choosing anything.

## Results on the test photos

Per photo:

| Truth \ Model says | Crack found | Expert needed | No crack found | Total |
|---|---|---|---|---|
| Crack | 527 (95.8 %) | 6 (1.1 %) | **17 missed (3.1 %)** | 550 |
| No crack | 5 false alarms (2.9 %) | 0 | 166 (97.1 %) | 171 |

Per pixel (crack class), how well the red line matches the dataset's crack outline:

| Set | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| crack-seg | 0.77 | 0.77 | 0.77 | 0.63 |
| CrackSeg9k | 0.63 | 0.63 | 0.63 | 0.46 |
| All | 0.65 | 0.65 | 0.65 | 0.49 |

## What the errors are (every error photo was looked at)
- **Missed cracks (17):** 9 are tiles of one photo, a painted decorative facade with very faint hairline cracks (its own labels are broken dashes). 2 are crack-seg labels that are not crack lines (a spalled patch, an image edge). The rest are faint hairline cracks. Weakness: faint hairline cracks on painted plaster.
- **False alarms (5):** all tiles of one photo of a board-marked concrete slab; the model reads the slab edge and a formwork line as a crack.
- **Expert needed (6):** all the same faint painted facade. This is the intended behaviour: when unsure, ask.

## Limits
- Public photos, mostly pavement and concrete surfaces, few columns or beams, no scale card: this shows the model finds crack lines; it does not show it works on informal buildings (D14) or with phone photos. Next check: our own pilot photos.
- The truth is the datasets' own outlines, not an engineer's; crack-seg outlines are wider than the cracks.
- No width in mm yet (day 3, needs the scale card). "No crack found" is never a safety statement (rule 1).
- Errors cluster in a few source photos, so the percentages move a lot with a few photos; treat them as a first indication.

## 30 Oct: models v2 and v3, no-crack photos added
Why: v1 raised false alarms on brick walls (mortar joints) and, in the class-demo check, said "Crack found" on all 30 full-scene building photos (`spec/features/class_demo.md`).
Added to every split, with empty masks (`src/segmentation/negatives.py`, all CC BY 4.0): 3,000 METU concrete close-ups [36], 1,500 SDNET2018 uncracked walls [22], the 203 Darbhanga brick images without a crack box [30]. The test split grows to 1,203 photos (482 more, all without a crack).
- **v2:** trained with all three sets.
- **v3:** trained with the Darbhanga brick photos only, each shown 5 times per epoch; METU and SDNET left out of training (v1 already had 0 or 1 false alarm on them).
Rule written before training (`spec/features/segmentation.md`): a new model replaces v1 only if, on the same test photos, it misses no more cracks and has fewer false alarms.

Test photos (1,203; same photos for all three):

| Model | Crack found | Missed crack | Expert needed | False alarm | of which brick (of 22) | Pixel F1 |
|---|---|---|---|---|---|---|
| v1 | 527 | 17 | 6 | 21 | 15 | 0.65 |
| v2 | 527 | 23 | 0 | 7 | 0 | 0.61 |
| **v3** | **538** | **12** | 0 | **7** | **0** | 0.63 |

Val photos (1,082), for the record: v1 missed 1 crack and raised 21 false alarms (18 brick); v2 missed 24 (22 of them GAPS384 asphalt road tiles: grainy concrete taught as "no crack" looks like asphalt) and raised 3; v3 missed 5 (3 asphalt) and raised 9 (0 brick). **On val, v3 misses more than v1 (5 against 1); on test, fewer (12 against 17).** Val and test together: v3 missed 17, v1 18; false alarms 16 against 42.
Decision: v3 passes the rule on the test photos and is now `models/unet_crack.pt`; v1 and v2 are kept (`models/unet_crack_v1.pt`, `_v2.pt`). The mixed val result is why the pilot photos must decide next.

Whole scenes (30 Haiti 2010 engineer photos, local only): v1 30 "Crack found", v2 29 (1 Expert needed), v3 28 (2 No crack found). Close-up no-crack photos do not teach the model what a window frame or a wire is; whole scenes need a "close-up of a surface?" check before the model (next step).

## 9 Nov: model v4, building concrete with crack masks (D48)
v4 adds the Özgenel Concrete Crack Segmentation Dataset [49]: 458 photos of concrete on METU campus buildings with crack masks, cut into 12 tiles each (`src/segmentation/ozgenel.py`; train 4,236, val 672, test 588 tiles, split by photo). Trained as v3 (`--exclude metu_ sdnet_ --repeat darb_=5`), 25 rounds; best val F1 0.776 (round 21; v3's val set had no Özgenel tiles, so the two are not comparable).
**A data bug, found by looking at the errors.** In a first run the "false alarms" on Özgenel tiles were real, clearly visible cracks with an empty mask: 194 photos carry an EXIF rotation their masks do not, and the preparation had stretched the masks instead of turning the photos. Fixed (photos turned by their tag, a size mismatch now stops the run, a test reproduces the case); on all 458 pairs the masked pixels are now darker than the rest (191 were not). That run was discarded and v4 retrained.
**Test** (photos never seen; METU tiles left out because they are cut from the Özgenel photos):

| Test set | Model | Cracks found | Missed | Expert needed | False alarms | Pixel F1 |
|---|---|---|---|---|---|---|
| Building concrete, Özgenel (281 crack / 307 clean tiles) | v3 | 250 | 30 | 1 | 21 | 0.78 |
| | **v4** | 261 | **17** | 3 | **5** | **0.85** |
| Public close-ups without METU (550 crack / 356 clean photos) | v3 | 538 | **12** | 0 | 6 | 0.63 |
| | **v4** | 530 | 18 | 4 | 6 | 0.64 |
| All 1,494 | v3 | 788 | 42 | 1 | 27 | |
| | **v4** | 791 | **35** | 7 | **11** | |

v4 is better on building concrete, the case closest to the concept, and slightly worse on the old public close-ups (16 of its 18 misses are CrackSeg9k photos, mostly pavement). Judged on all test photos, missed cracks first, it replaces v3 (D48). Its 6 extra misses there are its known weakness; the pilot photos decide next.
Outputs: `data/results/segmentation/v4_compare/` (v3 and v4 on both sets, one overlay per photo), `train_log_v4.csv`.

## Outputs (local, not committed)
- `models/unet_crack.pt`: the model in use (v3 since 30 Oct); `unet_crack_v1.pt`, `unet_crack_v2.pt`; training logs `train_log.csv` (v1), `train_log_v2.csv`, `train_log_v3.csv`.
- `data/results/segmentation/train_log.csv`: loss and val F1 per epoch.
- `data/results/segmentation/test/` (v3; v1, v2 and v3 also in `data/results/segmentation/v1/`, `v2/`, `v3/`): `scores.json`, `photos.csv`, and one overlay per photo in `crack_found/`, `no_crack_found/`, `missed_crack/`, `false_alarm/`, `expert_needed/` (left: the dataset's outline in white; right: the model's crack in red, with the statement).
- `data/results/segmentation/second_look_test.mp4`: video of 12 test photos from every outcome v3 produced, misses and false alarms included (v3 sent no test photo to an expert). Photos from CrackSeg9k (CC0), crack-seg (public domain) and, for no-crack photos, METU, SDNET2018 and Darbhanga (CC BY 4.0, credit the authors); credit them on the slide.
