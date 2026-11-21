# Autoencoder on sound concrete: first training and calibration (21 Nov 2026, D89)

Spec: `spec/features/anomaly_ae.md`. Code: `src/anomaly/` (`py src/anomaly/train.py`; `--recalibrate` keeps the weights). Model files: `models/anomaly_ae.pt` (4.5 MB) and `models/anomaly_ae.json` (thresholds and these results), not committed.

## Data (sound concrete only)
| Source | Train | Held out |
|---|---|---|
| METU crack-free class [36] | 19,000 | 1,000 |
| SDNET2018 uncracked walls [22] | 13,573 | 714 |
| PHI-Net task 2 undamaged [26][55] | 5,253 | 276 |
| Özgenel segmentation tiles without a crack [49] | 2,059 | 108 |

Every 20th photo of each source is held out. Images are scaled to a 128 px square; training uses random 64 px colour crops with flips. 25 rounds, Adam, on a laptop GPU: 3.5 min.

## The first thresholds did not fit (kept here as a finding)
A fixed threshold per pixel (99th percentile of held-out pixel errors) flagged almost nothing on real photos: on the Özgenel test tiles 3 % of crack pixels, and on the team's field photos the crack stayed below it. The threshold was set on rough concrete (METU, SDNET), where errors are large everywhere; on smooth painted plaster even a crack has small errors. The heatmap itself ranked crack pixels above sound ones well (area under the ROC curve 0.92).

## Calibration that is used
A pixel is unusual when it is above a small floor AND several times the error of the surface around it (41 px window at 768 px). The pair is chosen on what the viewer shows, areas after the rules of `src/anomaly/regions.py`, not on single pixels:
- condition: at most 2 % of held-out sound photos show any defect area (illustrative);
- among those, the pair that lights up the most cracks on the Özgenel **validation** tiles (at least 10 unusual pixels on the crack).

| Floor (percentile of sound pixel errors) | Ratio | Sound photos with a false area | Validation crack tiles lit |
|---|---|---|---|
| 99 | 2 | 13.7 % | 285 of 319 |
| 99 | 3 | 8.7 % | 285 of 319 |
| **99** | **6** | **1.7 %** | **284 of 319** |
| 99.5 | 6 | 0.5 % | 222 of 319 |
| 99.9 | 6 | 0 % | 30 of 319 |

Chosen: floor at the 99th percentile, ratio 6. False areas on held-out sound photos: 1.2 % dark patch, 0.5 % other, 0 % rust.

## Test (Özgenel test tiles, never seen in training or calibration)
| Measure | Result |
|---|---|
| Crack pixels ranked above sound pixels (area under the ROC curve) | 0.92 |
| Crack tiles where the heatmap lights up the crack | 256 of 281 |
| Crack-free tiles showing a false defect area | 0 of 307 |
| Photo score above 1 (tiles with / without a crack) | 0 % / 0 % |

The photo score (mean tile error against held-out sound tiles) never passed 1: the held-out photos are scored at 128 px and the app scores photos at 768 px, so the two do not compare. **It is not shown** until it is calibrated on photos at the app's scale.

## Field photos (by eye only)
On the team's two field photos the heatmap follows the whole crack, including the faint fork and the top end that the crack model missed, plus the door frame at the edge; the bank card lights up and is left out by rule. No defect area is shown on either photo, which is right: the wall is clean apart from the crack.

## What it does not see (added 21 Nov, D90)
- **Soft stains:** on drawn photos with a rust stain or a damp patch beside a crack, the heatmap lit up the crack and the card, not the stains (their error stayed near zero): the autoencoder redraws soft colour easily.
- **A colour rule was tried and dropped.** Pixels more orange-brown or darker than the wall's slowly changing background (median over a quarter of the photo, at 96 px). On 1,049 held-out sound photos:

| Saturation above background | Darker than background | Smallest area | Sound photos with false rust | with a false dark patch | Drawn stains found |
|---|---|---|---|---|---|
| 0.15 | 0.15 | 0.3 % | 5.7 % | 16.2 % | some |
| 0.25 | 0.25 | 0.3 % | 3.6 % | 9.1 % | one rust stain |
| 0.25 | 0.25 | 1 % | 1.0 % | 4.3 % | none |

No setting both kept false areas low and found the drawn stains; tuning it on our own drawings would be circular.
- **Small photos:** 837 PHI-Net task 3 test photos (310 with spalling, 527 without), 224 px enlarged to 672 px: no area on any of them.

## Limits
- No photos with rust, damp or spalling were used to test the colour rules: the rules are tested on drawn images only (`tests/test_anomaly.py`). Real rust photos are needed before anyone trusts the "possible rust" label.
- No source is labelled "ceiling"; most sound photos are walls and campus concrete.
- Shown only: nothing here changes a level or sends a case to an expert.
