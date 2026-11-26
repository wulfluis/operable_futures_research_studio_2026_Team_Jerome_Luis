# Crack model v5 (with the EPFL photos) against v4 (D100), 22 Nov 2026

## Question
Do the crack photos of E. Corres Sojo's EPFL group make our crack finder better on our kind of buildings? v5 is our model trained as v4 plus the EPFL tiles.

## Data
- Held-out test tiles (never trained on), per group: the old public close-ups (743, as in D32 and D48), Özgenel building concrete (588: 281 with a crack, 307 without) [49], EPFL buildings after earthquakes (115 tiles of 3 photos, all with a crack) [108], EPFL lab walls (87, all with a crack) [109].
- Our 76 real two-photo cases through the whole app (the field pairs and the test set reviewed by eye, D94), all with a real crack.

## v5's training
As v4 (`--exclude metu_ sdnet_ --repeat darb_=5`, 25 rounds), plus 6,511 EPFL building tiles and 369 EPFL lab tiles (`src/segmentation/epfl.py`). The authors' own split is kept by photo, so no photo is in training and test. A first run stalled when the GPU was shared and was restarted alone; best validation F1 0.760.

## Results: test tiles (missed cracks / false alarms)
| Group | v4 | v5 |
|---|---|---|
| Old public close-ups (743) | 36 / 5 | 52 / 7 |
| Özgenel building concrete (588) | 19 / 5 | 20 / 8 |
| EPFL buildings (115, all cracked) | 65 / – | 46 / – |
| EPFL lab walls (87, all cracked) | 48 / – | 6 / – |

## Results: our 76 real cases through the app
| | v4 | v5 |
|---|---|---|
| Crack found | 47 | 53 |
| No crack found (a real crack missed) | 9 | 3 |
| Retake (scene check, before any crack model) | 20 | 20 |

## Decision
v5 replaces v4 (D100): on our real cases it misses 3 cracks instead of 9, and on the EPFL buildings 46 instead of 65; it adds 3 false alarms on clean building concrete (8 of 307) and misses 16 more of the old public close-ups (mostly pavement and generic close-ups, off-concept, D14): recorded as its known weakness. The EPFL group's own model does a different job (thin centre lines for their measurement method) and is not ranked here (D104).

## Limits
76 real cases, all with a crack: false alarms on real photos are not measured here, only on the 307 clean Özgenel tiles. The EPFL test photos are 3 buildings only.
