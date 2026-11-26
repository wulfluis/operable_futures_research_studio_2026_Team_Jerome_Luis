# Autoencoder: what looks different from sound concrete (D89)

## Why
Jerome, 21 Nov, after the team's weld hackathon project (an autoencoder trained only on sound welds; its redraw error shows the defects). On Second Look the crack model finds cracks, but it misses faint hairlines (the photo is reduced to 832 px) and it does not look for other defects: rust stains, damp, spalling. An autoencoder trained only on sound concrete redraws any photo as sound concrete, so where the photo differs its error lights up. That gives three things:
1. **A heatmap** of what looks different, on any element: walls, ceilings, beams, columns.
2. **Other defects, separate from the cracks**: unusual areas that no traced crack explains, labelled by written colour rules.
3. **A score for the whole photo**: how unusual the surface is compared with the sound concrete it learned.

It never finds or measures a crack (the crack model and the bank card do that) and never sets a level (rule 5).

## Training data: sound concrete only
| Source | Photos | What they show |
|---|---|---|
| METU, crack-free class [36] | 20,000 | concrete surfaces of campus buildings, close up |
| SDNET2018, uncracked walls only [22] | 14,287 | concrete walls; bridge decks and pavements are left out (off-concept) |
| PHI-Net task 2, undamaged class [26][55] | 5,529 | whole elements: columns, beams, walls, slabs (CC BY-NC-SA 4.0: research and teaching only) |
| Özgenel segmentation tiles without a crack [49] | as in our split | building concrete, close up |

5 % of each source is held out for the thresholds. Our own field photos are never used for training. **Known limit:** no source is labelled "ceiling"; ceilings seen from below are covered only where they appear in these sets.

## The model (`src/anomaly/`)
- A small convolutional autoencoder on 64 × 64 colour tiles (colour is needed for rust), with a narrow code so it can only redraw what sound concrete looks like (assumption: the size is set by the held-out error, see the experiment note).
- Training images are scaled so their short side is 128 px, then random 64 px crops and flips.
- **Heatmap:** the photo is scaled to a long side of 768 px and cut into overlapping 64 px tiles (step 16); each pixel's error is the squared difference between the photo and the redraw, averaged over colour and over the tiles that cover it.
- **Unusual pixels (illustrative until calibrated):** above a small floor (the 99th percentile of held-out sound pixel errors) AND 6 times the error of the surface around them (41 px window). Chosen on what the viewer shows: at most 2 % of held-out sound photos may show a defect area, and among those the pair that lights up most cracks on the Özgenel validation tiles (`docs/experiments/2026-11-21_anomaly_ae.md`). A first, fixed per-pixel threshold did not fit smooth plaster and was replaced.
- **Photo score:** mean tile error against held-out sound tiles. Not shown yet: not calibrated for photos at the app's scale.

## Other defects, separate from the cracks (written rules, `src/anomaly/regions.py`)
- Unusual pixels within a few pixels of a traced crack (D88) belong to that crack and are left out; so is the bank card.
- The rest is cleaned (small specks removed) and split into areas. Each area gets one label from its colour compared with the surface around it:
  - **Possible rust stain:** clearly more orange-brown than the surface around it (hue 5° to 45°, saturation well above the surface's).
  - **Dark patch (damp or dirt?):** clearly darker than the surface around it.
  - **Other unusual area:** anything else the model did not expect.
- Every label says "possible": an engineer decides. The colour thresholds are illustrative until calibrated.

## In the crack viewer (D83, D88)
Separate layers, each switched on and off: the cracks (crack 1, 2, 3 in their colours, D88), the other defects (outlined, each with its label), and the heatmap. A chip counts each kind ("Rust? 1", "Dark 1", "Unusual 2"). The photo's score is not shown until it is calibrated.

## What it does not see (21 Nov, D90)
- **Soft stains.** Drawn rust stains and damp patches gave almost no error: the autoencoder redraws soft, blurred colour easily and lights up sharp, thin differences (cracks, edges). A colour rule on the photo itself (more orange-brown or darker than the wall's slowly changing background) was tried and dropped: at any setting that kept false areas on held-out sound concrete near 2 %, it also missed the drawn stains, and looser settings flagged 10 to 27 % of sound photos.
- **Small photos.** On 837 PHI-Net photos (224 px, enlarged) it showed no area, with or without spalling: enlarging blurs every detail under its floor.
- So the "possible rust" and "dark patch" labels only apply to sharp unusual areas; finding stains needs real close-up photos of rust, damp and spalling, to calibrate and test on.

## Possible spalling (D93, 21 Nov)
On close-ups of plastered walls, SAM 3.1 is asked for "missing plaster" and "broken plaster" (`src/anomaly/spalling_sam.py`); written rules keep an area only when its score is at least 0.5, it is an area and not a crack line (the traced cracks are removed first, and it survives an erosion of 1.2 % of the long side), it covers 0.5 % to 40 % of the photo, and its colour differs clearly from the wall around it (CIELAB difference at least 15). Shown as "Possible spalling or missing plaster (SAM 3.1, unverified)", a dashed outline and a "Spalling?" chip. Tried on 12 photos only (4 real with spalling, 2 field photos without, 6 drawn): it outlined the spalled patch of a survey photo and two plaster flakes of another, missed a large spall and a small one, and marked nothing on the 8 photos without spalling once the colour difference was raised from 12 to 15 (a door frame had 13.3). Not a measured result: real close-ups of spalling are needed. "Exposed concrete" is not asked: on painted walls it covered whole walls, and on PHI-Net it lit half the photos without spalling (D91).

### A spall is not a crack (D111, 26 Nov)
The crack model often marks a spall as a very wide crack. Now:
- the spalling check runs **before** the crack is measured, and only the crack's thin lines are taken out of SAM's area first (a wide area the crack model marked stays a candidate);
- the outline drawn is SAM's whole area, with the crack lines along its rim;
- an open crack's dark gap is not a spall: not much darker than the wall around it (CIELAB lightness at most 35 darker), and not a small, long, thin and dark strip (elongation at least 3.5, at least 18 darker, under 5 % of the photo); illustrative, from the 76 real cases checked by eye;
- the crack is cut out of the spall (grown by 1 % of the photo's long side), also after SAM's crack extension (D91): it is traced and measured only outside it, so the width, the opening and the sliding are the crack's;
- spalling on the close-up, or on the element photo at the crack, gives the type "spalling with exposed bar" (rule 2 of the type rules), so the level is raised by the written matrix;
- on the element photo, no crack line is drawn inside a spalled area or on exposed steel.
Evidence (76 real cases, 26 Nov): same answers as before (crack found 71), spalling found on 16 after the dark-gap rule (19 before it: 3 were open cracks), the false widths across spalls gone. Still "unverified": one reviewer.

## Not changed (until evaluated)
Shown only: the areas and the score do not raise the level or send a case to "Expert needed" yet. After the evaluation (rule 6) the team decides whether a high score sends the case to an expert and whether a confirmed rust stain feeds the risk matrix's "rust staining" raise. Never lowers anything.

## Evaluation (`docs/experiments/2026-11-21_anomaly_ae.md`)
- Held-out sound tiles: the share flagged as unusual (false alarms).
- Özgenel test tiles with crack masks: how well the heatmap separates crack pixels from sound pixels (area under the ROC curve), as a check that it sees what differs, not as a crack finder.
- The team's field photos, by eye only (never their numbers in the repository).

## Tests
`tests/test_anomaly.py`: the area rules on drawn images (an orange patch is possible rust, a dark patch is dark, crack areas and the card are left out, specks are dropped); the heatmap's tiling covers the whole photo; the thresholds come from the held-out errors; the viewer receives the layers.
