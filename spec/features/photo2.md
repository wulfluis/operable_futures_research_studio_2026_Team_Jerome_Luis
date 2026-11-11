# Feature: what the second photo adds (`src/element/photo2.py`, `src/element/spalling.py`)

Status: first version, 11 Nov 2026, decision D61. Thresholds illustrative until calibrated.

## Purpose
The resident sends two photos (D53): the crack close-up with a bank card, then the whole element. Until now the second photo gave only the element type. It also shows **where the crack is on the element, how large it is next to the element, and whether concrete has spalled or bars are exposed**. The level must use what both photos show (Jerome, 11 Nov), through the written risk matrix (rule 5).

## What each photo gives
| | Photo 1: close-up with the card | Photo 2: the whole element |
|---|---|---|
| Width in mm | **yes** (bank card) | no: the card is too small from afar |
| Crack shape, direction | yes | yes, coarser |
| Element | no | **yes**, element model (D52) |
| Where the crack is on the element | no | **yes**: the crack model on photo 2 |
| How long the crack is next to the element | no | **yes**, as a share of the element |
| Spalling or exposed bars | no (a close-up rarely shows it) | **yes**: spalling model (below) |

## Photo 2 analysis (`src/element/photo2.py`)
- **Assumption:** the resident frames the whole element, top to bottom, as the page asks; the photo is the element. The crack's place is read relative to the photo.
- The crack model (U-Net v4) runs on photo 2 with the same written rules (corners, patches, D51, D54, D60). Its longest crack line is kept as points in the element's own coordinates: u across (0 left, 1 right), v up (0 bottom, 1 top), and its extent (the larger of its width and height share).
- **Position on the element** (illustrative): for a column or a wall, a crack whose centre lies in the top or bottom 20 % of the height is in an **end zone** (near a joint or support), else **middle**; for a beam, the left or right 20 % of the length. No crack found on photo 2: position unknown, which counts as an end zone (never lowers). **Since the merge with Luis's work (D73):** the two photos linked (D66, D67) come first; photo 2's own reading is used only when they cannot be linked, and then "middle" only for TEST cases, since it assumes the element fills the photo (a real upload stays unknown, counted as an end zone). `app.core.where_on_element`.
- The crack model was trained on close-ups (D48); on wide photos it can mark frames and edges. Photo 2's crack is used for place and extent only, never for the width, and the reason says it was estimated from photo 2.

## Spalling model (`src/element/spalling.py`, `src/experiments/spalling_model.py`)
ConvNeXt-tiny trained on PHI-Net task 3 [53][55] (spalling / no spalling; 6,898 training, 837 test photos; CC BY-NC-SA 4.0), as the scene model (D54): 8 rounds, calibrated on validation. The spalling threshold is chosen on the validation photos so that at least 90 % of spalling photos are caught (illustrative): a missed spalling can lower a level; a false one sends the case up, to an engineer. Reported on the test photos once: share caught, false alarms.

## The level, from both photos (`auto_triage`, D55 extended)
- Spalling or exposed bar seen on photo 2: the type is **spalling with exposed bar** (risk matrix), whatever the line's direction.
- Otherwise the type from the written pattern rules on photo 1's crack shape and photo 2's element, as before.
- Position: from photo 2 (end zone or middle) instead of always unknown.
- Width: from photo 1 (bank card). No card: Expert needed, as before.
- The reason names what came from which photo.

## The 3D model (D57 extended)
With photo 2's crack: the crack is drawn **where photo 2 shows it and at its length next to the element** (u, v times the element's assumed width and height); the width is photo 1's. Without it: as before, centred, at photo 1's real length. The label says which.

## Limits
- The demo's photo 1 is synthetic (no public photo has a bank card); its photo 2 is a real PHI-Net damage photo: the same kind of damage, not the same crack (TEST, D59).
- The element is assumed to fill photo 2; a photo of half a column misplaces the crack.
- PHI-Net is post-earthquake damage, not slow deterioration.

## Tests
`tests/test_photo2.py`: placement (u, v, extent), end zone or middle per element, no crack on photo 2 means unknown, the spalling rule raises the type, the reason names both photos, the 3D model uses photo 2's place and proportion, never "safe".

## Real test cases from the dataset, and both photos on the wall (11 Nov, D62)
- **Real cases.** The simulated residents (D59) now use real photos only: photo 2 is a PHI-Net task 6 test photo of a damaged element; photo 1 is a close-up of **the same crack**, cut from that photo around the crack the crack model finds and enlarged. Same crack, same element: the 3D drawing, the type and the level follow one real crack.
- **Width estimated, never measured, test cases only.** These photos have no bank card. For TEST cases only, the width and length are estimated from photo 2: the element is assumed to fill the photo (the same assumption as its place, D61) and to have its typical size (`src/gh3d/layout.py`: column and wall 3000 mm high, beam 4000 mm long), which gives millimetres per pixel; the crack's width in pixels on photo 2 times that is the estimate, with an uncertainty of one pixel (coarse: about 13 mm per pixel on a 224 px column photo). Everything carrying it says "estimated from the assumed element size, not measured". A real upload without a card is unchanged: no width, Expert needed.
- **Both photos.** The phone page, the wall's gallery and the result page show photo 1 (the close-up) and photo 2 (the whole element), each with the crack in red.
- **The 3D crack** follows photo 2's crack line point by point (its zigzag and turns); the red frame keeps a small margin.
