# Feature: crack measurement with a reference object (`src/measure/`)

Status: version 0.2 in code (4 Nov 2026), tested on synthetic images only. Pipeline step P3 "crack reading", the width part, and step P2 "scale and angle" for the bank card (`spec/features/pipeline.md`). Decisions D41, D42. Every threshold is illustrative until tested on the pilot photos.

## What it does
From one crack close-up (scale P) that also shows a reference object of known size, it returns the crack **width in mm with its uncertainty**, the **length in mm**, and warnings. It is a measurement and never a safety statement. Its `matrix_input` (width and uncertainty) feeds `triage.matrix.assess()`.

## The bank card, found automatically (D42, `card.py`)
Any card in the ID-1 size works (bank, ID, access, loyalty card): only its outline counts. When no closed outline is found (field case 4 Nov: a light access card on a light wall, a finger across one edge), a second search joins straight edge pieces into four lines in two near-parallel pairs. Each side must lie at least 45 % on edges and the four together 70 % (illustrative), and the shape must fit a card under perspective as before.
Choosing among card-shaped candidates (field case 4 Nov, a card shown too large): a shape touching the photo's edge is rejected (a shadow joined to it); a shape lying wholly inside another card-shaped one is print on the card (a letter, a logo box); of the rest, those whose sides lie well on edges (score at least 0.6, or within 0.25 of the best) are kept and the largest wins. Before, the largest shape won. Limits illustrative.
Both searches run on every photo (before, the straight-edge search ran only when the first found nothing, so a closed outline over print stopped it). A candidate must also be one object: along all four sides the inside differs from the outside the same way (at least 1 grey level, illustrative); a shape over print or over a crack fails this on at least one side. Checked on phone screenshots only: screenshots carry the app's own drawn outline, which the search then finds, so original photos are needed for a real check.
The default reference is a bank-card-sized card (ISO/IEC 7810 ID-1, 85.60 x 53.98 mm: bank, ID, loyalty or an expired card), flat on the wall next to the crack, with the card number and name covered.
1. **Find.** Four-sided shapes from OpenCV contours (edges and brightness or colour thresholds) on a copy with a long side of 1000 px.
2. **Is it a card?** For each shape the camera pose is solved (planar PnP, IPPE). The shape counts as a card only if a card of this size, at most 35 degrees off straight, reprojects onto its corners within 2 % of the card's diagonal. A square tile or a long strip does not pass. The largest passing shape is taken.
3. **Corners, sub-pixel.** On the full photo, the strongest edge across each side is found over the middle 70 % of the side (the card's corners are rounded). One line is fitted per side, and the corners are where the lines cross.
4. **Tilt** = the angle between the card's normal and the camera axis, from the pose, with its left-right and up-down parts.

| Tilt (illustrative) | Status | Effect |
|---|---|---|
| up to 10 degrees | straight | none |
| 10 to 20 degrees | tilted | warning; the width is still corrected by the card |
| over 20 degrees | retake | Retake; the width is not passed to the matrix |

Why a retake above 20 degrees although the card corrects the perspective: a tilted view hides part of the crack behind its own lip and spreads fewer pixels across it, and [46] asks for the camera to be perpendicular.
Assumptions: the photo is not cropped (the camera centre is the image centre), and the focal length is about 0.75 x the long side (a phone main camera, about 26 mm equivalent). Zoom or a crop makes the tilt less exact, not the width (the width comes from the card's corners).
If no card is found: Retake ("bank card not found"). The app and the command line also accept corners given by hand.

## One photo in, one result out (`analyse.py`)
Quality gate (sharpness, brightness; `src/quality/gate.py`) -> card and tilt -> crack mask in the search box (default: the whole photo; card excluded; only the longest piece by default) -> width and length -> outcome:
- **Retake:** the gate fails, no card, tilt over 20 degrees, or fewer than 3 px across the crack.
- **Width not measured:** no crack found; the case goes to an engineer (D3).
- **Measured:** width, uncertainty and length, plus `matrix_input` for `triage.matrix.assess()` and the width band of the upper end. It is never a level: the level needs the pattern and the element.

## App (`app.py`, Streamlit, local only)
`streamlit run src/measure/app.py`. Upload a JPEG or PNG, or use the synthetic demo photo (width and tilt sliders). It shows the outcome, the width (widest and median, with uncertainty), the length, the pixels across the crack, the camera tilt, warnings, the overlay, and offers downloads (JSON, PNG). Search box sliders and corners by hand cover the cases where the automatic steps fail. Photos stay in memory. "Save result" writes only the JSON and the overlay to `data/measurements/` (ignored by git). Rule 9: consent first; no faces, addresses or card numbers in the frame.

## Other reference objects (command line)
| Preset | Object | Points | Corrects tilt |
|---|---|---|---|
| `card` | bank or ID card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm | found automatically (or 4 corners) | yes (homography), with the tilt check |
| `a4`, `a6` | ISO 216 sheet | 4 corners | yes |
| `square50` | 50 mm square on the scale card (proposal: add it to the card) | 4 corners | yes |
| `bar50` | 50 mm scale bar on the scale card | 2 ends | no |
| `euro1` | 1 euro coin, 23.25 mm, backup (D24) | 2 points across | no |
| own | `--ref-size WxH` or `--ref-length L` | 4 or 2 | 4: yes |

A 4-corner reference gives a homography from image pixels to mm in the plane of the reference, so a camera that is not perpendicular to the wall is corrected. A 2-point reference gives one scale and assumes the camera is perpendicular. Either way, the crack must lie **in the plane of the reference, next to it** [46] ch. 5.2. Points are clicked twice: once roughly, then precisely in a zoomed view.

## Method
1. **Crack pixels.** A mask from a file (`--mask`, from the U-Net later or drawn by hand), or a classical finder until the U-Net exists: black-hat (pixels darker than the wall around them) inside a search box, an Otsu threshold, and only thin or elongated pieces kept. The reference and a band around it are excluded.
2. **Centre line.** Zhang-Suen thinning, with short side branches pruned.
3. **Width**, as in [46] ch. 5.2 and [48]: at every centre-line point, perpendicular to the local direction. The edges lie at half the depth between the wall and the crack, read from the grey-level profile to sub-pixel precision. Where the contrast is too low, the mask edges are used instead. Each end of the width is mapped to mm on its own, so perspective is included. Points near ends and junctions are skipped. Widths are smoothed over about 3 px of the centre line, and the **widest point** is reported, together with the median.
4. **Uncertainty** = ±1 pixel at the local pixel size [46], plus the scale error from the clicks (±1 px per point, assumption).
5. **Length.** The longest path along the centre line (main length) and the sum of all branches (total), each smoothed so that pixel staircases do not add length.

## Shape, for the pattern rules (D43)
Besides width and length, the measurement reports the crack's form: the **angle of the main path to vertical** (0 = vertical, 90 = horizontal, on the wall plane; "vertical" is the image's up-down carried onto the wall, so the phone must be upright), the **straightness** (end-to-end distance / main length), the **coherence** of the local directions along the whole centre line (1 = one direction, 0 = none), the number of **closed cells**, and the **branch ratio** (all branches / main length). `src/triage/pattern_rules.py` reads them (`spec/features/pattern_rules.md`).

## Rules (illustrative)
| Check | Effect |
|---|---|
| Under 3 px across the crack [46] | Warning "retake closer"; `matrix_input` is None, so the width is not used (rule 2, D3) |
| 3 to 6 px across | Warning: about ±30 % [46] |
| No crack found / too small | Width not measured; `matrix_input` is None |
| Crack more than 150 mm from a 4-corner reference | Warning: put the reference next to the crack |
| Crack in several pieces | Warning: the main length covers the longest piece only |

## Tests (`tests/test_measure.py`, `tests/test_card.py`)
Homography against a known mapping; card scale in any click order; refusal of bad references; straight cracks at three angles (width 0.8 mm, length 30 mm); a synthetic photo with a tilted camera and a bank card (width 1.0 mm, length 135.8 mm: measured 1.03 mm widest, 1.00 mm median, 135.8 mm); a crack under 3 px is not trusted; no crack found; the CLI writes JSON and an overlay without the words "safe", "clearance" or "permit". Card: found at tilts 0 to 26 degrees with corners within 1 px and the tilt within 1.5 degrees of the camera model; tilt status at 0, 15 and 28 degrees; no card in a plain wall; a square and a 3:1 strip are not taken for a card; the whole analysis without a search box (1.0 mm, 135.8 mm); 30 degrees is a Retake; the app runs on the demo photo. A synthetic 12 MP photo is analysed in about 2 s.

## Not yet
- Accuracy on real photos: photograph the width lines of the scale card (0.1 to 3 mm), measure them with this code and compare (the first test against a known answer, 12 Nov). The ±1 px is adopted from [46], not yet shown for our code.
- The card search on real photos: patterned cards, cards on a similar-coloured wall, fingers over the edges, other card-shaped rectangles (switches, labels). To test on the pilot photos.
- The focal-length assumption on zoomed photos.
- Sliding (offset) next to opening [48]; averaging a burst of photos [46].
- The classical finder struggles with stains, joints, shadows and textured plaster: expect wrong pieces on real walls and check the overlay. It is replaced by the U-Net.
