# Feature: crack type and location in the app, and the first risk category (`src/app/`, `src/element/`)

Status: first version, 11 Nov 2026, decisions D53 and D55 (the category is given automatically; no one confirms on site). Local branch, not pushed. Thresholds illustrative until calibrated.

## Purpose
Until now the app said only whether there is a crack and, with a bank card, how wide it is. The risk matrix (`src/triage/matrix.py`) also needs **where** the crack is (the element and the position on it) and **what type** it is (the pattern). This feature adds both, so the app can give Category 1, 2 or 3 for the first time, through the written matrix (rule 5).

## Who answers what (D25, D55)
| Question | Resident | Answered by |
|---|---|---|
| Element (column, beam, wall) | never asked | element model on the second photo (D52), with "doubt means column" |
| Crack type (pattern) | never asked | written rules from the crack's shape and that element (D43); with a bank card the shape is measured on the wall, without one it is read in the photo's pixels (D57) |
| Position on the element | never asked | not known from the photos: counts as an end zone (never lowers a level) |
| Level | - | the written risk matrix, by itself (rule 5, D55) |

## Resident flow (phone page)
1. Crack close-up, as before; the scene check first (D54): a photo that is not a close-up is a Retake.
2. When the statement is "Crack found" or "Expert needed": **"Where is the crack?"** asks for a second photo of the whole column, beam or wall. A close-up is refused with "step back".
3. The element model reads it; the written rules give the type; the matrix gives the level at once.

## The level, given automatically (`auto_triage` in `src/app/core.py`)
- Element: the model's answer mapped to the matrix (column, beam; wall counts as load-bearing, since the photo cannot tell; "others" and "unsure" give none). No slab class.
- Type: only a suggestion the rules call **clear** is used; an unclear one gives none.
- Confidence: the lower of the crack model's and the element model's probability.
- Position: unknown, so an end zone.
- Width: the widest width with its uncertainty from the bank card; none without a card.
- Without a card the type is still read, from the U-Net's crack line in pixels (D57); the level stays Expert needed, since the matrix needs the width, and the reason says the width is missing rather than the type.
- Any missing piece gives Expert needed through the matrix, with a sentence saying why. Category 3 and Expert needed are flagged for the engineer (rule 4).

## Live wall
Keeps the four statement buttons. Each photo carries its widest width and, once known, its element and level; the result's own page shows element (with source: model or field worker), position, type (suggested or confirmed) and the level with its reason. The element photo and the level update the wall entry.

## Photos (rule 9)
The element photo is treated as the crack photo: in memory, never saved, only pixels read, consent by sending. Only a small copy of the crack overlay goes to the wall, as before; the element photo is not shown on the wall.

## Limits
- The element model was tested on PHI-Net (mixed photos, no informal homes): one column in twenty confidently called something else; it only suggests.
- The element photo is squeezed to 224 x 224 px, as the PHI-Net images (assumption: PHI-Net resized whole photos).
- The pattern rules assume an upright phone and are illustrative (D43).
- Without the model file `models/element_convnext_tiny_phinet6_v2.pt`, the element is unknown and the field worker chooses.

## Tests
`tests/test_element_predict.py`: model loading, calibration, a missing model file, the mapping. `tests/test_app_classification.py`: element and type suggestions, no category before confirmation, the category after confirmation through the matrix, unknown position counts as an end zone, no width gives Expert needed, the wall entry updated, never "safe".
