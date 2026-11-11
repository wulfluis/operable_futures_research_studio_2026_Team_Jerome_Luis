# Which crack widths a phone can see and measure at which distance (11 Nov 2026)

Midterm action item 7 (R1, R2): "Is a scale card realistic? Which crack sizes are visible at which field of view? Focus on structurally relevant cracks." Calculation from camera geometry (`spec/features/capture_visibility.md`, `src/measure/visibility.py`, 5 tests). Camera values are typical phone values (assumptions); real photos lose detail to focus, shake, compression and light, so these are best cases until our width check photos confirm them.

## What it shows
1. **The cracks that matter structurally can be measured from a few metres.** With a normal 12 MP main camera, a 3 mm crack (our top width band; FEMA 306's 1/8 in. is 3.2 mm [93]) is measurable from about 3 m, and a 2 mm crack (JBDPA class IV starts above 2 mm [98]) from about 2 m. This answers R2's "limited photo scale": the field of view limits the fine cracks, not the wide, structurally relevant ones.
2. **Fine cracks need a close or zoomed photo.** At about 1 m, the 12 MP main camera *sees* cracks from about 0.3 mm but *measures* them only from about 1 mm (3 pixels across, D24). To measure the 0.3 mm band edge it needs about 0.3 m, or about 1 m with a 3x tele lens. The app's hint "go nearer for a fine crack" stays right; a hint "use the 2x or 3x lens" would help.
3. **The bank card is findable up to about 5 m** with the main camera, and to 10 m and beyond with a 2x or 3x lens (the card bridge searches down to 1.2 % of the photo's side, D59). This matches the element photos at about 3 m that linked in the field test of 11 Nov.
4. **Check against literature:** Seo et al. 2021 [105] told 0.3, 0.5 and 1 mm apart at 2 m only with a 64 MP camera and a 2x lens; the same calculation gives about 2 pixels across 0.3 mm in that set-up, at the edge of reading, as they found.

## On "is a card realistic" (R1)
The card is needed for millimetres, not for finding the crack or its type: the type is read without a card (D57) and the position through the card or key points (D58, D59). Since the wide cracks are the ones that matter, a later option is to take the scale from something else of known size in the element photo (a door leaf, a standard brick or block, a tile); not built, and each needs its size to be known.

## Limits
Camera square to the wall; tilt makes it worse. "Seen at 1 px" is an assumption for a dark crack on a light wall. The 48 MP mode's real sharpness is below its pixel count. Validate with the width check photos (`docs/width_gauge/`, `src/measure/width_check.py`).

## Table
| Camera | Distance | mm per px | Seen from (about 1 px) | Measurable from (3 px) | Good from (6 px) | Card length | Card bridge finds it | Finest reference width measurable |
|---|---|---|---|---|---|---|---|---|
| Main 12 MP | 0.3 m | 0.099 | 0.10 mm | 0.30 mm | 0.59 mm | 864 px | yes | 0.3 mm (matrix band) |
| Main 12 MP | 0.5 m | 0.165 | 0.17 mm | 0.50 mm | 0.99 mm | 518 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Main 12 MP | 1 m | 0.330 | 0.33 mm | 0.99 mm | 1.98 mm | 259 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Main 12 MP | 2 m | 0.660 | 0.66 mm | 1.98 mm | 3.96 mm | 130 px | yes | 2 mm (JBDPA III/IV) |
| Main 12 MP | 3 m | 0.991 | 0.99 mm | 2.97 mm | 5.94 mm | 86 px | yes | 3 mm (matrix band) |
| Main 12 MP | 5 m | 1.651 | 1.65 mm | 4.95 mm | 9.91 mm | 52 px | yes | none of the references |
| Main 12 MP | 10 m | 3.302 | 3.30 mm | 9.91 mm | 19.81 mm | 26 px | no | none of the references |
| Main 48 MP | 0.3 m | 0.050 | 0.05 mm | 0.15 mm | 0.30 mm | 1728 px | yes | 0.2 mm (JBDPA class I/II) |
| Main 48 MP | 0.5 m | 0.083 | 0.08 mm | 0.25 mm | 0.50 mm | 1037 px | yes | 0.3 mm (matrix band) |
| Main 48 MP | 1 m | 0.165 | 0.17 mm | 0.50 mm | 0.99 mm | 518 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Main 48 MP | 2 m | 0.330 | 0.33 mm | 0.99 mm | 1.98 mm | 259 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Main 48 MP | 3 m | 0.495 | 0.50 mm | 1.49 mm | 2.97 mm | 173 px | yes | 1.6 mm (FEMA 306 1/16 in.) |
| Main 48 MP | 5 m | 0.826 | 0.83 mm | 2.48 mm | 4.95 mm | 104 px | yes | 3 mm (matrix band) |
| Main 48 MP | 10 m | 1.651 | 1.65 mm | 4.95 mm | 9.91 mm | 52 px | no | none of the references |
| Tele 2x 12 MP | 0.3 m | 0.050 | 0.05 mm | 0.15 mm | 0.30 mm | 1728 px | yes | 0.2 mm (JBDPA class I/II) |
| Tele 2x 12 MP | 0.5 m | 0.083 | 0.08 mm | 0.25 mm | 0.50 mm | 1037 px | yes | 0.3 mm (matrix band) |
| Tele 2x 12 MP | 1 m | 0.165 | 0.17 mm | 0.50 mm | 0.99 mm | 518 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Tele 2x 12 MP | 2 m | 0.330 | 0.33 mm | 0.99 mm | 1.98 mm | 259 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Tele 2x 12 MP | 3 m | 0.495 | 0.50 mm | 1.49 mm | 2.97 mm | 173 px | yes | 1.6 mm (FEMA 306 1/16 in.) |
| Tele 2x 12 MP | 5 m | 0.826 | 0.83 mm | 2.48 mm | 4.95 mm | 104 px | yes | 3 mm (matrix band) |
| Tele 2x 12 MP | 10 m | 1.651 | 1.65 mm | 4.95 mm | 9.91 mm | 52 px | yes | none of the references |
| Tele 3x 12 MP | 0.3 m | 0.033 | 0.03 mm | 0.10 mm | 0.20 mm | 2559 px | yes | 0.2 mm (JBDPA class I/II) |
| Tele 3x 12 MP | 0.5 m | 0.056 | 0.06 mm | 0.17 mm | 0.33 mm | 1535 px | yes | 0.2 mm (JBDPA class I/II) |
| Tele 3x 12 MP | 1 m | 0.112 | 0.11 mm | 0.33 mm | 0.67 mm | 768 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Tele 3x 12 MP | 2 m | 0.223 | 0.22 mm | 0.67 mm | 1.34 mm | 384 px | yes | 1 mm (matrix band, JBDPA II/III) |
| Tele 3x 12 MP | 3 m | 0.335 | 0.33 mm | 1.00 mm | 2.01 mm | 256 px | yes | 1.6 mm (FEMA 306 1/16 in.) |
| Tele 3x 12 MP | 5 m | 0.558 | 0.56 mm | 1.67 mm | 3.35 mm | 154 px | yes | 2 mm (JBDPA III/IV) |
| Tele 3x 12 MP | 10 m | 1.115 | 1.12 mm | 3.35 mm | 6.69 mm | 77 px | yes | none of the references |

