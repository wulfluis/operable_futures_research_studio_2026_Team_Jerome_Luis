# Feature: which crack widths a phone can see and measure at which distance (`src/measure/visibility.py`)

Status: first version, 11 Nov 2026. Answers midterm action item 7 (R1, R2): "Is a scale card realistic? Which crack sizes are visible at which field of view? Focus on structurally relevant cracks." A planning calculation from camera geometry, not a measurement; every camera value is an assumption until checked on our own photos.

## Inputs
- **Camera** (assumption, typical phone values): 35 mm-equivalent focal length and the number of pixels along the photo's long side. The equivalent focal length refers to the 43.27 mm diagonal of a 36 x 24 mm frame; a 4:3 photo's long side then spans 34.62 mm of that frame.
  - Main camera, 12 MP: 26 mm, 4032 px.
  - Main camera, 48 MP full resolution: 26 mm, 8064 px (real sharpness is lower than the pixel count; assumption).
  - Tele 2x: 52 mm, 4032 px. Tele 3x: 77 mm, 4032 px.
- **Distance** from the phone to the wall, camera held square to it.

## Calculation
- Wall covered along the long side: `W = d x 34.62 / f_eq`. Size of one pixel on the wall: `W / px`.
- **Smallest measurable width**: 3 px across the crack (the pixel rule, D24, [46]; `MIN_PX_ACROSS` in `src/measure/crack.py`). **Good measurement** from 6 px.
- **Smallest crack likely to be seen at all**: about 1 px wide (assumption: a dark, clean crack on a light wall can show at one pixel; to check on our photos).
- **Bank card** (85.6 mm long side): its length in pixels, and whether it is at least 1.2 % of the photo's long side, the smallest card the card bridge searches for (D59, `LOOK_MIN_SHARE`).

## Width references (what is structurally relevant)
- Our risk matrix's width bands: 0.3, 1 and 3 mm (illustrative, `risk_matrix_v3.json`).
- JBDPA damage classes [98] via [97]: under 0.2 mm (I), 0.2 to 1 (II), 1 to 2 (III), over 2 mm (IV).
- FEMA 306 [93], RC wall, ductile flexure: shear cracks over 1/8 in. (3.2 mm) leave "insignificant"; other guides use 1/16 in. (1.6 mm).

## Output
A table per camera and distance: mm per pixel, smallest seen width, smallest measurable width, card length in pixels and whether the card bridge can find it, and the highest of the reference widths that can be measured. Written to `docs/experiments/2026-11-11_capture_visibility.md` by `python3 src/measure/visibility.py --markdown`.

## Check against literature
Seo, Kim, Chung (2021) [105] told 0.3, 0.5 and 1 mm apart at 2 m only with a 64 MP camera and a 2x lens. The calculation for that case (9248 px, 52 mm) gives about 0.14 mm per pixel, so 0.3 mm spans about 2 px: at the edge of what can be read, as they found.

## Tests (`tests/test_capture_visibility.py`)
The pixel size grows in proportion to distance and shrinks with focal length and pixel count; the 3 px rule matches `MIN_PX_ACROSS`; the Seo et al. case gives 0.3 mm between 1 and 3 px; the card at 1 m with a 12 MP main camera is found by the card bridge, and at 10 m it is not; the table has no word "safe".
