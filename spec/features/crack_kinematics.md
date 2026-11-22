# Crack kinematics: opening and sliding (D99)

## Why
Jerome, 22 Nov: the app must read cracks the way an engineer does. E. Corres Sojo's thesis [46] (ch. 5.1, 5.2) reads a crack by its **kinematics**: how far the two faces moved apart (**opening**, across the crack) and along each other (**sliding**, along the crack). Opening alone is the width we already measure; sliding together with opening is what an engineer looks at to judge which mechanism is active (for example shear). The thesis computes it from the crack photo with the method of Pantoja-Rosero et al. 2022 [48].

## What it does
1. **The principle [48].** The two faces of a crack were one surface before it opened: the bumps of one face match the hollows of the other. Moving one face back onto the other gives the displacement of the crack at that point. Its part across the crack is the opening, its part along the crack the sliding.
2. **Our implementation (`src/measure/kinematics.py`), own code.** The EPFL code is GPL-3.0 [111] and this repository is MIT, so it is not copied; the method is re-implemented from the paper and checked on its published data [110]. At stations every 12 px along each traced crack, in a window of about 6 crack widths each side: each face is read as a profile along the crack (in the app, from the edges the width measurement already found in the photo, `src/measure/crack.py`; on the published masks, from the mask). The shift along the crack that best lines the bumps of one face up with the hollows of the other (correlation at least 0.6) is the sliding; the gap left between the faces is the opening. Translation only (no rotation): an assumption over a short window.
3. **Output per crack:** opening and sliding at each station, their medians and largest values, the share of sliding (sliding / opening), and a one-line reading: "mostly opening" (sliding under 0.3 x opening), "opening and sliding" (0.3 to 1), "mostly sliding" (over 1). Thresholds illustrative until calibrated. In mm with a bank card, else in pixels.
4. **Shown only.** Nothing here changes a level (rule 5). One photo shows the displacement since the crack formed, not a movement over time.

## In the app
- Crack viewer: small arrows at the stations (across the crack: opening; along it: sliding); chips `OPENING` and `SLIDING`; the reading line.
- Result: a "Based on" panel naming the method [48] and the thesis [46].
- Grasshopper (`src/gh3d/`): the arrows on the 3D crack.

## Read like an engineer (all shown only)
- **Movement:** the chips `OPENING`, `SLIDING`, `MOVEMENT` and the arrows in the crack viewer (button ⇄); the main crack's values in the chips, every crack's arrows on the photo. In the 3D model (Grasshopper and the wall's 3D view) the same arrows, enlarged to be seen, with the values in the label.
- **The error of every width:** a grey band ± 1 pixel on the "width along the crack" chart (`WIDTH_UNCERTAINTY_PX`, [46] ch. 5.2: photo widths matched a crack microscope within one pixel at most points on a 50-year-old crack; dust and worn edges make it larger).
- **Photo advice** on the phone page, from [46] ch. 5.2: phone flat to the wall, wipe the dust off the crack, on a shiny wall try without flash.
- **Growth since an earlier photo** (`src/measure/growth.py`): the resident may give the result number of an earlier photo of the same crack. Both must be measured with a bank card (mm). Grown only when the widest width grew by more than the two photos' errors combined (root of the sum of squares, an assumption) ([46] ch. 5.3, D23). Narrower by more than the error: "more likely a different spot, light or distance", never "closing".
- **"Read like an engineer" panel** in the result window (`src/app/engineer.py`): how the crack moved, how sure the width is, a crack along a bar weakens the bar's grip (bond, [46] ch. 2 and 4), the growth, and what it rests on; the sources by name. It says that the level does not come from this work.

## Not wired to the level (open decision)
The written matrix already has two signs this work can measure: `growing` (growth beyond the measurement uncertainty, D23) and `offset` (one side displaced, measured as sliding beyond its uncertainty). Both stay unset by the app until Jerome decides (D99); today only a person can report them.

## Limits
- A clean crack with straight faces has no bumps to match: sliding is then not readable and is shown as "not readable", never as zero.
- Dust and eroded crack lips (old cracks) make the faces less sharp [46] (ch. 5.2, Chillon): larger error.
- Too thin a crack (under about 3 px) has no measurable faces: not read.

## Evidence (`docs/experiments/2026-11-22_kinematics.md`)
On the 56 distinct toy cracks of [110] with known movements: opening read on 56, sliding on 53; sliding within 0.85 px on average; opening 1.4 px too large on average (the drawn masks include their border pixels: the same order as the ±1 px of [46]); "mostly opening" against "sliding too" right on 50 of 53.

## Tests
`tests/test_kinematics.py`: a jagged crack opened by a known vector gives back its opening and sliding; a pure opening gives sliding near zero; a straight crack gives "not readable"; units follow the scale. `tests/test_engineer_reading.py`: growth only beyond the error, needs a card both times, a narrower crack is not called closed; the panel reads the movement, names its sources and never says safe; faces from the measured edges give the same movement; the viewer's arrows stay in crack order. Evaluation on the published dataset: `src/experiments/kinematics_eval.py`, note in `docs/experiments/`.
