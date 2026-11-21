# Every crack in the photo, not only the longest (D88)

## Why
Jerome, 21 Nov, on a team field photo: the crack viewer drew only part of the crack. The crack model had found the crack in four pieces, and the measurement kept only the longest piece (`longest_piece` in `src/app/core.py`). The branch running to the door frame and the crack's top end were dropped, about a quarter of the crack pixels the model found. Dropping pieces is also a safety problem: the level uses the widest point, and the widest point may lie on a dropped piece, so the width could be reported too small (AGENTS.md rule 2: doubt never lowers a level).

## What it does
1. **Pieces become cracks.** The crack model's mask (after the written rules of `src/segmentation/decide.py`) is split into its connected pieces. Pieces smaller than `MIN_EXTENT_SHARE` of the photo's long side are left out as specks. Pieces closer than `GAP_SHARE` of the photo's long side to each other belong to the same crack (the model often leaves a small gap in a faint crack); pieces further apart are **independent cracks**. Both values are illustrative until calibrated (`src/measure/cracks.py`).
2. **Every crack is measured** with the same code as before (`src/measure/crack.py`): its length, its branches, and its width at every point.
3. **The whole photo's answer:**
   - **Width (used by the level): the widest point over all cracks**, with that crack's uncertainty and pixels across. No piece can be skipped.
   - **Length, angle and shape: the main crack**, the one with the longest main line. The type rules (`src/triage/pattern_rules.py`) read the main crack as before.
   - Total length: all cracks together. The median width: over all cracks.
   - The number of independent cracks is kept (`n_cracks`), and each crack's own numbers are kept for the viewer.
4. **Without a bank card** the same grouping is used, in the photo's pixels (no mm, rule 7).

## The crack viewer (`src/app/crack_view.py`, D83)
- Every branch of every crack is drawn, not only the main line.
- **Each independent crack has its own colour** when the photo holds more than one: pink, sky blue, amber, violet, orange, fuchsia, blue, red, in that order (never green, which reads as "safe", rule 1). A button switches between the crack colours and the width colours (violet thin to yellow wide); with one crack, the width colours are shown first.
- A "Cracks" chip gives the number of independent cracks; hovering it lists each crack's length and widest point.
- Widest and narrowest are looked for on every crack. The narrowest leaves out the two ends of every crack, where cracks thin to nothing (shown only; the level uses the widest).
- The width profile at the bottom stays the main crack's.

## Not changed
- The level's rules and thresholds; the type rules read the main crack only.
- The 3D model draws the main crack as before; the other cracks in 3D are a next step.
- A crack the model does not see at all is still missed: faint hairlines may vanish when the photo is reduced to 832 px for the crack model. Running the model on tiles of the full photo is a next step, after a speed test.

## Tests
`tests/test_all_cracks.py`: two far-apart lines are two cracks; two lines with a small gap are one crack; a speck is left out; the level's width is the widest over all cracks even when the widest is on a shorter crack, while the length and the angle are the main crack's; the viewer receives every crack with its own paths.
