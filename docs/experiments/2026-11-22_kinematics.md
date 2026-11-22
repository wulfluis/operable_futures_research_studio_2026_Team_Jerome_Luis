# Opening and sliding read from one photo (D99), 22 Nov 2026

## Question
Can our own implementation of the crack-kinematics method of Pantoja-Rosero et al. 2022 [48], used in E. Corres Sojo's thesis [46], read how a crack moved (opening across it, sliding along it) from a traced crack?

## Data
The published toy cracks of [48] (Zenodo 6632071 [110], CC BY 4.0): binary crack masks made by moving one side of a drawn crack by a known rigid movement (rotation, x and y shift). The dataset's folders repeat the same masks; 56 distinct cracks remain (`src/experiments/kinematics_eval.py` removes the repeats by checksum). True opening and sliding: the known shift projected across and along the crack's main direction (the rotation, at most about 3°, is ignored).

## Method
`src/measure/kinematics.py`, own code (the EPFL code is GPL-3.0 [111]): each face read as a profile along the crack in windows of about 6 crack widths; the shift that best lines one face's bumps up with the other's hollows is the sliding, the gap left the opening. A first version matched the faces' points directly (iterated nearest points): its sliding jumped between neighbouring stations and was dropped the same day.

## Results (56 cracks; median true opening 7.2 px)
| | Read | Mean error |
|---|---|---|
| Opening | 56 of 56 | 1.5 px (1.4 px too large on average) |
| Sliding | 53 of 56 | 0.85 px |
| "Mostly opening" or not | right on 50 of 53 | |

The opening is read about 1.4 px too large: the drawn masks include their border pixels on both faces. This is the order of the ±1 pixel error of [46] (ch. 5.2), and it is not corrected (one dataset is not enough to calibrate it). In the app the faces come from the edges the width measurement finds in the grey values, not from a mask.

## Limits
- Toy masks, not photos: on real old cracks, dust and worn edges blur the faces [46].
- Faces without bumps (a smooth, straight crack) give "sliding not readable", never zero.
- One photo shows the movement since the crack formed, not whether it still moves.

Shown only: nothing here changes a level (rule 5).
