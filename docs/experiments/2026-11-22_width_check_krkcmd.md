# Our width rule against 14,198 hand-measured crack widths (D108), 22 Nov 2026

## Question
How close is the app's width rule to a person measuring the same crack? Until now the ±1 pixel came from E. Corres Sojo's tests [46], not ours.

## Data
krkCMd [113] (CC BY 4.0): brightness profiles across cracks in lab concrete specimens, scanned at 6400 dpi (0.004 mm per sample), each with the width an operator read on it. 19,098 rows; 14,198 with a crack (the others are healed, width 0). Hand widths: median 0.175 mm, 10 % under 0.052 mm, 10 % over 0.389 mm: thin cracks, our hardest range.

## Method
`src/experiments/width_check_krkcmd.py`: the app's rule (`src/measure/crack.py`, grey case): the darkest point, the wall's brightness on each side, the edges where the brightness is half-way back (half-depth). In the app the wall level is read just outside the crack model's mask; here, without a mask, from the brighter half of the profile on each side (assumption). Then the same profiles averaged down to phone-like pixels (0.02, 0.05, 0.1 mm), with a random pixel offset, and measured again.

## Results
| Pixel size | Read | Mean error | Bias | Within 10 % of the hand width |
|---|---|---|---|---|
| 0.004 mm (the scan) | 14,165 | 0.062 mm | +0.060 mm | 18 % |
| 0.02 mm | 14,162 | 0.065 mm | +0.064 mm | 14 % |
| 0.05 mm (a phone at about 30 cm) | 14,138 | 0.060 mm | +0.059 mm | 23 % |
| 0.1 mm (a phone at about 60 cm) | 14,043 | 0.084 mm | +0.083 mm | 14 % |
| The authors' deep-learning metasensor | all | 0.015 mm | −0.005 mm | 70 % |
| The authors' analytic edge detector | all | 0.039 mm | −0.038 mm | 26 % |

## What it means
- Our rule reads thin cracks about 0.06 mm too wide, at every pixel size: a systematic offset, not noise. The half-depth edge sits further out than where the operator puts the crack's edge. For a 0.18 mm crack that is about a third too wide.
- It errs on the safe side (wider, so never a lower band), but it can push a crack near 0.3 mm or 1 mm into the next band.
- At 0.1 mm per pixel most cracks are under 3 pixels across, and the error grows, as [46] says.
- Tried the same day (D109): a deeper edge level (0.7) halves the error here (0.034 mm against 0.070 mm at 0.05 mm per pixel, on held-out specimens) but reads drawn 1 to 2 mm cracks of known width 0.12 to 0.18 mm too narrow; not applied, since too narrow could lower a level.
- Before: choose the edge level (half-depth or deeper) on one half of the rows and check it on the other, then apply it in the app. A lab scan is not a phone photo: the offset must be checked again on our own photos with a crack gauge.
