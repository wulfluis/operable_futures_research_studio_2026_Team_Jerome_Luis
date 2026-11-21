# Stronger crack tracing: two thresholds and SAM 3.1 as an extension (D91)

## Why
Jerome, 21 Nov, on a real survey photo: the crack viewer traced one branch of a crack network and none of the spalled patch; he asked for "a very powerful vision". The crack model's confidence map showed the left branches at almost zero, so no threshold could recover them.

## What it does
1. **Two thresholds (hysteresis, as in edge detection).** A pixel is crack above 0.5, and a fainter pixel above 0.25 is crack too when it touches such a group (`segmentation.decide.grown`). Faint continuations are kept; faint spots on their own are not. The confidence is the confident part's, so faint growth does not lower it.
2. **SAM 3.1 extends the crack (`src/segmentation/sam_extend.py`).** SAM 3.1, asked for "crack" (score 0.4 or more), traces lines on the photo; only the lines that touch the crack model's crack (within 2 % of the long side) are added. The crack model still decides whether there is a crack, so SAM never turns a crack-free photo into "Crack found". SAM's lines are thicker than the crack: they say where it runs; the width is still read from the photo.
3. One SAM 3.1 copy serves the element photo (D70) and the close-up, one call at a time.

Flip averaging (the model also reading the photo flipped) was tried and left off: on the validation tiles it missed more cracks (21 against 19 on Özgenel, 33 against 29 on public tiles).

## Evidence (`docs/experiments/2026-11-21_sam_vision.md`)
On the Özgenel test tiles (281 with a crack, 307 without), against the crack model v4 alone:

| | Missed cracks | False alarms | Crack pixels traced | Pixels off the crack |
|---|---|---|---|---|
| Crack model v4 | 19 | 5 | 81.6 % | 1.4 % |
| SAM 3.1 alone (score 0.5) | 18 | 35 | 73.8 % | |
| Crack model, extended by SAM (score 0.4) | 19 | 5 | 84.4 % | 2.6 % |

"Off the crack": more than 6 px from the true crack (SAM's lines are thick).

## Since D92 (21 Nov, the same day): every SAM line, once the model found a crack
On a real survey photo the branches of the network met at a spalled patch, not at the line the model found, so the connected-only rule dropped them and nothing visible changed. Now, once the crack model has found a crack, every SAM crack line (score 0.4 or more) counts, minus the long straight edges the model's own rule ignores (D51: a door frame, a wall-ceiling line). The model still decides whether there is a crack, so missed cracks and false alarms do not change. Test tiles (photos where the model found a crack): 85.2 % of the crack traced (connected-only 84.9 %, model 82.0 %); pixels more than 6 px off the true crack 5.2 % (2.6 %, 1.4 %): more extra lines, each shown in its own colour so a wrong one is seen.

## Not done
- **Spalling as an area.** SAM asked for "exposed concrete" lit 64 % of PHI-Net photos with spalling and 50 % without: no better than a guess, not shown. Other prompts: see the experiment note.
- The 3D model still draws the main crack only.

## Tests
`tests/test_sam_extend.py`: a connected SAM line is added, a far one is not; no model crack means SAM adds nothing; a missing or failing SAM leaves the model's crack; faint pixels count only when joined to a confident crack.
