# Stronger crack tracing: two thresholds, flip averaging and SAM 3.1 (21 Nov 2026, D91)

Spec: `spec/features/sam_vision.md`. Scripts: `src/experiments/sam_vision_eval.py`, `src/experiments/sam_extend_eval.py`. Scores: `data/results/sam_vision/` (local). Özgenel tiles [49]: validation 319 with a crack and 353 without; test 281 and 307, never used to choose anything.

## Why
On a real survey photo the crack model traced one branch of a network; its confidence on the other branches was near zero, so no threshold could recover them.

## Two thresholds and flip averaging (validation tiles)
| Variant | Crack pixels traced (Özgenel) | Missed (Özgenel) | False alarms (Özgenel) | Missed (public) |
|---|---|---|---|---|
| One threshold 0.5 (before) | 83.8 % | 19 | 3 | 29 |
| Two thresholds 0.5 / 0.25 | 85.7 % | 18 | 3 | 27 |
| Flip averaging, one threshold | 83.5 % | 21 | 1 | 33 |

Kept: two thresholds. Left off: flip averaging (more missed cracks, the key error).

## SAM 3.1 asked for "crack" (test tiles)
| | Missed | False alarms | Crack pixels traced |
|---|---|---|---|
| Crack model v4 | 19 | 5 | 81.6 % |
| SAM alone, score 0.3 | 17 | 64 | 75.4 % |
| SAM alone, score 0.5 | 18 | 35 | 73.8 % |
| Crack model extended by SAM, score 0.4 | 19 | 5 | 84.4 % |

SAM alone draws cracks on 7 to 13 times more crack-free tiles: it never decides. As an extension (only SAM lines touching the model's crack) it traces more of each crack with the same decisions; pixels more than 6 px off the true crack rise from 1.4 % to 2.6 % (SAM's lines are thicker). Kept at score 0.4 (validation: 85.7 % to 88.0 %).

## Spalling as an area: not reliable (PHI-Net task 3 test, 310 photos with spalling, 527 without)
| Prompt, score | Lit with spalling | Lit without |
|---|---|---|
| "exposed concrete", 0.5 | 199 (64 %) | 266 (50 %) |
| "broken concrete", 0.5 | 161 (52 %) | 116 (22 %) |
| "spalled concrete", 0.5 | 103 (33 %) | 125 (24 %) |
| "rebar", 0.3 | 9 (3 %) | 11 (2 %) |

None separates spalling well enough to show; "broken concrete" is the only one with a clear difference. PHI-Net photos are small (224 px) and mostly whole elements, not close-ups: a fairer test needs close-up photos of spalling.

## On real photos (by eye, local only)
On the team's field photo the extended crack follows the top end and both forks at the bottom that the model missed; on a survey photo it follows both left branches of the network.
