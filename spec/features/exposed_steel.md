# Exposed steel on the close-up (D107)

## Why
Jerome, 22 Nov, on a real case: a column whose cover has fallen off, a vertical bar and a stirrup in the open. The app traced "cracks" along the stones and the stirrup's shadow, put the widest point on the stirrup, and did not say what mattered: the steel is exposed. The spalling rule of D93 needs sound wall around the broken area, and here the whole close-up is broken concrete.

## What it does
1. **SAM 3.1 looks for exposed steel on photo 1** (`src/element/steel.py`), asked for "rebar", "exposed rebar" and "rusty steel bar". Seen when one answer scores at least `MIN_SCORE` 0.5 and covers at least `MIN_SHARE` 0.2 % of the photo (illustrative, chosen by eye on 100 real close-ups, see Evidence). "Steel bar" is not asked: it also finds railings and gates.
2. **When steel is seen:**
   - the reason says so in one sentence, and the crack viewer outlines the steel (a "Steel?" chip, its own colour, like the other defects);
   - the traced crack lines may follow the steel and the broken concrete, so their widths are not used: the width is reported as not measured, and the level gets no width;
   - the type is "spalling with exposed bar" (the matrix pattern `spalling_exposed_bar`), as when the element photo shows spalling (D61).
3. **The level** comes from the written matrix as always (rule 5). Without a usable width it is "Expert needed": an engineer reviews it, and the reason names the exposed steel. Exposed steel never lowers a level.
4. Without SAM 3.1 on the laptop nothing changes and nothing is claimed.

## Evidence (`src/experiments/exposed_steel_eval.py`, 22 Nov)
On the 100 real close-ups (field pairs, survey cases, EPFL buildings), reviewed by eye: about 15 show exposed bars clearly; the rule finds them, the column of Jerome's example included, and raises no false alarm on the others (painted stripes, a drain pipe, a window grille and a gate scored under 0.5 for the three prompts). It misses about 8: bars under rubble, thin or in deep shadow. One reviewer, no second opinion: an assumption to check on more photos. The spalling model of D61 was also run on the close-ups and not used: it scores ordinary peeled plaster as high as broken concrete.

## Limits
- A bar seen through a narrow crack, or under dust, is often missed.
- A metal frame, pipe or grille very close to the wall may still be taken for a bar: shown as "Steel?", never as certain.
- Shown as "possible" until an engineer has reviewed more cases.

## Tests
`tests/test_exposed_steel.py`: a high-scoring bar answer is seen, a low one or a speck is not, "steel bar" answers are ignored, no SAM means not known; when steel is seen the width is not used and the type is spalling with exposed bar.
