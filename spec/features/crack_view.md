# Crack viewer: explore the crack (D83)

## Why
Luis, 19 Nov: the result should be something to explore, not to read. Zoom into the crack and out again, move the photo up, down, left and right with the mouse, click the narrowest point and a loupe opens; far less text; the measured numbers built into the picture.

## What it does (`src/app/crack_view.py`)
- Photo 1 (long side 2000 px, no marks) fills a dark frame. Scroll or pinch zooms around the pointer (0.6x to 60x of the whole photo), drag moves it, double-click zooms in, arrow keys move, + / − / 0 zoom and fit, Esc closes the loupe.
- The crack's centre line is drawn in the colour of its width at each point (violet thin, pink, orange, yellow wide; never green, rule 1). Two markers: **Widest** (red ring, the width the risk matrix uses) and **Narrowest** (cyan ring, shown only).
- A click on a marker, on the Max / Min chips or anywhere on the photo flies there and opens the **loupe**: a round magnifier (4x the current view, scroll on it to change), a caliper drawn across the crack between its measured edges, and the width there. Dragging the loupe moves it along the photo and the width follows (nearest measured point within about 20 px on screen; "No crack under the loupe" otherwise).
- Chips: Max (± uncertainty in the loupe), Min, Median, Length; until 20 Nov also the camera\'s angle to the wall, now only under "Measurement details" (D85) (19 Nov, Luis: the angle, not the tilt): 90° minus the tilt the bank card gives, so 90° is straight on, with the left-right and up-down parts and a small drawing of the camera at that angle; red when the photo must be retaken. The width profile along the crack runs at the bottom: hovering it marks the point on the photo, a click flies there. A scale bar follows the zoom (≈, from the median pixel size; the tilt makes it approximate).
- Buttons: the width line on/off, the model's red mask on/off.

## Numbers (rule 7)
- Every number is the measurement's own (`CrackMeasurement.profile_*`, D81; length = `length_main_mm`, the same as the reason line). Nothing is interpolated or invented.
- **mm only when measured with a bank card** (`MeasuredResult.measured`). With a card whose width is not trusted, or with no card, widths are in the photo's pixels and say "no card, no mm".
- The narrowest point leaves out the main line's two ends (8 % of its length, at least 3 widths), where every crack thins to nothing. It is shown only: the level uses the widest point (`src/triage/matrix.py`).
- No level, rule or threshold changes.

## Where
- Phone page: after the answer, instead of the static picture; a retake or no-crack answer keeps the picture. Measurement details, the shape and downloads fold into "Measurement details"; the notes behind element and type into "How it was read".
- Wall: the result opened over the wall (D77) shows the viewer on the left; the window on the right holds the title, level and reason, the rest folded into "Measurement, photo 2 and heatmaps". The result's own page the same.

## Photos (rule 9)
The viewer's photo is a copy in memory only, as the wall's other pictures; never written to disk or exported (`WallEntry.zoom`, `WallEntry.marks`).

## Tests
`tests/test_crack_view.py`: widest and narrowest from the profile, tips left out, length as measured, pixels without a card, untrusted card width in pixels, nothing to draw, data cannot end the script, the mask without its text band.

## Known limit seen while building
On `data/samples/sample_crack_card.jpg` the long, very straight main crack is dropped by the "long straight line is an edge" rule (D51) and only its side branch is measured. The viewer makes this visible; the rule is not changed here.

## Shape row (D85, 20 Nov)
A second row of chips under the width chips, from the same measurement: Angle from vertical (photo 1, phone assumed upright; "in photo" without a card), Straight (end-to-end distance / main length, 1 = straight), Coherence (1 = one direction, 0 = none), Cells (closed cells enclosed by crack lines; from 2 the draft rules read map cracking), Branches (all branches / main length). Each chip explains itself on hover. No angle measured: no row. Shown only, never used for the level here. The camera angle to the wall is no longer a chip in the viewer; it stays under "Measurement details" (D85).
