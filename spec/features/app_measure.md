# Feature: bank-card measurement in the class demo (`src/app/`)

Status: first version, 4 Nov 2026, Luis's experiment (D45); folded into the class demo itself on 4 Nov (D46): the copy in `src/app_measure/` and its launcher are gone, the page is `src/app/second_look.py`, the logic `src/app/core.py`, served by the team's launcher on the fixed ngrok link and QR (D40). Tested on synthetic photos only.

## Purpose
The crack width in mm inside the class demo, with its look unchanged (`app/background.py`, `app/logo.py`) and the pipeline functions of `segmentation/predict.py` and `decide.py`.

## What it does
1. As the original: size and format check, quality gate, U-Net crack map (model v3), written rule, red overlay.
2. Looks for a bank card (85.60 x 53.98 mm, `src/measure/card.py`, D42).
3. **No card:** the answer is the original's, statement, reason and overlay pixel for pixel (tested).
4. **Card found:**
   - The card and a band of 1.5 % of the photo's long side around it are taken out of the crack map before the written rule, so its straight dark edges are never "Crack found".
   - The U-Net says where the crack is. Its mask is about twice as wide as the crack (synthetic sample: 1.89 mm against 1.40 mm true), so inside it, widened by 3 px, the classical finder (`src/measure/crack.py`) takes the crack's edges at full photo resolution. On the sample this gives 1.48 mm against 1.40 mm, and a length of 174 mm against 173.3 mm. When the classical finder finds nothing there, the U-Net mask is used, with a warning that the width may be too large.
   - Length, branches and crack shape come from the U-Net's crack line, which stays continuous on a faint crack. The classical finder sees a faint crack only in pieces (field case 4 Nov: a length far too short), so every piece along the line counts for the width, none for the length. Synthetic sample: 177 mm against 173.3 mm.
   - Shown below the original answer: width at the widest point ± uncertainty (±1 px [46] plus the scale error), median width, length of the main crack and of all branches, pixels across, camera tilt (straight, tilted or retake; left-right and up-down), the width band of the risk matrix for the upper end of the width (not a level), the crack shape and, in a field-worker section, the draft pattern suggestion (D43). Also: JSON and overlay downloads.
   - The overlay shows the card outline and a ring at the widest point; the reason line carries the width and the length.
5. The statements stay the original four. The tool never outputs "safe", a clearance or a level. Tilt over 20° or fewer than 3 px across: "Width not measured" with the reason, and the statement is unchanged.

## Live wall: a page per result (added 4 Nov)
- As the original: one button per statement opens its collection.
- New in the copy: each photo is a link to its own page, `/?wall=<key>&item=<n>`, and carries its widest width as a small label.
- The page sits on plain black with no background crack. It shows the statement, the reason and the larger overlay, and the numbers large: widest width ± uncertainty, median, length and all branches, pixels across, camera tilt with left-right and up-down, model confidence, width band (not a level), crack shape, warnings and "not a safety check".
- Without a card it says "Width not measured" with the reason. "Back to the wall" reopens the same collection.
- The wall keeps, in memory only, a 480 px and a 1400 px copy and the result's numbers, for the newest 40 shared results.

## How it runs
As the class demo (`spec/features/class_demo.md`): `py src/app/launch.py`, port 8501, the team's fixed ngrok link and the QR in the slides (D40), Cloudflare as the fallback. Live wall: `http://localhost:8501/?wall`; a result's page: `/?wall=<key>&item=<n>`; QR alone: `/?qr`. (Until D46 a separate launcher served the copy on port 8503 with a self-signed certificate on the local Wi-Fi; it was removed so the team has one app on one link.)

## Limits
- Tested on synthetic photos only. Thresholds (card band, the 3 px widening) are illustrative until calibrated on real photos with measured cracks.
- The pattern suggestion assumes the phone is held upright.

## Tests (`tests/test_app_measure.py`)
- Without a card the result is identical to the original.
- With a card the width, length and tilt match a synthetic photo's known answer.
- Card edges alone are not a crack.
- No "safe" word.
- Only shared results reach the wall.
- The page keeps the class demo's layout and adds the card hint.
- The wall needs its key.
- There is one app: no `src/app_measure/` copy (D46).
