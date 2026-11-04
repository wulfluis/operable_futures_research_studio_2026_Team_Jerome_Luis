# Feature: class demo copy with bank-card measurement (`src/app_measure/`)

Status: first version, 4 Nov 2026, Luis's experiment. Decision D45. Tested on synthetic photos only.

## Purpose
Try the crack width in mm inside the class demo without touching it. The original (`src/app/`, `src/segmentation/`, `spec/features/class_demo.md`) stays exactly as it is; this copy imports its look (`app/background.py`, `app/logo.py`) and its pipeline functions (`segmentation/predict.py`, `decide.py`) and changes none of them.

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
`python3 src/app_measure/launch_measure.py`. It uses port 8503, not 8501, so both apps can run side by side. It uses https on this laptop's Wi-Fi address with a self-signed certificate (made in `data/results/app_measure/`, ignored by git) and a QR of that link. The team's ngrok link (D40) is not used. Phones must be on the same Wi-Fi and accept the certificate warning once. Live wall: `https://localhost:8503/?wall`. QR alone: `/?qr`.
For people on other networks: `python3 src/app_measure/launch_measure.py --public` adds a Cloudflare quick tunnel (`brew install cloudflared`; no account): a random public https link with a proper certificate, new at every start, and its QR in `data/results/app_measure/qr_public.png`. Anyone with the link can send photos; the wall opens only on this laptop or with the printed key. The link dies when the launcher stops.
Under the deck's QR (team ngrok link https://cranium-bladder-tapioca.ngrok-free.dev, D40): `python3 src/app_measure/launch_measure.py --ngrok`. It needs the team's ngrok authtoken on this laptop (`brew install ngrok`, `ngrok config add-authtoken <token>`, never in git) and Jerome's app must not serve the link at the same time (one laptop per link). The QR and Jerome's files stay unchanged; only the app behind the link changes, so the team agrees before it is used (open).

## Limits
- Tested on synthetic photos only. Thresholds (card band, the 3 px widening) are illustrative until calibrated on real photos with measured cracks.
- The pattern suggestion assumes the phone is held upright.

## Tests (`tests/test_app_measure.py`)
- Without a card the result is identical to the original.
- With a card the width, length and tilt match a synthetic photo's known answer.
- Card edges alone are not a crack.
- No "safe" word.
- Only shared results reach the wall.
- The page keeps the original's layout and adds the card hint.
- The wall needs its key.
- The original page does not import the copy.
