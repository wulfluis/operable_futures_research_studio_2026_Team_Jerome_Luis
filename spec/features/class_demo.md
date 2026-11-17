# Feature: class demo, scan a QR and send a crack photo (`src/app/`)

Status: first version, 30 Oct. A demo of box [6] (crack or no crack, `spec/features/segmentation.md`) for the studio, not the resident app.

## Purpose
Students scan a QR code on the projector, take a photo of a crack with their own phone and get Second Look's answer on the phone. A live wall on the projector shows the answers of everyone who agreed to share.

## How it runs
- Needs `cloudflared` once: `winget install --id Cloudflare.cloudflared`.
- `py src/app/launch.py` on the laptop starts the app (Streamlit, port 8501) and a tunnel: the team's fixed ngrok link (4 Nov, D40: the same link and QR at every start; one laptop at a time), or, when ngrok is missing or fails, a temporary Cloudflare quick tunnel with a random https link (the launcher says the QR in the slides will not work then). It prints the link, saves a QR code to `data/results/app/qr.png` and opens the live wall in the browser.
- The model runs on the laptop's GPU; the laptop must stay on, plugged in and online. Closing the launcher closes the link.
- Phone page: `/`. Live wall: `http://localhost:8501/?wall`, opened in the laptop's own browser. It opens only when the host is localhost and the request did not come through the Cloudflare link (which adds `Cf-*` headers and keeps the public host name), so a student cannot open the wall from the link. Fallback: `/?wall=<key>` with the random per-launch key (30 Oct: a key to copy proved impractical on stage). The launcher prints that link for a teammate (31 Oct); it is sent privately and never shown on screen, since anyone with it sees the shared photos.

## Look
Since 12 Nov (D76): GitHub's design (Primer, light), `src/app/theme.py`. White page, system font, thin grey borders, 6 px corners, nothing moves and no background picture. Header: the word mark, large, in bold with wide letter spacing, then "HUMAN IN THE LOOP" in small bold capitals after a grey rule as tall as the name (no version label, 12 Nov), one grey line under it; no quote (D78). Answers are pills in fixed colours, never green (green reads as "safe", rule 1): Category 3 and Crack found red, Category 2 amber, Category 1 blue, Expert needed purple, Retake and No crack found grey. (Until 11 Nov: a black page with a big animated white crack behind it, `background.py`, removed.)

## Phone page
Title "Second Look" as the word mark (below), then:
1. One tick box (optional): "Share results" (puts the result on the class screen; 30 Oct, shortened from "Show my result on the class screen." to "Show results"; 4 Nov, "Share results": the photo is shared with the class, not only shown).
2. The consent terms, right above the upload: "By sending a photo you confirm you took it, it shows no faces, house numbers or addresses, and you agree it is analysed." Sending the photo is the consent (D34, 30 Oct: the separate consent tick was removed to keep one tick).
3. Take or upload one photo (JPEG or PNG, at most 10 MB).
4. Answer: the photo with the crack in red, the statement (Retake, Crack found, No crack found or Expert needed), a one-sentence reason with the confidence, and "This is not a safety check." Since 4 Nov (D46): with a bank card flat next to the crack, a "Measurement · bank card" block adds the width (widest and median, ± uncertainty), the length, the pixels across, the camera tilt, the width band of the upper end (not a level), the crack shape, a field-worker pattern suggestion and "Result (JSON)" / "Overlay (PNG)" downloads (`spec/features/app_measure.md`). Never Category 1/2/3: the pattern and the element are not known.

## Live wall
- Word mark (D38, still since D76): the two O's of "Look" are scan frames side by side, one in ink, one in grey; no crack in it, no animation (`src/app/logo.py`). Screen readers get "Second Look".
- Top: the header (word mark, "HUMAN IN THE LOOP", "Live · shared results appear by themselves" on the right). No quote (removed 12 Nov, D78).
- Middle (D78): one tab per risk level, highest first: Category 3, Expert needed, Category 2, Category 1, each with its count in a grey counter, as GitHub's tabs; the opened tab is underlined in orange. The tab is the level from the written risk matrix (`core.risk_group`); a crack with no level yet, or a model unsure, is under Expert needed (label "Waiting for photo 2" when photo 2 is missing); a retake or a photo without a crack is not on the tabs. **No photo is shown by default** ("Choose a tab to see its results."). Clicking a tab opens its cards (newest first): the photo without the statement's band, photo 2 inset, then the number and labels (TEST, widest width when measured, the level, the element). A click on a card opens its result over the wall, on the same page (D77): the wall goes blurred and lighter, the clicked crack stays sharp, and a window beside it holds the rest, scrolled inside; a click beside them or ✕ closes it. Another tab switches, a second click closes. The wall keeps the newest 200 shared results overall (D79; 40 before), so older cases stay. Only results whose sender ticked the box.
- Bottom: the model in use (`MODEL_LINE` in `second_look.py`, to update with the model) in a small grey footer.
- Result page (the window over the wall, D77, and a direct link `/?wall=...&item=<n>`): laid out as a GitHub issue: "Back to the wall", the statement with its number, pills for the statement and the level, the reason; boxes for photo 1, photo 2, SAM 3.1's picture, the measurement, and element, position, type and level; then the 3D view on a light grey ground.
- No QR on the wall (1 Nov). The QR has its own page, `http://localhost:8501/?qr`: the code alone, centred, 80 % of the screen height, shown on the projector when students should scan. (Before: a fixed bottom-right corner.)
- Refreshes every 4 seconds without dimming the page; each result is stored once as a small JPEG, so a refresh re-encodes nothing (30 Oct: the first version re-encoded every photo every 3 s and looked frozen).

## Location and type (11 Nov, D53)
After "Crack found" or "Expert needed" the phone page asks for a photo of the whole element and shows element, suggested type and position; the risk category follows by itself (D55). The wall's photos and result pages carry element and level. Spec: `spec/features/app_classification.md`.

## Card tick (11 Nov, D56)
A second tick above the upload, "I put a bank card next to the crack", off by default: without it the app never looks for a card. It sits above "Share results".

## Photos (rule 9)
- Processed in memory, never written to disk; GPS and other metadata are never read (only the pixels are used).
- The wall keeps shared results in memory only, a small copy of the answer image; they are gone when the app stops.
- The link works only while the launcher runs; one photo per upload, 10 MB limit.

## Limits
The model is trained on public close-ups of cracks (`docs/experiments/2026-10-29_unet_first_training.md`).
Checked on 30 Oct with 30 full-resolution photos of damaged buildings (Haiti 2010 engineer photos, local only): all 30 came back "Crack found" at about 90 % confidence, in 0.25 s each (1.4 s at most). On close-ups it marked the real cracks; on whole scenes it also marked window frames, wires, roof edges and people. The model has never seen anything but close-ups of surfaces.
Model v3 (brick no-crack photos added, D32) still read 28 of the 30 as "Crack found". Model v4 (building-concrete crack masks added, D48) has not been run on these 30 photos yet; it was trained on close-ups too, so the same limit is expected. So the phone page asks for close-ups (20 to 40 cm, wall fills the photo). A real fix needs a "close-up of a surface?" check before the model, or training with photos of non-crack lines (brick joints, edges, frames). Until then, wrong answers are part of the demo: an engineer confirms.
10 Nov (D51): the model still marked the line between a wall and a ceiling as a crack. A written rule now ignores a long straight line that reaches a side of the photo (`spec/features/segmentation.md`, "Straight edges are not cracks"); the answer says so in its reason.

## Tests (`tests/test_app_core.py`)
Consent is required, only shared results reach the wall, the wall keeps the newest 24, counts, wall key check, link read from the tunnel's output, QR code made.

## Simulated residents: the test inbox (11 Nov, D59)
To show the wall, the levels and the 3D models without a class full of phones, `py src/app/simulate.py` drops test cases into `data/inbox/` (local, ignored by git), one every few seconds. The running app picks each one up only if that folder exists, and treats it as a resident's upload: same pipeline, consent by sending, shared on the wall, written for Grasshopper. A case is a folder with `crack.jpg`, optionally `element.jpg`, and `case.json` (`card_placed`, `kind`). Every simulated result is marked **TEST** on the wall and in its numbers. The cases: synthetic photos with a bank card and cracks of known width and direction (`src/measure/synthetic.py`) paired with real column, beam and wall photos from the PHI-Net task 6 test set; real building-concrete crack close-ups without a card (Özgenel test tiles [49]); a wall without a crack; a whole building (PHI-Net task 1 test). Processed cases move to `data/inbox/done/`.
