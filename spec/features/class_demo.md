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
Dark page (#0b0b0b, Streamlit dark theme) with one big white crack behind everything, drawn light (half opacity, 30 Oct), forking near the top like a real wall crack (`src/app/background.py`). It is drawn as an SVG from a fixed seed, not a photo: no licence, sharp on any screen. Text and photos sit in front of it. Its two ends do not stop at the screen edge: at the top it fades in from the page colour (gradient over the background, 2 to 14 % of the screen height), at the bottom it runs under the band behind the model line (31 Oct, so it does not look cut out and pasted on). A red flash (#d62728) runs down the crack from the top, entering each branch and hairline when it reaches the point the branch leaves from, then the crack is white again for 2.5 s and the flash repeats (SVG animation inside the background image, no script; about 5 s per cycle).

## Phone page
Title "Second Look" as the word mark (below), then the team's quote in bold capitals (see Live wall), then:
1. One tick box (optional): "Share results" (puts the result on the class screen; 30 Oct, shortened from "Show my result on the class screen." to "Show results"; 4 Nov, "Share results": the photo is shared with the class, not only shown).
2. The consent terms, right above the upload: "By sending a photo you confirm you took it, it shows no faces, house numbers or addresses, and you agree it is analysed." Sending the photo is the consent (D34, 30 Oct: the separate consent tick was removed to keep one tick).
3. Take or upload one photo (JPEG or PNG, at most 10 MB).
4. Answer: the photo with the crack in red, the statement (Retake, Crack found, No crack found or Expert needed), a one-sentence reason with the confidence, and "This is not a safety check." Since 4 Nov (D46): with a bank card flat next to the crack, a "Measurement · bank card" block adds the width (widest and median, ± uncertainty), the length, the pixels across, the camera tilt, the width band of the upper end (not a level), the crack shape, a field-worker pattern suggestion and "Result (JSON)" / "Overlay (PNG)" downloads (`spec/features/app_measure.md`). Never Category 1/2/3: the pattern and the element are not known.

## Live wall
- Word mark (4 Nov, from the deck cover): in both titles the two O's of "LOOK" are scan frames. On load the second O slides out of the first and a red crack draws through it; it plays once (CSS only, `src/app/logo.py`). Screen readers get "Second Look".
- Top, very large: "SECOND LOOK v1.0" (the version in lower case; the word mark), left-aligned with the quote under it (the two form one block, centred on the screen). Right of the title, under "v1.0", "READING CRACKS..." types itself letter by letter, holds, disappears and starts again (4 s loop, CSS only).
- Screen bottom: the model in use (U-Net, ResNet-34 encoder, crack model v4 since 9 Nov, trained on 12,477 photos, D48; `MODEL_LINE` in `second_look.py`, to update with the model) stands still as one line of grey capitals (white at the main crack's opacity, so the same grey as the background crack), from the left edge of the screen to the right edge (letter size from the letter count, words justified; `model_line` in `background.py`; D35). A band in the page colour sits behind it, fading out at its top, so the crack does not run through the letters. Only the red flash on the crack and the "Reading cracks..." typing move.
- Middle: four large buttons, one per statement (Crack found, No crack found, Expert needed, Retake), each with its count, in a row centred on the screen. The buttons have no box: white text, red when opened or under the mouse. One continuous cracked line with a few hairlines, grey like the background crack, runs under the whole row, with one branch rising from it to each button (off the button's centre, each by a different amount, so not symmetrical); it is drawn in four seamless pieces, and the piece under a hovered or opened button turns red (CSS `:has`, current Chrome, Edge or Firefox) (`crack_underline_svg` in `background.py`; 30 Oct, replaced a cracked outline, then a line per button). The line "Click a result to see its photos." was removed (30 Oct). **No photo is shown by default.** Clicking a button opens that statement's collection below (every shared result of it still in memory, 6 per row, newest first; since D46 each photo carries its widest width and opens its own page with the numbers large); another button switches, a second click closes. The wall keeps the newest 40 shared results overall. Expert needed stays: it is one of the outputs (rule 2). Only results whose sender ticked the box.
- Right under the title, starting at the same left edge, bold capitals on one line: "A repair that conceals a crack without addressing its root cause is not a solution, It is a temporary disguise for a potentially continuing defect." (team's wording, 30 Oct). The phone page shows it under its title too, wrapped over several lines (one line would be unreadable on a phone).
- No QR on the wall (1 Nov). The QR has its own page, `http://localhost:8501/?qr`: the code alone, centred, 80 % of the screen height, shown on the projector when students should scan. (Before: a fixed bottom-right corner.)
- Refreshes every 4 seconds without dimming the page; each result is stored once as a small JPEG, so a refresh re-encodes nothing (30 Oct: the first version re-encoded every photo every 3 s and looked frozen).

## Photos (rule 9)
- Processed in memory, never written to disk; GPS and other metadata are never read (only the pixels are used).
- The wall keeps shared results in memory only, a small copy of the answer image; they are gone when the app stops.
- The link works only while the launcher runs; one photo per upload, 10 MB limit.

## Limits
The model is trained on public close-ups of cracks (`docs/experiments/2026-10-29_unet_first_training.md`).
Checked on 30 Oct with 30 full-resolution photos of damaged buildings (Haiti 2010 engineer photos, local only): all 30 came back "Crack found" at about 90 % confidence, in 0.25 s each (1.4 s at most). On close-ups it marked the real cracks; on whole scenes it also marked window frames, wires, roof edges and people. The model has never seen anything but close-ups of surfaces.
Model v3 (brick no-crack photos added, D32) still read 28 of the 30 as "Crack found". Model v4 (building-concrete crack masks added, D48) has not been run on these 30 photos yet; it was trained on close-ups too, so the same limit is expected. So the phone page asks for close-ups (20 to 40 cm, wall fills the photo). A real fix needs a "close-up of a surface?" check before the model, or training with photos of non-crack lines (brick joints, edges, frames). Until then, wrong answers are part of the demo: an engineer confirms.

## Tests (`tests/test_app_core.py`)
Consent is required, only shared results reach the wall, the wall keeps the newest 24, counts, wall key check, link read from the tunnel's output, QR code made.
