# Midterm presentation (6 Nov 2026)

Live deck (private, share it from the page): https://claude.ai/artifact/QexYufRijMQWAu436T5kbj

This folder is a copy of its source, saved on 31 Oct 2026:
- `deck.json`: slide order (12 slides) and fonts.
- `slides/`: one HTML file per slide. Slides 2 and 3 hold the Surfside and Rana Plaza animations, drawn as 3D wireframes (31 Oct).
- `embeds/`: those two animations on their own, in 3D (`*_3d.html`, made by `build_3d.py`: a small wireframe renderer, no library, since the slides allow no network) and the earlier 2D versions (`*_2d.html`), to swap back if needed. Open a `*_3d.html` with `#11` after the name to jump ahead in the story (the number counts story steps of 0.7 s).
- `assets/`: the app's crack (`crack.svg`, cover and last slide), the cracked line (`underline.svg`) and the red-progress cracks for slides 2 to 12 (`crack12_NN.svg`).
- `crack_progress.py`: makes the cover crack and the red-progress cracks from `src/app/background.py`, both ends faded into the black as in the app; set `N` to the number of slides.

Facts and numbers on the slides are cited by number from `docs/sources.md` ([15], [41], [57] to [63]).
Still open: the QR on the live-demo slide (it changes at each app launch) and the news clip.
