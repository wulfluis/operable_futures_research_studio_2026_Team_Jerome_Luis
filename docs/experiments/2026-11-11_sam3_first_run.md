# SAM 3.1 on the laptop: first run by text prompt (11 Nov 2026)

Offline test, not in the app (reading list [81][82], `src/experiments/sam3_test.py`). Why: the card, the element and the wall in the far element photo are hard to find with our own edge search (D58, D59, D61). SAM 3.1 finds objects from a short text ("credit card", "wall") without training.

## Setup
- Weights: Hugging Face `facebook/sam3.1` (gated, access granted to Luis on 11 Nov), file `sam3.1_multiplex.pt`, 3,502,755,717 bytes, SAM License (Meta, 19 Nov 2025). Kept in `models/sam3.1/`, never in git.
- Code: Ultralytics 8.4.210 `SAM3SemanticPredictor`, which reads the SAM 3.1 checkpoint. Text tokenizer from PyPI `openai-clip` 1.0.1 (only `clip.simple_tokenizer`; the script adds the call that Ultralytics' own CLIP fork has).
- Weights check: the file fills 1,110 of the image model's 1,118 tensors. The 8 not filled are the neck's lowest-resolution level, which Ultralytics drops anyway (`scalp=1`).
- Laptop: MacBook Pro M4, 16 GB, GPU (MPS), fp32.
- Image size 644 px (Ultralytics' default). **1008 px (SAM 3's own size) ran out of GPU memory** on 16 GB while two app copies were running.

## Result on `data/samples/sample_crack_card.jpg` (synthetic close-up, 3000 x 2250)
| Prompt | Objects | Best confidence |
|---|---|---|
| credit card | 1 | 0.96 (outline exact by eye) |
| crack | 3 | 0.71 (main line and branch) |
| wall, door, window, ceiling, floor, concrete column, concrete beam | 0 | – |

- Time: 13 s for the first photo including loading the model; 9.8 s to read the image, 2.6 s for the nine prompts.
- "wall" gives nothing on a close-up of plaster: there is no wall *object* in the photo. Whether SAM 3.1 tells wall, door and column apart must be tested on a room or element photo (next step).

## Result on a room photo (`data/samples/raum_tuer.png`, 3024 x 4032, Luis's flat, no crack or card in it)
| Prompt | Objects | Best confidence | Share of photo |
|---|---|---|---|
| wall | 8 | 0.89 | 41 % |
| door | 5 | 0.95 | 6 % |
| door frame | 4 | 0.94 | 7 % |
| ceiling | 2 | 0.95 | 5 % |
| floor | 3 | 0.81 | 11 % |
| window | 2 | 0.41 | 0.3 % |
| concrete column, concrete beam, credit card, crack | 0 | – | – |

- **The walls come out as separate objects** (each coloured on its own in `data/results/sam3/raum_tuer_walls.png`): the front wall (0.79, 21 % of the photo) and the right-hand wall (0.89, 16 %) are split exactly at the room's corner; the corridor walls through the door are further, smaller objects (0.29 to 0.83).
- Objects on the wall are left out of the wall mask (book shelf, switch, plant, armchair); the cornice is not wall.
- Door frame and door leaf are told apart.
- No false column, beam, card or crack in this photo.
- Time: 10.1 s for the image, 2.4 s for ten prompts (MPS, 644 px).

What it could give the app (not built, to agree first): the wall mask the crack lies on (which wall), the boundary between two wall masks as a true vertical line (D61), the door-frame mask as the opening corner (option B).

## More prompts on the same room photo (`data/results/sam3/raum_tuer_more_prompts.png`)
- Found well: painted wall (8, 0.89), cornice (4, 0.91, exact), furniture (9, 0.97), plant (4, 0.95), floor tiles (64 single tiles, 0.59).
- Wrong or weak: "arch" (0.58) also takes the square door frames; the light switch is found only as "electrical outlet" (0.68); "wall corner" finds nothing, though the corner is the border between two wall masks.
- Nothing, correctly or not testable here: plaster, brick, concrete (a painted wall shows no material), stain, peeling paint, mould (none in the photo).
- 11.1 s for the image, 7.3 s for 15 prompts.
- Reading: SAM 3.1 finds *things you can point at* by name. It does not judge material under paint, and it knows nothing about loads (rule 6).

## Elements only (default of the script since 11 Nov; `data/results/sam3/raum_tuer_elements.png`)
Only the risk matrix's elements are asked for (Luis, 11 Nov: the app needs only the parts that matter for cracks). Word test on the room photo: column, pillar, concrete column, beam, concrete beam, lintel, slab, concrete slab: **nothing** (none in the room, and no false find); ceiling 2 (0.95); wall 8 (0.89). So "slab" is asked as "ceiling" or "slab", "column" as "column" or "pillar" (pillar still to check on a photo with a column).
- Result: slab 2 (0.95, the ceiling of the room and of the corridor), wall 8 (0.89, 45 % of the photo), column 0, beam 0.
- Load-bearing or infill wall cannot be told from a photo; the script says so on every wall.
- 9.5 s for the image, 1.0 s for the six words.
- **Open: column and beam are untested.** No photo with a visible column or beam is on the laptop.

## Luis's element list (11 Nov, now the script's default)
Column (column / concrete column), wall (concrete wall), beam (beam / concrete beam), slab (ceiling slab / concrete ceiling), balcony, stairs (staircase / stairs), window, expansion joint. Room photo (`raum_tuer_elements.png`):
- wall 7 (0.68): all walls found, but less sure than with "wall" (0.89); the walls are painted, not visibly concrete.
- slab 2 (0.95): room and corridor ceiling.
- window 2 (0.41): **wrong**, the mirror in the corridor (there is no window in the photo). A window at 0.41 must not be trusted.
- column, beam, balcony, stairs, expansion joint: 0 (none in the photo, no false find).
- 9.8 s for the image, 3.8 s for the 13 words.

## Limits
- Two photos; nothing measured against labels yet. One room, painted plaster, not an informal building (D14).
- Too slow for the app as it is (about 10 s per photo on the laptop, more on a small server; assumption).
- Licence: the SAM License allows use, copies and derivative works; read its full terms before any use beyond this test.
