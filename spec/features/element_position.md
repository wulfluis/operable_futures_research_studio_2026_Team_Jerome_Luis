# Feature: where on the element the crack is, from the two photos (`src/element/locate.py`)

Status: first version, 11 Nov 2026, decisions D58 and D59 (the bank card as the bridge). Luis, local branch `element-position`. Tested on synthetic photo pairs only. Every limit below is illustrative until checked on real photo pairs and with the engineer.

## Purpose
The app has two photos: the crack close-up and the whole element (D53). Until now they were not connected, so the app did not know where on the element the crack is (position always "unknown", counted as an end zone) and read the crack's direction against the phone, not against the element. This step finds the close-up inside the element photo. From that it reads:
1. **The element's axis**: the long edges of the column or beam in the element photo. The crack's angle is then measured against the element, so a phone held at a slant no longer changes the type.
2. **The position**: end zone or middle of the element, from the crack's distance to the visible ends of the element.

## How
0. **Link through the bank card first (D59).** The first field test (11 Nov) failed: key points did not link a close-up to an element photo about 3 m away on a phone with a modest camera. The card lies flat on the same wall as the crack and is in both photos, so its four corners in each give the mapping of the wall from one photo to the other exactly, however thin the crack looks from afar.
   - The card in the close-up is the one the app measured with (only with the tick "I put a bank card next to the crack", D56). Its look is kept: the card straightened to 1 px per mm.
   - In the element photo the card is searched for by that look (template matching on a copy 2000 px long, at 20 sizes from 1.2 % of the photo's side up to the close-up card's own size, turned as it lay in the close-up, and half round). The look includes a band of wall round the card (30 % of its size): a black card with little print is almost uniform inside and matched a plant in the field test of 11 Nov; with the wall round it, it is a dark card on a light wall. The eight best places are checked in turn; at each, the card finder looks for a card's outline in a small window (the same two searches as for the close-up, the side contrast of D56 measured just inside the edge so a stripe near it does not count, fit to a card within 5 % of its diagonal, illustrative). The first outline that also looks like the card (correlation at least 0.5, illustrative) is the card.
   - Refused: a plain card (no look to tell it from a switch or a tile), a second place at about the same size that looks nearly as much like the card (within 0.1), and a card that does not look smaller than in the close-up.
   - The four corners are matched in the same order in both photos (clockwise from the top left, starting on a long side); this assumes both photos are taken roughly upright. The look also tells a card turned half round.
   - The crack is carried into the element photo through the card's homography, so it is placed correctly even when it is too thin to see there.
   - About 1 s on a 12 MP photo.
1. **Otherwise link by key points.** Both photos in grey, long side at most 1600 px. SIFT key points (OpenCV), matches kept by the ratio test (0.75), a homography by RANSAC. The link counts only with at least 20 matching points that fit it, and the close-up must land inside the element photo as a convex shape smaller than 80 % of it (illustrative).
2. **The crack in the element photo.** The crack's line from the close-up (with a card: the measured line; without: the U-Net line in pixels, D57) is carried into the element photo through the homography.
3. **The element's edges** (column or beam only, as read by the element model). Straight edges of at least 35 % of the photo's long side (Canny, Hough), within 15 degrees of the photo's vertical for a column or horizontal for a beam. Pieces on one line are joined. The nearest such edge on each side of the crack is kept; both are needed. The axis is their mean direction; the element's depth is the distance between them at the crack.
4. **The ends.** An end is where both edges stop, taking the nearer one. An end within 2 % of the photo's side from the border is cut by the photo, so not seen.
5. **Position** (the risk matrix's locations):
   - **End zone**: some part of the crack lies within one element depth of a seen end. The risk matrix gives the zone's length as about one element depth (assumption, in `risk_matrix_v3.json`).
   - **Middle**: both ends are seen and every part of the crack is more than one depth from both.
   - **Unknown** (counts as an end zone, as before): anything else. That includes no link, no two edges, a wall, an element the model is unsure of, or an end cut by the photo.
6. **Angle for the type rules.** With a link and two edges, the crack's angle to the element's axis replaces the angle to the photo's vertical in the type rules (D43). Otherwise the old angle stays.

## The true vertical, for walls and whenever the photos are linked (11 Nov, D61)
A wall has no axis of its own, and the crack's angle was read against the phone. Once the photos are linked, the element photo shows straight lines of the room: door and window frames, corners (vertical), the ceiling, a cornice, a lintel (horizontal). From them the app reads the true vertical where the crack is:
- Straight edges at least 15 % of the photo's long side (Canny, Hough), within 15 degrees of the photo's vertical or horizontal (illustrative). A horizontal line counts by its perpendicular.
- Each line counts by its length, more when it is near the crack (weight 1 / (1 + distance / long side)), so a corridor's floor lines far away count little. The tilt is the weighted median; at least 2 lines, together at least 30 % of the long side, else none.
- The crack's direction (the main line carried into the element photo) is measured against it: **vertical** up to 25 degrees, **diagonal** 35 to 55, **horizontal** from 65, between the bands "in between" (the bands of the type rules, D43).
- For a wall, this angle replaces the photo's angle in the type rules (rule 8: diagonal in a wall). For a column or beam the element's own edges stay first.
- The direction is shown even when the element is not clear ("Direction: diagonal, 42 degrees from vertical, read against the room's vertical lines"); it gives no type and no level by itself without an element.
- Perspective: a phone tilted up or down makes vertical lines lean towards each other; the lines near the crack count most, which limits it (assumption, to check on real photos).

## Never lower by doubt
Only "middle" can give a lower level than before; it needs a link, two edges, both ends seen and the crack clear of both end zones. Anything short of that keeps "unknown", which counts as an end zone (rule 2). A wall or a slab always keeps "unknown". "Joint" is not read (it needs the beam and the column at once); a crack near a joint falls in the end zone.

## In the app
- The Position cell shows the position and "from the two photos", or "not known" with the reason.
- A panel "What the app found in the two photos (for testing)" lists each step: card in the close-up, the best look scores in the element photo, whether a card outline was found there, key points, the element's edges. No photo is kept for it.
- The phone page shows a small copy of the element photo with the close-up's outline (white), the card (cyan), the crack with a ring around it (red, so a thin crack can be found) and the two edges (yellow), so the sender sees what the app linked and where the crack is. The element photo stays off the wall (rule 9), as before.
- The result page shows the position and where it came from.

## Limits
- Plain plaster with no pattern gives too few key points; then only the card can link the photos.
- The card bridge needs the card in both photos, not plain, and at least about 2 % of the element photo's side (about 3 to 6 m with a phone's main camera, illustrative).
- Close-up at about 1 m and element photo at several metres work best. At about 10 m the close-up covers only a few dozen pixels of the element photo and the link usually fails.
- An edge that runs on into a wall or a beam beyond the element's end makes the element look longer. The end then seems farther away than it is, and "middle" can be read too often. This needs real photos and the engineer before "middle" is trusted.
- One straight crack line; a branching crack is read along its main path.

## Tests (`tests/test_element_position.py`)
Synthetic photo pairs made from one high-resolution textured scene: a column with both ends seen, the close-up 4 times closer.
- The link finds the close-up where it was cut out.
- A crack in the middle is "middle"; one near an end is "end zone"; a column cut by the photo border gives "unknown".
- An unrelated photo gives no link and "unknown".
- A close-up turned by 20 degrees still reads a horizontal crack as across the column.
- A wall gives "unknown".
- Through the app, the position reaches the level, and nothing says "safe".
- The card links a plain wall where key points cannot, with a crack 1 px wide; it puts the crack's centre where it is in the element photo; a plain card-shaped thing elsewhere is not taken, and a photo with only that thing is not linked; a card near the column's end gives the end zone; through the app with the card tick.
