# FEMA 306 (ATC-43, 1998): what it says about crack direction and width [93]

**Document:** Applied Technology Council (ATC-43 Project), *FEMA 306: Evaluation of Earthquake Damaged Concrete and Masonry Wall Buildings, Basic Procedures Manual*, Federal Emergency Management Agency, 1998.
- Full text (270 pages, scan with text layer, NIST): https://nehrpsearch.nist.gov/static/files/NIST/PB2001100763.pdf (SHA-256 `7dbdf475d87331d07d2abf411ff8390bb89c1e2029e44b67a57ff16f14c5d853`)
- ATC copy, front matter only (16 pages): https://www.atcouncil.org/files/FEMA306.pdf
- Library entry (EERI): https://mitigation.eeri.org/resource-library/building-professionals/evaluation-of-earthquake-damaged-concrete-and-masonry-wall-buildings-basic-procedures-manual-fema-306
- The PDF is not kept in the repository (13 MB); download it from the link.

**What we used it for (11 Nov 2026):** to check whether the angle bands of our written crack-type rules (D43, `src/triage/pattern_rules.py`: vertical up to 25 degrees, diagonal 35 to 55, horizontal from 65) are backed by engineering literature. Trigger: a field-test crack read at 28 degrees from vertical fell between the bands and was called unclear. Result: FEMA 306 backs the principle (flexure across the member, shear diagonal) but gives shear cracks in walls as 35 to 70 degrees from the horizontal, wider than our diagonal band; it gives no 25 or 65 degree limit. The rules in code are unchanged until the engineer decides.

Read 11 Nov 2026 from the full NIST scan (270 pages, text layer). Page numbers are the manual's own. Scope: **earthquake damage to concrete and masonry wall buildings** (wall piers, spandrels, coupling beams, infilled frames), not columns or slabs in general, and not cracks from settlement, shrinkage or corrosion.

## Crack direction (section 5.3.1 "Cracking", p. 83)
- "Flexural cracks are those that develop perpendicular to flexural tension stresses. **In wall piers, flexural cracks run horizontally; in wall spandrels, the cracks run vertically.**"
- "Shear cracks are those that result from diagonal tension stresses ... The cracks run diagonally, **typically at an angle of 35° to 70° from the horizontal**. The angle of cracking depends on normal forces (e.g., axial load) and on the geometry of the component." Cyclic loading makes them cross in X patterns.
- Flexural cracks often curve into diagonal shear cracks; their widths are to be measured separately.
- The guides ask the user to find "vertical cracking in the compression zone of wall piers" and "horizontal cracking in the compression zone of wall spandrels" (crushing), and cracks that may show lap-splice slipping.
- Strength formula (p. 89): the critical inclined shear crack is taken at 35 degrees **from the vertical** unless a corner-to-corner crack limits it (a design assumption, not a recognition rule).
- Infilled frames, chapter 8, a damage classification guide (labelled "INPS2" in the scan's text, p. 207): cracking across the panel's diagonals; at large drifts secondary cracks "at an angle of about 45 degrees to 65 degrees to the horizontal".

## Crack width by severity (Component Damage Classification Guides, chapter 5.5)
Example RC1A, isolated wall or stronger pier, ductile flexural (p. 96 to 97):
- Insignificant: no crack wider than 3/16 in. (4.8 mm), no shear crack wider than 1/8 in. (3.2 mm), no significant spalling or vertical cracking.
- Slight: crack widths up to 1/4 in. (6.4 mm), shear cracks up to 1/8 in.
- Moderate: spalling or vertical cracking at the toe.
- Other guides use 1/16 in. (1.6 mm) for insignificant shear cracking and 1/8 in. for the next step (for example lines of RC1B, RC3). The limits differ by component and behaviour mode.
These are for earthquake damage that engineers then analyse; they are much wider than the 0.3 mm and 1 mm edges of our width bands, which serve a different purpose (`spec/features/risk_matrix.md`).

## What it means for our pattern rules (D43; not changed in code)
1. **Backed:** the principle that the crack's direction against the element tells flexure from shear: across the member is flexure (horizontal in a pier, vertical in a spandrel or beam), diagonal is shear.
2. **Our diagonal band is narrower than FEMA 306's.** FEMA 306: shear cracks typically 35 to 70 degrees from the horizontal, that is **20 to 55 degrees from the vertical** in a wall pier. Our rule calls only 35 to 55 degrees from vertical "diagonal" and up to 25 degrees "vertical". A crack 28 degrees from vertical in a wall pier (field test, 11 Nov) lies inside FEMA's shear range, while our rules call it "in between", unclear (which gives Expert needed, so it is not lowered).
3. **Not backed by it:** the numbers 25 and 65 degrees, and anything for columns (FEMA 306 treats walls). The angle range is "typical" and depends on axial load and geometry, so no fixed band is exact.
4. Proposal for the engineer, not done: widen the wall's diagonal band towards FEMA's 20 to 55 degrees from vertical, or keep the gap and let unclear cracks go to an expert as now.

## Not read
ACI 224R-01 [94]: sold by ACI, no free official copy. The IAAC library may have access.
