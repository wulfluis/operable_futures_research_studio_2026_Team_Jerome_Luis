# Engineering sources for crack-pattern rules with angles (search of 11 Nov 2026)

**Why:** our written crack-type rules (D43, `src/triage/pattern_rules.py`) use angle bands: along the element up to 25 degrees, diagonal 35 to 55, across from 65. They were set by us, "illustrative until a structural engineer sets it". This search looked for public, engineering-backed texts that give crack patterns with rules and degrees, to back or correct those bands. Numbers in brackets refer to `docs/sources.md`. Only what was read is reported; "not read" is said where a text could not be opened.

Short list of the links with one or two sentences each: `docs/literature/crack_rules/README.md`.

## Result in one paragraph
**No public guideline found gives fixed angle bands for classifying crack patterns.** The engineering texts agree on the principle (a crack across the member's axis is flexure, a diagonal crack is shear, a crack along the bars or in the compression zone points to splitting or crushing), and give angles only as *typical ranges* that depend on load and geometry. The one numeric shear range found in a guideline is FEMA 306's: shear cracks in walls "typically at an angle of 35° to 70° from the horizontal" [93]. One peer-reviewed image-analysis paper uses a fixed cut-off of its own: within 22.5 degrees of the horizontal is flexural, the rest shear, and says it is not general [96]. Crack widths by damage class are better documented than angles (JBDPA [98] via [97], FEMA 306 [93]).

## Sources read

| Source | Who backs it | What it says about angles | What it says about widths | Use for us |
|---|---|---|---|---|
| **FEMA 306** (1998) [93], `docs/literature/fema_306.md` | Applied Technology Council for FEMA; engineering guideline for earthquake-damaged walls | Flexural cracks perpendicular to tension: horizontal in wall piers, vertical in spandrels. **Shear cracks "typically at an angle of 35° to 70° from the horizontal"**, depending on axial load and geometry; X patterns under cyclic load (§5.3.1, p. 83). Infill panels: diagonal cracks, secondary ones "about 45 degrees to 65 degrees to the horizontal" (p. 207) | Per component and behaviour mode, e.g. RC wall, ductile flexure: insignificant up to 3/16 in. (4.8 mm), shear cracks up to 1/8 in. (3.2 mm) (p. 96 to 97); other guides 1/16 in. (1.6 mm) for shear | Backs the principle; **our diagonal band (35 to 55 from vertical) is narrower than FEMA's (20 to 55 from vertical in a wall pier)** |
| **FEMA 307** (1998) [95] | Same; technical background to FEMA 306 | No degrees. Diagonal cracks "reflect associated shear stresses, but they may be a natural part of ductile flexural action"; in flexural plastic hinges they start from the ends of flexural cracks (ch. 3, reinforced masonry, p. 37) | "Crack width alone is a poor indicator of damage severity" (citing Hanson 1996); "the location, orientation, number, and distribution of the cracks must be considered as important as, if not more important than, the crack width" (p. 37) | Backs using pattern and position, not width alone (our matrix) |
| **Yang, Chang, Wu 2019** [96], Sensors, peer-reviewed, CC BY | Researchers, tests on shear-critical tubular RC (containment-vessel) specimens | Crack orientation from images; **"horizontal cracks, whose orientation is between 0 to 22.5 degrees or 157.5 to 180 degrees, are classified into flexural cracks"**, the rest as shear; observed shear cracks mostly at 45 or 135 degrees. The authors say this classification is not necessarily applicable to all structure types | – | The only fixed angle cut-off found; close to our 25-degree band, but chosen for their specimens |
| **Quan, Choi, Jin 2025** [97], Materials, peer-reviewed, CC BY, quoting **JBDPA** [98] | Japan Building Disaster Prevention Association guideline (1991, rev. 2001, 2015), the Japanese post-earthquake practice | No angles in the class table | **Damage classes of RC members: I under 0.2 mm; II about 0.2 to 1.0 mm; III about 1.0 to 2.0 mm with local crushing of cover; IV over 2.0 mm with exposed bars and spalling; V buckled bars, cracks in core concrete, visible deformation** (Table 4) | Width bands of the same order as our risk matrix's edges (0.3, 1, 3 mm, illustrative); a candidate reference for the engineer |
| **Eurocode 2** (EN 1992-1-1) [101], via The Concrete Centre | European design standard | Shear design assumes compression struts at 21.8 to 45 degrees to the member axis (1 ≤ cot θ ≤ 2.5) | – | A **design** assumption for strength, not a rule for reading a crack in a photo; shows diagonal cracks can be flatter than 45 degrees |

## Checked, not read
- **JBDPA guideline itself** [98]: in Japanese; the English conference papers on it (WCEE 2017 paper 3987, 13th WCEE paper 1633) refused automated download.
- **New Zealand C5, Concrete Buildings** [99]: the official PDF refused automated download. Likely relevant (assessment of existing RC buildings); open in a browser.
- **AeDES field manual** (JRC, EUR 22868 EN) [100]: no file on the record page.
- **ACI 224R-01** [94]: sold by ACI.
These can be read if someone downloads them in a browser and puts them in `data/` (not committed).

## What this means for our rules (proposal for the engineer; nothing changed in code)
1. The principle of the rules is backed [93][95][96]: direction against the element tells flexure from shear.
2. Exact bands are not given by any guideline; any band stays "illustrative until calibrated" and is for the engineer to set.
3. Two options for walls: widen "diagonal" towards FEMA 306's 20 to 55 degrees from vertical, or keep the gap (25 to 35) and send cracks in it to an expert as now.
4. The JBDPA width classes [98] are a documented, engineering-used reference for our width bands; worth showing to the engineer next to our 0.3, 1 and 3 mm.
5. FEMA 307 [95] supports what the matrix already does: pattern and position weigh as much as width.
