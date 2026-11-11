# Sources behind our crack-type rules (links)

The engineering sources we use to check the written crack-type rules (D43, `src/triage/pattern_rules.py`) and the width bands of the risk matrix. Each link has one or two sentences on why it is here. Full notes: `../crack_pattern_rules_sources.md` and `../fema_306.md`; numbers refer to `docs/sources.md`. Read 11 Nov 2026. PDFs are not kept in the repository; open the links.

## 1. The principle: flexure across the member, shear diagonal
- **FEMA 306, Basic Procedures Manual** (Applied Technology Council for FEMA, 1998) [93]: https://nehrpsearch.nist.gov/static/files/NIST/PB2001100763.pdf
  An engineering guideline for earthquake-damaged concrete walls. Section 5.3.1 (p. 83) states the rule our type rules are built on: flexural cracks run across the member (horizontal in a wall pier, vertical in a beam-like spandrel), shear cracks run diagonally.
- **FEMA 307, Technical Resources** (same authors, 1998) [95]: https://nehrpsearch.nist.gov/static/files/FEMA/PB2001101733.pdf
  The background volume to FEMA 306. Here because it says crack width alone is a poor indicator of damage and that location, orientation, number and distribution matter as much (p. 37), which backs why our risk matrix uses pattern and position, not width alone.

## 2. Angle limits
- **FEMA 306, §5.3.1** (same document as above) [93]: https://nehrpsearch.nist.gov/static/files/NIST/PB2001100763.pdf
  Gives shear cracks in walls as "typically at an angle of 35° to 70° from the horizontal" (20 to 55 degrees from the vertical). Here because it is the only guideline range we found, and it is wider than our "diagonal" band of 35 to 55 degrees from the vertical.
- **Yang, Chang, Wu (2019), Sensors 19(19) 4304** (peer-reviewed, CC BY) [96]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6806102/
  An image-analysis study that classifies cracks within 22.5 degrees of the horizontal as flexural and the rest as shear. Here because it is the only fixed angle cut-off we found, close to our 25-degree band; the authors say it does not apply to every structure.

## 3. Crack width classes (Japan)
- **JBDPA damage classes, quoted in Quan, Choi, Jin (2025), Materials 18(19) 4638** (peer-reviewed, CC BY) [97][98]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12526491/
  Table 4 gives the Japanese guideline's damage classes for RC members: under 0.2 mm, 0.2 to 1 mm, 1 to 2 mm, over 2 mm, then buckled bars. Here as an engineering-used reference to compare with our matrix's width bands (0.3, 1 and 3 mm, illustrative); the guideline itself (JBDPA, 1991, revised 2001 and 2015) is in Japanese and was not read directly.

## Not yet read (worth reading)
- **New Zealand, Seismic Assessment of Existing Buildings, Part C5 Concrete Buildings** [99]: https://building.govt.nz/assets/Uploads/building-code-compliance/b-stability/b1-structure/seismic-assessment/c5-concrete-buildings.pdf
  Official guidance for assessing existing RC buildings; likely relevant, but the site refused automated download. Open in a browser.
