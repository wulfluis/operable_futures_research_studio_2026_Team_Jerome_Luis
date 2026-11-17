# Questions for Enrique Corres Sojo, from his thesis

Prepared 16 Nov 2026 for the email of 17 Nov. Source: Corres Sojo E., *Contributions to rebar-to-concrete interaction and its structural implications for design and monitoring applications*, EPFL thesis n° 10772, 2024, doi:10.5075/epfl-thesis-10772 [46]. Read again in full on 16 Nov from the EPFL repository: chapter 1, chapter 4 (4.1, 4.2, 4.4, 4.7), chapter 5 in full, the conclusions of chapters 2 and 3, chapter 6. Our earlier notes: `corres_sojo_2024.md`.

Page numbers are the thesis's own printed numbers (checked against the PDF: printed page = PDF page − 17). Each question says what in the thesis prompted it and why it matters for Second Look, so the email shows we read the work.

## What to tell him about us, in two lines

Second Look triages cracks in ageing reinforced-concrete homes from two phone photos: the crack close up with a bank card as the scale, and the whole element. A crack model (U-Net, trained partly on the Özgenel dataset he also used [49]) marks the crack; the width is measured across it at every point with ±1 pixel as the stated uncertainty, following his 3-pixel and 6-to-8-pixel rule (ch. 5.2.4); written rules give one of five levels, and an engineer confirms every serious case. It never says a building is safe.

## The five to send (most useful to us, most specific to his work)

1. **How far can the ±1 pixel hold on thin, old, covered cracks?** His ±1 pixel comes from a CNC-cut panel with painted lips (ch. 5.2.3, p. 132) and the cleaned Chillon crack, where one point read 1.18 mm against 0.73 mm by hand (Table 5.1, p. 145). The algorithm he used reports 2 to 14 pixels of error for openings of 0.4 to 4 mm (Pantoja-Rosero et al. 2022, quoted on p. 130). Our cracks are mostly 0.1 to 1 mm, dusty, often in paint or plaster. *Question:* would you expect the photo width to be biased upward on eroded or dusty lips (as at point D), and would you widen our uncertainty beyond ±1 pixel, or ask for a cleaner surface first?

2. **A crack in plaster or render, over concrete.** He shows the crack is narrower at the bar than at the surface, more so with a larger cover (ch. 4.4, p. 106: "the crack width at the bar location is likely smaller than the crack width observed on the concrete surface, particularly for large covers"). Most of our photos show cracks in a render layer over the concrete. *Question:* is there a rule of thumb linking a crack width seen in render to the crack in the concrete below, or should a width read through render never lower a level on its own?

3. **Which visible signs point to a brittle mechanism, from one photo?** He writes that small openings may not mean enough resistance where fragile failure modes govern (ch. 1.1, p. 4; ch. 5.1, p. 129) and that kinematics show which shear-transfer mechanisms act (p. 129). We have no DIC, only the crack's shape, direction, place on the element, and possibly a step between the lips. *Question:* which of these would you trust to flag a possibly brittle (shear) case for an engineer, and is "a diagonal crack at 35° to 55° in a wall or near a support" a defensible first filter? A real case from our tests: a fine crack starting at the top corner of a door opening, near vertical then turning diagonal.

4. **A low-cost version of the marker technique for residents.** His markers on both lips track long-term movement without DIC (ch. 5.3, pp. 135 to 143); with a single uncalibrated camera, earlier marker studies reach about ±1 pixel (p. 131); temperature must be corrected and long-term trends need more than a year (pp. 136, 147); averaging 10 images reduces noise and bias (p. 147). We would like residents to stick printed markers on both sides of the crack and photograph them with the bank card every few months. *Questions:* with a phone and printed markers, what is the smallest change worth reporting; how many markers, and how placed (he advises a large d1 and a small d2, p. 143); and would a burst of 10 phone frames give the averaging benefit?

5. **Known-width images to test our measurement.** His laboratory series has known openings from 0.1 to 2.5 m away, taken with a 16 MP smartphone (OnePlus 6) and a Nikon D800 (ch. 5.2.3, Fig. 5.2). *Question:* could we use those images, or the Chillon images with the hand measurements, to check our width measurement against known values? We would cite them and share the results.

## More questions, if the conversation continues

**Measuring from photos (ch. 5.2)**

6. **One scale reference.** He recommends several visual references on large surfaces because the pixel size is not uniform (p. 134). We use one bank card beside the crack and correct the perspective from its four corners. Up to what distance from the card along the crack would he trust a single reference?
7. **Flash.** He recommends photos with and without flash, depending on how rough the surface is (p. 134). Did one of the two clearly give better crack detection? Should our app ask for both?
8. **Sliding from one photo.** The direct method gives sliding to about ±2 pixels (p. 134). A step between the crack lips is one of our warning signs. From a phone photo, would he trust a measured sliding enough to raise a level, and what sliding is meaningful for a shear crack?
9. **Width across the crack, point by point.** We read the width perpendicular to the crack's centre line at every point and smooth it over about 3 pixels, rather than fitting rigid lip movements in 256 × 256 windows (full-edge or finite-edge, p. 134, with windows of about twice the largest aggregate). Is that a reasonable simplification for opening only?
10. **What fools the detector.** His future work lists corrosion traces, water, sediment marks and moss as untested (ch. 6.2, p. 161). In our tests the crack model also marks electric cables, frames and plants. Has he seen typical false detections on real structures?

**Reading cracks for safety (ch. 1, 2, 4)**

11. **Cracks along a bar.** He shows that cracks parallel to the bar (splitting and spalling) reduce bond, and proposes a reduction as a function of their opening (ch. 2.5, conclusions 8 and 9, p. 43). For triage, with the bar diameter unknown from a photo, at what opening would a crack along a bar become a concern? Is assuming a small bar diameter (say 12 mm) a safe default?
12. **Crack spacing from a photo.** Bar stress can be estimated from the crack width only with the crack spacing, cover, bar diameter and bond (ch. 4.2, p. 99; conclusion 5, p. 120). A photo of several parallel cracks gives their spacing. Would a coarse bound on the bar stress be useful for screening, or more misleading than helpful? (Today we do not attempt it.)
13. **Direction of tension.** On the Chillon crack the movement was mostly perpendicular to the crack, as expected in a shear panel (p. 144). We draw "pulled apart perpendicular to the crack" on our 3D drawing. When would that simplification mislead?

**Over time and thresholds**

14. **Rechecks between 1 and 3 months.** He says permanent change needs measurements over more than a year to separate daily and seasonal movement (p. 147). For a resident's recheck after a few months, what rule avoids false alarms: same time of day and season, or only growth above a set multiple of the uncertainty?
15. **Width limits for existing buildings.** The thesis gives no safety thresholds; it cites the Swiss inspection guide (OFROU/ASTRA 62016), the Spanish guide (DGC 2012) and Zaborac et al. 2019 [50][51][52]. Which would he use for ageing residential reinforced concrete, not bridges?
16. **Review.** Would he, or someone at IBETON, look at our 30 to 40 calibration cards (element × pattern × width × warning signs)?

## His own open points we can mention (ch. 6.2, pp. 159 to 161)

- Initial crack characterisation in adverse conditions (corrosion traces, water, sediment marks, moss) is untested: our field photos could be a test set for him.
- Long-term marker monitoring was validated over one day only.
- He wishes the techniques were in "a single open-access program" for practising engineers: Second Look is open and does the direct detection and measurement in one app, a possible point of exchange.

## Do not claim in the email

- Any width threshold as his: the thesis gives none (our notes, finding 19).
- That our ±1 pixel is validated: it is his result on his set-up; ours is still to be checked with the width-check photos (`spec/features/measurement.md`).
- Any stress value: the thesis says bar stress needs spacing, cover, diameter and bond, which a photo does not give.
