# Sources for the midterm feedback (links)

Research of 11 Nov 2026 on the open points of the midterm review (reviewers R1 to R4, action plan items 1 to 9). Each link has one or two sentences on what it is and why it is here. Numbers refer to `docs/sources.md`. Every figure below was read on the page itself; "abstract only" says where the full text could not be opened.

## Context and storytelling (action items 1 and 2: post-disaster triage, "100,000 buildings, 100 engineers")
- **Türkiye Earthquakes Recovery and Reconstruction Assessment** (Presidency of Strategy and Budget of Türkiye, with UN and EU support, March 2023) [102]: https://www.sbb.gov.tr/wp-content/uploads/2023/03/Turkiye-Recovery-and-Reconstruction-Assessment.pdf
  Official report on the February 2023 earthquakes. Page 32: by 6 March 2023, damage assessments were completed for 1,712,182 buildings in 11 provinces (35,355 collapsed, 17,491 to be demolished urgently, 179,786 severely damaged). Here as a real, sourced scale for the reviewers' scenario of many buildings and few engineers.
- **Build Change** (NGO, founded 2004) [107]: https://buildchange.org/about
  An organisation whose engineers and builders retrofit homes in disaster-prone countries; by its own figures more than 280,000 buildings built or retrofitted. Here as an example of the partner R2 and R3 asked for: one that can not only inspect but also repair.

## How engineers assess today (R4: "show how an engineer assesses a crack today")
- **ATC-20 Building Safety Evaluation** (Applied Technology Council, 1989; addendum 1995; case studies 1997) [103]: https://atcouncil.org/atc-20
  The standard US procedure for rapid and detailed post-earthquake safety evaluation by volunteer engineers and inspectors, posting buildings green (inspected), yellow (restricted use) or red (unsafe); forms and placards are free to download. Here as the non-AI workflow our triage would sit in front of, and as the baseline to describe in the methodology.
- **FEMA 306 and FEMA 307** [93][95]: see `../crack_rules/README.md`. The engineering guideline for judging crack direction and width in damaged concrete walls.

## Existing work and baselines (R1.4, R3.4, action item 5)
- **Chachra et al. (2022), Scientific Reports, CC BY** [104]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9142599/
  A deep-learning model trained on about 6,500 labelled images finds photos of damaged buildings in social-media posts after earthquakes (damaged-building recall 80.5 %, precision 77.4 % on its test set). Here as prior work on crowdsourced post-earthquake photos: it finds damaged buildings, not crack types, which is the gap Second Look addresses.
- **OmniCrack30k** [72] and our reading list: `../reading_list.md`. The existing trained crack model to test as a baseline against our U-Net (R3: "start from an existing model").

## Scale and capture (R1.1, R2.2, action item 7: is a card realistic, which cracks are visible at which distance)
- **Seo, Kim, Chung (2021), Journal of the Korean Geosynthetics Society** (KICT; Korean with English abstract) [105]: https://koreascience.or.kr/article/JAKO202111262780836.pdf
  Tests smartphone cameras for crack widths at a distance: with a 64 MP camera and a 2x add-on lens, widths of 0.3, 0.5 and 1 mm could be told apart within 2 m; without a lens, fine widths needed one. Here because it gives a measured answer to "which crack sizes are visible at which field of view".
- **Jeng, Chao, Chen (2025), Eng (MDPI)** [106] (abstract only): https://www.mdpi.com/2673-4117/6/6/122
  Proposes a standard test rig (crack-width calibration plate on a simulated wall, phone held at set distances) to check how accurately phone apps measure crack widths. Here as a method for our own width check against known widths (the baseline feature due 12 Nov).

## Not found
- No source found that gives the number of engineers deployed in Türkiye 2023 next to the number of buildings; the "100 engineers" in the reviewers' scenario stays an illustration.
- Nepal 2015 survey figures (volunteer engineers, number of surveys) were found only in secondary pages that could not be read; not used.
