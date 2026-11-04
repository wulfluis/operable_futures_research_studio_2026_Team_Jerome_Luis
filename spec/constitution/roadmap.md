# Roadmap

1. Calibrate: agree the risk matrix and thresholds per element (columns, beams, slabs, walls) with structural engineers. Fix the photo protocol (scale card, S, O and P shots, distance). Set consent and data rules.
2. Build and test: collect and label phone photos, train the detector and segmenter, compare against buildings an engineer has already assessed.
   Goal: flag every crack the engineer rated Category 3, and send only a few cases to an expert.
3. Widen the scope, only if the Phase 2 goal is met: more materials, ground fissures, multi-storey frames, offline use.

## Engineering partners
- Engineers Without Borders: labelling photos, signing off the matrix and thresholds, access to assessed buildings.
- A third-party structural engineer: independent check of thresholds and of the baseline ratings. Agreement under construction.
Both are needed for every task marked "at risk" below (`spec/decisions.md` D10).

## Dates (from the syllabus, 26/27 MaAI01)
| Date | Session | Target outcome |
|---|---|---|
| 29 Oct | Segmentation tree, full data map, repo + AGENTS.md | Agent working on real data as a first test |
| 2 Nov | Expert package sent to the professor (team milestone) | Contact with the thesis author |
| 5 Nov | Midterm submission deadline | |
| 6 Nov | **Midterm presentations** | |
| 12 Nov | Full SDD constitution, first feature with tests, first MCP connection attempt | A first feature that can be measured against the baseline |
| 16 to 20 Nov | Hackathon week (no studio) | Beta version |
| 26 Nov | Full plan/build harness, MCP connectors to all mapped sources | Agent works inside defined boundaries across several live sources |
| 3 Dec | Features + tests, comparison against the non-AI method | Measurable evidence of whether the tool beats the baseline, and where it fails |
| 10 Dec | Physical/digital proof of concept; limits and ethics check; UI build | Checked against physical reality; risks and responsibilities explicit |
| 14 Dec | UI, testing in realistic conditions, narrative draft | The story matches what the system does |
| 17 Dec | **Final presentations** | Live demo + pitch |
| by 24 Dec | Blog post on blog.iaac.net (at most 1 week after the end of the studio) | |

Final deliverables: problem & data brief, project repository (with the baseline comparison), operable prototype with interface and proof of concept, Lean Canvas, final presentation with a 10-slide pitch deck (`PITCH_DECK`), IAAC blog post (`docs/BLOG_POST`).

## Plan to the final presentation (26 Oct to 24 Dec 2026)

![Timeline](../../figures/09_timeline.png)

Weeks start on Monday; the figure labels each week with its Thursday. The plan follows the pipeline story (figure `01d_pipeline_story.png`): [n] is box n there. We build the crack line of the pipeline and the prototype around it; the field worker's app, the facade model and the element model are designed, not built (D30). "At risk" means the task depends on the engineering partners.

| Group | Task | Weeks | At risk |
|---|---|---|---|
| Photos | [2] Quality gate on public sets (done) | 1 | |
| Photos | [5] Pilot photos with the scale card, S, O and P (about 30 cracks, `spec/features/photo_protocol.md`) | 2 | |
| Photos | [2] Tune the quality gate on the pilot photos | 2 to 3 | |
| Models | [6] U-Net crack segmentation (CrackSeg9k, crack-seg) | 3 to 4 | |
| Models | [6] Width in mm from the scale card, with its uncertainty: first feature against the baseline (12 Nov) | 3 to 4 | |
| Models | [6] Pattern model, pre-trained on PHI-Net tasks 3 and 8 (stretch goal) | 4 to 5 | |
| Written rules | [4] [7] Dispatch rule and risk matrix v3 in code (done) | 1 | |
| Written rules | [8] Building risk score in code (`spec/features/building_score.md`) | 2 to 3 | |
| Written rules | [7] Calibrate the matrix and thresholds with the engineer | 3 to 5 | yes |
| Engineer | [9] Agree scope with Engineers Without Borders and the structural engineer | 1 to 2 | |
| Engineer | [9] Engineer-labelled photos and baseline ratings | 4 to 6 | yes |
| Engineer | [9] Comparison against the non-AI baseline (3 Dec) | 6 | yes |
| Prototype | First MCP: OpenStreetMap building data into the building case (12 Nov) | 3 | |
| Prototype | Beta: end to end on the pilot photos (hackathon, 16 to 20 Nov) | 4 | |
| Prototype | Plan/build harness, MCP to all mapped sources (26 Nov) | 5 | |
| Prototype | Recheck loop (crack IDs, overlay) and learn loop (corrections logged) | 5 to 6 | yes |
| Prototype | Proof of concept, limits and ethics check (10 Dec) | 7 | |
| Prototype | UI, testing in realistic conditions (10 to 14 Dec) | 7 to 8 | |
| Syllabus | Midterm deck and problem & data brief (submit 5 Nov, present 6 Nov) | 1 to 2 | |
| Syllabus | Lean Canvas | 6 to 7 | |
| Syllabus | 10-slide pitch deck, final presentation (17 Dec) | 7 to 8 | |
| Syllabus | Blog post (by 24 Dec) | 8 to 9 | |

Fallback: if no engineer ratings exist by 26 Nov, the 3 Dec comparison uses the pilot photos against our own width measurements only, the triage matrix stays illustrative, and the report says so.

## Open items
- Impact section: indicator with unit, baseline, target, evidence.
- Engineering partners: agreement with the third-party structural engineer is under construction.
