# Mission

Second Look is a human-in-the-loop triage system for cracks in reinforced concrete. It checks the photo and the material, finds and measures the crack in millimetres, classifies its pattern, and applies written, auditable risk rules, returning a level with its confidence. An engineer confirms every Category 3 and every referral, and their corrections retrain the models. It never issues a safety clearance.

## Problem
In dense, ageing neighbourhoods, residents see cracks every day but cannot tell a harmless one from a sign of structural failure.
Engineers are too few to inspect every building. Tripoli, February 2026: two blocks collapsed on 8 February (14 to 15 dead, AFP and Reuters),
and 114 buildings were ordered evacuated. Rana Plaza, Dhaka, 2013: cracks were visible the day before the collapse.

## Why AI is needed
- **The number of buildings outgrows the number of engineers.** Checking a building today means an engineer on site, looking at it and posting it green, yellow or red (ATC-20 [103]). When many buildings need it at once this does not scale: after the February 2023 earthquakes, Türkiye completed damage assessments for 1,712,182 buildings in 11 provinces within a month [102]. Ageing neighbourhoods face the same shortage without a disaster.
- **The step that takes an expert's eye is reading the crack in the photo.** Finding which pixels are crack, measuring its width in millimetres from a reference, telling its direction against the element, and recognising the element cannot be written as simple rules; learned models do it, the same way for every photo, in seconds (U-Net pre-trained on ImageNet and fine-tuned on public crack photos, D48; element model, D52; SAM 3.1 for the parts, D62).
- **What stays without AI:** the risk level comes from written, auditable rules (rule 5), and an engineer confirms every Category 3 and every referral. A surface photo cannot show what is inside the structure (out of scope), so the tool ranks where an engineer should go first; it never says a building is safe.
- **How we know it helps:** each feature is measured against the non-AI way of doing it. First, the width in mm against a crack-width gauge (`docs/width_gauge/`, `src/measure/width_check.py`); then the whole triage against an engineer's visual check of the same photos (`spec/features/evaluation.md`).

## Mission
Give residents and field workers a fast, cheap early-warning triage from a phone photo, and give scarce engineers a ranked list of which buildings to visit first.

## Scope
Cracks in reinforced concrete elements (walls, columns, beams, slabs). Other materials go to an engineer (`spec/decisions.md` D17). The unit of analysis is the building, seen at three scales (S, O, P) and over time (D21).

## Users
- Residents of old buildings, field workers and community leaders: take the photo and receive a level, a reason and a next step.
- Structural engineers: review every Category 3 and every referral, confirm or correct it, and so improve the models (`spec/features/pipeline.md`). Engineers and municipalities also use the ranked list to decide where to inspect first.

## Success (works, used, matters)
- Works: beats the non-AI baseline (an engineer's visual check) on held-out, engineer-labelled photos.
- Used: runs inside a real workflow and people trust it.
- Matters: impact indicators move. Indicator, unit, baseline, target and evidence are still to be defined (see `roadmap.md`).

## Failure (stop condition)
We stop or redesign if the tool misses Category 3 cases that engineers would flag.

## Out of scope
Permits, legal clearances, foundation or rebar assessment, any claim a surface photo cannot support. Materials other than reinforced concrete are not assessed (they are referred).
