# Engineer map: when and where (review 26 Nov 2026)
- Time: always read from the photo's own record (EXIF DateTimeOriginal); shown to the engineer as "Taken".
- Place: only when the resident ticks "Share where the photo was taken" (needs "Share with the engineers' wall"). From the photo's GPS record; if the phone removed it, the resident may tap "Use my phone's location" (browser location, with the phone's permission).
- Shown only on the engineer wall (this laptop): one map with every shared place, coloured by level (Category 3 red, Expert needed violet, Category 2 amber, Category 1 blue, never green); a click opens the result. The result window shows time, coordinates and a link to OpenStreetMap.
- Never on the phone page, never in git. Kept in memory with the shared result (and in the wall's local export, `data/`, ignored).
- Code: `src/app/place.py`, `core.analyse(share_location=...)`, `result_blocks.where`, `second_look.wall_map`. Tests: `tests/test_review_26nov.py`.
