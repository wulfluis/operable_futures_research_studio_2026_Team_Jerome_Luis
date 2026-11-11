# Feature: building data over MCP, first connection (`src/mcp_osm/`)

Status: specified and coded on 10 Nov 2026 for the studio session of 12 Nov (first MCP connection attempt). Decision D49. Source: OpenStreetMap. Next (26 Nov): the agent harness that calls it, and more sources.

## Purpose
The building case (`spec/features/building_score.md`) asks the field worker or the resident about the building: age, floors added, changes, open ground floor, typology, storeys, occupants. Some of it may already be mapped. An MCP server gives an agent one standard way to ask "what is known about the building at this place?" and to get back only checked facts, as suggestions, with their source.

## What MCP is here
MCP (Model Context Protocol) is an open standard for connecting an agent to outside tools and data. Our server (`src/mcp_osm/server.py`, the official `mcp` Python SDK) offers one tool, `building_at`. The agent is the client. The server does the lookup; it decides nothing.

## Source
OpenStreetMap, through the public Overpass API, ODbL 1.0. Every answer carries "OpenStreetMap contributors" and the licence link; a dossier that stores it keeps them. Good-citizen rules: one request per second at most, a 25 s timeout, a client name that says what we are (no personal contact data), an answer cache of one hour, no bulk queries. The endpoint can be changed with `SECOND_LOOK_OVERPASS_URL` (for example a self-hosted server later).

## The tool: `building_at(lat, lon, radius_m=30)`
- Searches building outlines (ways tagged `building`) within `radius_m` (1 to 200 m) of the point. Building outlines that are multipolygon relations are not read (limit).
- Picks the outline that contains the point; if none does, the nearest one. It says which (`match`: `contains` or `nearest`) and the distance in metres.
- Returns **only checked fields**:

| Field | From the tag | Check |
|---|---|---|
| `storeys` | `building:levels` | a whole number from 1 to 200, else unknown (so "5;6" and "five" are unknown) |
| `year_built` | `start_date` | a 4-digit year from 1000 to this year, optionally with month and day; "~1980" and "1970s" give the year with `year_approximate` true; anything else is unknown |
| `building_type` | `building` | one word from a fixed list (apartments, residential, house, detached, terrace, commercial, retail, industrial, office, school, hospital, church, garage, warehouse); "yes" is unknown; any other word becomes "other" |
| `material` | `building:material` | exactly one word from a fixed list (concrete, brick, cement_block, stone, wood, steel, metal, glass, adobe); another word becomes "other"; several values are unknown |
| `height_m` | `height` | a number from 1 to 1000 m, else unknown |
| `footprint_m2` | the outline's geometry | computed by us (shoelace formula on a local flat projection) |

- Also returned: `unknown` (which of the six are missing), `not_in_osm` (added floors, changes, open ground floor, occupants: OpenStreetMap never has them, so they stay unknown), `source`, `queried` (UTC time) and `trust: "context_only"`.
- No building in range: `found: false`, `reason: "no_building_within_radius"`. Network error, timeout or a bad answer: `found: false`, `reason: "unavailable"`. The tool never raises into the agent.
- Bad input (a latitude outside -90 to 90, a longitude outside -180 to 180, a radius outside 1 to 200): an error result that names the field.

## What the server never returns (AGENTS.md rules 8 and 9)
- **No free text.** Names, descriptions, notes, operators, websites and any other tag value never leave the server. Tag text is written by anyone on the internet and could carry instructions meant for an agent; only validated numbers and words from fixed lists pass.
- **No addresses and no names** (`addr:*`, `name`): they identify households.

## How the building case may use it (`src/mcp_osm/case.py`)
- OpenStreetMap facts are **suggestions** for the field worker to confirm or correct. They are never stored as answers.
- **They may raise a risk flag but never clear one.** A year more than 50 years ago (`AGE_FLAG_YEARS`, the same assumption as in `building_score.md`) may raise the age flag. A recent year, a missing year and a wrong-looking year change nothing: "unknown" already counts as a flag, and only a person's answer can clear it. This keeps the rule that outside data never lowers a level.
- `building_case_suggestions(result, today_year)` returns `suggestions` (storeys, year, material, type, footprint), `flags_it_may_raise` and `clears_no_flag: true`.

## Limits
- Coverage is likely thin in informal neighbourhoods, often only an outline (assumption; not measured). Many fields will be unknown, and that is not a problem: unknown counts as a flag.
- Tags can be wrong, outdated or vandalised. `building:levels` follows OpenStreetMap's own convention for what counts as a storey. Nothing is verified against the building.
- The footprint comes from the mapped outline, not from a survey.
- Live checks depend on a public server that may be slow or refuse requests.

## Tests (`tests/test_mcp_osm.py`, no network: the fetch function is replaced)
Parsing of levels, years, types, materials and heights, including ambiguous and absurd values; the footprint of a 20 x 30 m rectangle within 1 %; the outline that contains the point beats a nearer neighbour; nearest within the radius; nothing in range; free text, names and addresses never appear in the answer, including a tag that contains an instruction; a network error becomes `unavailable`; bad coordinates and radius; the cache and the one-second spacing; the server lists `building_at` and returns the same answer through MCP; the case mapping raises the age flag for an old building, raises nothing for a new or unknown year, and never clears a flag.

## Not yet
The agent harness that decides when to call it (26 Nov); further sources (the Haiti and Nepal building surveys, national registers); the building risk score itself (`building_score.md`, not coded); a check of how much of a real informal neighbourhood is mapped.

## Fallback to the official OSM API (11 Nov 2026, D64)
On the evening of 11 Nov all public Overpass servers failed even a one-node query (502 "at capacity", 504, timeouts), while OpenStreetMap itself answered. So when Overpass fails, `building_at` asks the official OSM API's map call (`api/0.6/map.json`) for a small box around the point (at most 60 m from it: that API is meant for editing and wants small, rare requests) and reads the same checked fields. A building mapped as a multipolygon relation gets the outline of its largest outer way. Every answer says which route it came from (`"via": "overpass"` or `"osm_api"`). Tests use fakes and never reach the network.
- Live check, 11 Nov 2026 about 19:30, through the MCP tool: Overpass unavailable, fallback found the test landmark (Basílica de la Sagrada Família, relation 9194723, building type church).
- Limit: for a large building mapped as a relation, the box holds only part of its outline, so the footprint can be far too small (302.6 m² for the test landmark) and the match can read "nearest" although the point is inside. Storeys, type, material and year are unaffected.
- Also fixed that evening: the laptop's Python could not verify HTTPS certificates; requests now use certifi's certificates.
