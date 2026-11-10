"""How the building case may use OpenStreetMap facts. Spec: spec/features/mcp_building_data.md.

The facts are suggestions for the field worker. They may raise a risk flag but never clear one: a year more than
AGE_FLAG_YEARS ago may raise the age flag; a recent, missing or odd-looking year changes nothing, because unknown
already counts as a flag and only a person's answer clears it (AGENTS.md rule 8, spec/features/building_score.md).
"""

from datetime import datetime, timezone

AGE_FLAG_YEARS = 50  # assumption, as in spec/features/building_score.md


def building_case_suggestions(result: dict, today_year: int | None = None) -> dict:
    """result: the answer of building_at. -> suggestions, the flags it may raise, and that it clears none."""
    today_year = today_year or datetime.now(timezone.utc).year
    out = {"suggestions": {}, "flags_it_may_raise": [], "clears_no_flag": True,
           "needs_a_person": ["age", "added_floors", "changed", "open_ground_floor", "typology", "storeys", "occupants"]}
    if not isinstance(result, dict) or not result.get("found"):
        return out
    facts = result.get("facts", {})
    out["suggestions"] = {"storeys": facts.get("storeys"), "year_built": facts.get("year_built"),
                          "material": facts.get("material"), "building_type": facts.get("building_type"),
                          "footprint_m2": facts.get("footprint_m2"), "source": "openstreetmap", "status": "suggestion"}
    year = facts.get("year_built")
    if isinstance(year, int) and today_year - year > AGE_FLAG_YEARS:
        out["flags_it_may_raise"].append("age")
    return out
