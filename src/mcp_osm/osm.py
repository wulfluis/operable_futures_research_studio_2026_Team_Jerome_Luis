"""Building facts from OpenStreetMap for one place: checked fields only, never free text.
Spec: spec/features/mcp_building_data.md. Outside data is untrusted (AGENTS.md rule 8): only validated numbers and
words from fixed lists leave this module; names, addresses and any other tag text are dropped.
"""

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

OVERPASS_URL = os.environ.get("SECOND_LOOK_OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSM_API_URL = os.environ.get("SECOND_LOOK_OSM_API_URL", "https://api.openstreetmap.org/api/0.6/map.json")
FALLBACK_RADIUS_MAX = 60  # metres: the OSM API is for editing, so the fallback asks only for a small box
USER_AGENT = "SecondLook-studio-prototype/0.1 (IAAC research project; building context for crack triage)"
TIMEOUT_S = 25
MIN_SPACING_S = 1.0  # at most one request per second to the public server
CACHE_TTL_S = 3600
RADIUS_MIN, RADIUS_MAX, RADIUS_DEFAULT = 1, 200, 30  # metres

SOURCE = {"name": "OpenStreetMap contributors", "licence": "ODbL 1.0", "url": "https://www.openstreetmap.org/copyright"}
NOT_IN_OSM = ["added_floors", "changed", "open_ground_floor", "occupants"]  # the building case asks these; OSM never has them
FIELDS = ("storeys", "year_built", "building_type", "material", "height_m", "footprint_m2")

BUILDING_TYPES = {"apartments", "residential", "house", "detached", "terrace", "commercial", "retail", "industrial",
                  "office", "school", "hospital", "church", "garage", "warehouse"}
MATERIALS = {"concrete", "brick", "cement_block", "stone", "wood", "steel", "metal", "glass", "adobe"}

_cache: dict[tuple, tuple[float, dict]] = {}
_last_request = [0.0]


# ---------- tag checks: a value passes only in a form we know ----------

def parse_levels(raw) -> int | None:
    """building:levels: a whole number from 1 to 200; '5;6', 'five' and '500' are unknown."""
    if not isinstance(raw, str) or not re.fullmatch(r"\s*\d{1,3}(\.0)?\s*", raw):
        return None
    n = int(float(raw))
    return n if 1 <= n <= 200 else None


def parse_year(raw, this_year: int | None = None) -> tuple[int | None, bool]:
    """start_date -> (year, approximate). 'YYYY', 'YYYY-MM' and 'YYYY-MM-DD' are exact; '~YYYY' and 'YYYYs' are approximate."""
    this_year = this_year or datetime.now(timezone.utc).year
    if not isinstance(raw, str):
        return None, False
    s = raw.strip()
    m = re.fullmatch(r"(~)?(\d{4})(s)?|(\d{4})-\d{2}(-\d{2})?", s)
    if not m:
        return None, False
    year = int(m.group(2) or m.group(4))
    if not 1000 <= year <= this_year:
        return None, False
    return year, bool(m.group(1) or m.group(3))


def clean_type(raw) -> str | None:
    if not isinstance(raw, str) or raw.strip() == "" or raw.strip() == "yes":
        return None
    word = raw.strip().lower()
    return word if word in BUILDING_TYPES else "other"


def clean_material(raw) -> str | None:
    if not isinstance(raw, str) or ";" in raw or raw.strip() == "":
        return None
    word = raw.strip().lower()
    return word if word in MATERIALS else "other"


def parse_height(raw) -> float | None:
    """height: metres, '12', '12.5' or '12 m'; 1 to 1000 m."""
    if not isinstance(raw, str):
        return None
    m = re.fullmatch(r"\s*(\d{1,4}(?:\.\d{1,2})?)\s*(m)?\s*", raw)
    if not m:
        return None
    h = float(m.group(1))
    return h if 1 <= h <= 1000 else None


# ---------- geometry on a local flat projection (metres around the query point) ----------

def to_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    return ((lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0, (lat - lat0) * 110_540.0)


def polygon_area_m2(xy: list[tuple[float, float]]) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def point_in_polygon(px: float, py: float, xy: list[tuple[float, float]]) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def distance_to_polygon_m(px: float, py: float, xy: list[tuple[float, float]]) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


def outline_xy(element: dict, lat: float, lon: float) -> list[tuple[float, float]] | None:
    """The way's outline in metres around (lat, lon); None when the answer has no usable geometry."""
    try:
        pts = [to_xy(lat, lon, float(g["lat"]), float(g["lon"])) for g in element["geometry"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts if len(pts) >= 3 else None


def select_building(elements: list[dict], lat: float, lon: float) -> tuple[dict, str, float] | None:
    """The outline that contains the point; otherwise the nearest. -> (element, 'contains' | 'nearest', metres)."""
    nearest = None
    for el in elements:
        if not isinstance(el, dict) or el.get("type") not in ("way", "relation"):
            continue
        xy = outline_xy(el, lat, lon)
        if xy is None:
            continue
        if point_in_polygon(0.0, 0.0, xy):
            return el, "contains", 0.0
        d = distance_to_polygon_m(0.0, 0.0, xy)
        if nearest is None or d < nearest[2]:
            nearest = (el, "nearest", d)
    return nearest


# ---------- the answer ----------

def summarise(element: dict, lat: float, lon: float, match: str, distance_m: float, queried: str,
              this_year: int | None = None) -> dict:
    """Checked fields only. The element's tags are read here and nowhere else; no tag text is copied into the answer."""
    tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    year, approximate = parse_year(tags.get("start_date"), this_year)
    xy = outline_xy(element, lat, lon)
    facts = {"storeys": parse_levels(tags.get("building:levels")),
             "year_built": year,
             "building_type": clean_type(tags.get("building")),
             "material": clean_material(tags.get("building:material")),
             "height_m": parse_height(tags.get("height")),
             "footprint_m2": round(polygon_area_m2(xy), 1) if xy else None}
    osm_id = element.get("id")
    return {"found": True,
            "osm": {"type": "relation" if element.get("type") == "relation" else "way",
                    "id": osm_id if isinstance(osm_id, int) else None},
            "match": match, "distance_m": round(distance_m, 1),
            "facts": {**facts, "year_approximate": approximate if year is not None else False},
            "unknown": [k for k in FIELDS if facts[k] is None],
            "not_in_osm": NOT_IN_OSM, "trust": "context_only", "source": SOURCE, "queried": queried}


def not_found(reason: str, queried: str, extra: dict | None = None) -> dict:
    return {"found": False, "reason": reason, "trust": "context_only", "source": SOURCE, "queried": queried, **(extra or {})}


def build_query(lat: float, lon: float, radius_m: int) -> str:
    return f'[out:json][timeout:{TIMEOUT_S}];way(around:{int(radius_m)},{lat:.6f},{lon:.6f})["building"];out tags geom;'


def _tls():
    """Certificates from certifi when installed: the python.org build on macOS ships without system certificates,
    so every HTTPS request failed there (live check, 11 Nov 2026)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_overpass(query: str, url: str | None = None, timeout: float = TIMEOUT_S) -> dict:
    """POST one query to the Overpass API, at most one request per second."""
    wait = MIN_SPACING_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()
    req = urllib.request.Request(url or OVERPASS_URL, data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                                 headers={"User-Agent": USER_AGENT}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_tls()) as r:  # noqa: S310 (a fixed https endpoint)
        return json.loads(r.read().decode("utf-8"))


def fetch_osm_api(lat: float, lon: float, radius_m: int, url: str | None = None, timeout: float = TIMEOUT_S) -> dict:
    """Fallback when Overpass fails (11 Nov 2026: all public Overpass servers at capacity): the official OSM API's
    map call for a small box around the point, at most one request per second."""
    r = min(int(radius_m), FALLBACK_RADIUS_MAX)
    dlat = r / 111_320.0
    dlon = r / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    bbox = f"{lon - dlon:.6f},{lat - dlat:.6f},{lon + dlon:.6f},{lat + dlat:.6f}"
    wait = MIN_SPACING_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()
    req = urllib.request.Request(f"{url or OSM_API_URL}?bbox={bbox}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_tls()) as resp:  # noqa: S310 (a fixed https endpoint)
        return json.loads(resp.read().decode("utf-8"))


def from_osm_api(data: dict) -> list[dict]:
    """The OSM API's answer in Overpass's shape: buildings with their outline as lat/lon points. A building mapped as
    a multipolygon relation gets the outline of its largest outer way. Tags are passed on unread (summarise checks
    them)."""
    elements = data.get("elements", []) if isinstance(data, dict) else []
    nodes = {e["id"]: e for e in elements if isinstance(e, dict) and e.get("type") == "node" and "lat" in e}
    ways = {e["id"]: e for e in elements if isinstance(e, dict) and e.get("type") == "way"}

    def geometry(way: dict) -> list[dict]:
        return [{"lat": nodes[n]["lat"], "lon": nodes[n]["lon"]} for n in way.get("nodes", []) if n in nodes]

    out = []
    for w in ways.values():
        if "building" in (w.get("tags") or {}):
            out.append({"type": "way", "id": w["id"], "tags": w["tags"], "geometry": geometry(w)})
    for rel in (e for e in elements if isinstance(e, dict) and e.get("type") == "relation"):
        if "building" not in (rel.get("tags") or {}):
            continue
        outers = [geometry(ways[m["ref"]]) for m in rel.get("members", [])
                  if m.get("type") == "way" and m.get("role") == "outer" and m.get("ref") in ways]
        outers = [g for g in outers if len(g) >= 3]
        if outers:
            out.append({"type": "relation", "id": rel["id"], "tags": rel["tags"], "geometry": max(outers, key=len)})
    return out


def check_input(lat, lon, radius_m) -> str | None:
    """None when the input is fine, else a message that names the field."""
    for name, value, lo, hi in (("lat", lat, -90, 90), ("lon", lon, -180, 180), ("radius_m", radius_m, RADIUS_MIN, RADIUS_MAX)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            return f"{name} must be a number from {lo} to {hi}"
    return None


def building_at(lat: float, lon: float, radius_m: int = RADIUS_DEFAULT, fetch=None, now=None,
                fetch_fallback=None) -> dict:
    """The facts of the building at a place, or why there are none. Never raises. Overpass first; when it fails,
    the official OSM API (fetch_fallback(lat, lon, radius_m); by default only when fetch is the real one)."""
    queried = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bad = check_input(lat, lon, radius_m)
    if bad:
        return not_found("bad_input", queried, {"message": bad})
    key = (round(lat, 5), round(lon, 5), int(radius_m))
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1]
    fallback = fetch_fallback if fetch_fallback is not None else (fetch_osm_api if fetch is None else None)
    try:
        data = (fetch or fetch_overpass)(build_query(lat, lon, int(radius_m)))
        if not isinstance(data, dict):
            raise ValueError("the answer is not a JSON object")
        elements = data.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError("elements is not a list")
        chosen, via = select_building(elements, lat, lon), "overpass"
    except Exception:  # network, timeout, bad JSON, a server error page: try the fallback, else unavailable
        if fallback is None:
            return not_found("unavailable", queried)
        try:
            chosen, via = select_building(from_osm_api(fallback(lat, lon, int(radius_m))), lat, lon), "osm_api"
        except Exception:  # noqa: BLE001  both routes failed
            return not_found("unavailable", queried)
    result = (not_found("no_building_within_radius", queried, {"via": via}) if chosen is None
              else {**summarise(chosen[0], lat, lon, chosen[1], chosen[2], queried), "via": via})
    _cache[key] = (time.monotonic(), result)
    return result
