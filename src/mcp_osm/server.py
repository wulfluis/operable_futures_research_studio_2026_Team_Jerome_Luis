"""MCP server: building facts from OpenStreetMap for the building case. Spec: spec/features/mcp_building_data.md.

Run it over stdio (any MCP client can start it):
    py src/mcp_osm/server.py
"""

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_osm import osm

server = MCPServer(
    "second-look-osm",
    instructions=("Facts about the building at a place, from OpenStreetMap. The facts are context only: they never "
                  "lower a level and they are suggestions for a field worker to confirm. Fields that are not known "
                  "are listed under 'unknown'."))


@server.tool()
def building_at(lat: float, lon: float, radius_m: int = osm.RADIUS_DEFAULT) -> dict:
    """The building at a latitude and longitude: storeys, year built, type, material, height and footprint area.

    Only checked fields are returned (no names, addresses or free text). radius_m is how far to look, 1 to 200 m.
    found is false when there is no building in range or the source is unavailable (see reason)."""
    return osm.building_at(lat, lon, radius_m)


if __name__ == "__main__":
    server.run("stdio")
