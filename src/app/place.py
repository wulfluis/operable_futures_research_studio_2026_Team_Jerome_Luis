"""When and where a photo was taken (review 26 Nov 2026), for the engineer's map on the wall.

The time is read from the photo's own record (EXIF DateTimeOriginal). The location only when the resident ticks
"Share where the photo was taken": from the photo's GPS record, or, when the phone removed it (most phone browsers
do), from the phone's own location with the resident's permission. A household's location is personal (rule 9):
it is kept in memory with the shared result, shown only on the engineer's wall, never on the phone page or in git.
Spec: spec/features/engineer_map.md."""

from __future__ import annotations

import io
from datetime import datetime

from PIL import Image

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
DATETIME_ORIGINAL, DATETIME = 36867, 306


def _deg(v) -> float | None:
    try:
        d, m, s = (float(x) for x in v)
        return d + m / 60 + s / 3600
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read(data: bytes) -> dict:
    """{"taken_at": "YYYY-MM-DD HH:MM" or None, "lat": float or None, "lon": float or None}, from the photo's record."""
    out = {"taken_at": None, "lat": None, "lon": None}
    try:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        with Image.open(io.BytesIO(data)) as im:
            ex = im.getexif()
    except Exception:  # noqa: BLE001  a photo without a readable record has no time or place
        return out
    raw = ex.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or ex.get(DATETIME)
    if raw:
        try:
            out["taken_at"] = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    gps = ex.get_ifd(GPS_IFD)
    lat, lon = _deg(gps.get(2)), _deg(gps.get(4))
    if lat is not None and lon is not None and not (lat == 0 and lon == 0):
        out["lat"] = round(-lat if str(gps.get(1, "N")).upper().startswith("S") else lat, 6)
        out["lon"] = round(-lon if str(gps.get(3, "E")).upper().startswith("W") else lon, 6)
    return out


def valid(lat, lon) -> bool:
    try:
        return -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180 and not (float(lat) == 0 and float(lon) == 0)
    except (TypeError, ValueError):
        return False
