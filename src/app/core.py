"""Class demo logic without the web page: consent, the live wall, the tunnel link, the QR. Spec: spec/features/class_demo.md."""

from __future__ import annotations  # "threading.Lock | None" needs this before Python 3.13

import hmac
import io
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import qrcode
from PIL import Image

from segmentation.predict import Result, run

MAX_BYTES = 10 * 1024 * 1024  # one photo, 10 MB
WALL_SIZE = 40  # results kept on the live wall (all statements together)
WALL_THUMB = 480  # px, long side of a wall copy
STATEMENTS = ("Crack found", "No crack found", "Expert needed", "Retake")
NOT_A_SAFETY_CHECK = "This is not a safety check. An engineer confirms every crack that matters."
Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse absurdly large images (a 48 MP phone photo still passes)


class ConsentMissing(Exception):
    pass


class PhotoRejected(Exception):
    pass


@dataclass
class Wall:
    """Shared results for the projector, in memory only; gone when the app stops."""
    size: int = WALL_SIZE
    _items: deque = field(default_factory=deque)
    _counts: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, statement: str, image: Image.Image) -> None:
        """Keeps a small JPEG, encoded once here, so the wall's refresh does not re-encode every photo."""
        thumb = image.convert("RGB")
        thumb.thumbnail((WALL_THUMB, WALL_THUMB))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=85)
        with self._lock:
            self._items.appendleft((statement, buf.getvalue()))
            while len(self._items) > self.size:
                self._items.pop()
            self._counts[statement] += 1

    def items(self) -> list[tuple[str, bytes]]:
        with self._lock:
            return list(self._items)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {s: self._counts[s] for s in STATEMENTS}


def analyse(data: bytes, consent: bool, share: bool, wall: Wall, model, device: str = "cpu",
            gpu: threading.Lock | None = None) -> Result:
    """One upload: consent first, then size and format, then the pipeline. Only pixels are read, never metadata."""
    if not consent:
        raise ConsentMissing("Tick the consent box first.")
    if len(data) > MAX_BYTES:
        raise PhotoRejected("The photo is larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("JPEG", "PNG", "MPO"):
                raise PhotoRejected("Send a JPEG or PNG photo.")
            img.load()
            photo = img.copy()
    except (OSError, Image.DecompressionBombError) as e:
        raise PhotoRejected("This file could not be read as a photo.") from e
    with gpu or threading.Lock():
        result = run(photo, model, device)
    if share:
        wall.add(result.statement, result.overlay)
    return result


def wall_key_ok(given: str | None, key: str | None) -> bool:
    return bool(given) and bool(key) and hmac.compare_digest(given, key)


TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def tunnel_url(line: str) -> str | None:
    """The quick tunnel prints its random https link once in its log; find it."""
    m = TUNNEL_URL.search(line)
    return m.group(0) if m else None


def make_qr(url: str, path: Path | None = None) -> Image.Image:
    img = qrcode.make(url, box_size=12, border=2).get_image().convert("RGB")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img
