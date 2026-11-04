import io

import numpy as np
import pytest
from PIL import Image

from app.core import (MAX_BYTES, ConsentMissing, PhotoRejected, Wall, analyse, make_qr, tunnel_url,
                      wall_key_ok)


class FakeModel:
    """Stands in for the U-Net: 'sees' a crack wherever the photo is dark."""

    def __call__(self, x):
        return (x.mean(1, keepdim=True) < -1.0).float() * 20 - 10


def _jpeg(crack=True, fmt="JPEG"):
    rng = np.random.default_rng(0)
    a = rng.integers(110, 190, (600, 800, 3)).astype(np.uint8)
    if crack:
        a[298:302, 50:750] = 10
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, fmt)
    return buf.getvalue()


def test_no_consent_no_analysis():
    wall = Wall()
    with pytest.raises(ConsentMissing):
        analyse(_jpeg(), consent=False, share=True, wall=wall, model=FakeModel())
    assert wall.items() == []


def test_only_shared_results_reach_the_wall():
    wall = Wall()
    r = analyse(_jpeg(), consent=True, share=False, wall=wall, model=FakeModel())
    assert r.statement == "Crack found" and wall.items() == []
    analyse(_jpeg(crack=False), consent=True, share=True, wall=wall, model=FakeModel())
    assert [s for s, _ in wall.items()] == ["No crack found"]
    assert wall.counts()["No crack found"] == 1 and wall.counts()["Crack found"] == 0


def test_wall_keeps_the_newest_and_small_copies():
    wall = Wall(size=3)
    for i in range(5):
        wall.add(f"s{i}", Image.new("RGB", (2000, 1500)))
    assert [s for s, _ in wall.items()] == ["s4", "s3", "s2"]
    assert max(Image.open(io.BytesIO(wall.items()[0][1])).size) <= 480


@pytest.mark.parametrize("data, why", [
    (b"not an image", "could not be read"),
    (b"x" * (MAX_BYTES + 1), "10 MB"),
], ids=["not_an_image", "too_large"])
def test_bad_uploads_are_refused(data, why):
    with pytest.raises(PhotoRejected, match=why):
        analyse(data, consent=True, share=True, wall=Wall(), model=FakeModel())


def test_other_formats_are_refused():
    with pytest.raises(PhotoRejected, match="JPEG or PNG"):
        analyse(_jpeg(fmt="BMP"), consent=True, share=False, wall=Wall(), model=FakeModel())


def test_wall_key():
    assert wall_key_ok("abc", "abc")
    assert not wall_key_ok("abd", "abc") and not wall_key_ok(None, "abc") and not wall_key_ok("", "")


def test_tunnel_link_is_found_in_its_log():
    line = "2026-10-30T10:00:00Z INF |  https://quiet-river-bench-oak.trycloudflare.com                |"
    assert tunnel_url(line) == "https://quiet-river-bench-oak.trycloudflare.com"
    assert tunnel_url("INF Starting tunnel") is None


def test_qr_is_saved(tmp_path):
    img = make_qr("https://example.trycloudflare.com", tmp_path / "qr.png")
    assert (tmp_path / "qr.png").exists() and img.size[0] == img.size[1] > 100


def test_launcher_sees_a_port_already_in_use():
    import socket
    from app.launch import port_in_use
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert port_in_use(port)
    assert not port_in_use(port)
