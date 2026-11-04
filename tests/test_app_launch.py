import io
import json

import app.launch as launch


def fake_api(tunnels):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return lambda url, timeout=2: R(json.dumps({"tunnels": tunnels}).encode())


def test_fixed_link_is_seen_only_when_ngrok_serves_it(monkeypatch):
    monkeypatch.setattr(launch.urllib.request, "urlopen", fake_api([{"public_url": launch.FIXED_URL}]))
    assert launch.ngrok_online()
    monkeypatch.setattr(launch.urllib.request, "urlopen", fake_api([{"public_url": "https://other.ngrok-free.dev"}]))
    assert not launch.ngrok_online()


def test_no_ngrok_running_means_offline(monkeypatch):
    def refuse(url, timeout=2):
        raise OSError("connection refused")
    monkeypatch.setattr(launch.urllib.request, "urlopen", refuse)
    assert not launch.ngrok_online()


def test_fixed_link_is_https():
    assert launch.FIXED_URL.startswith("https://") and not launch.FIXED_URL.endswith("/")
