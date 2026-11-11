"""Every test keeps away from the app's real folders: the 3D export folder (emptied when a wall starts) and the test
inbox (read by a background thread). 11 Nov: the page tests started the app and emptied data/results/cracks while
the real app was running."""

import pytest


@pytest.fixture(autouse=True)
def _private_app_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("SECOND_LOOK_EXPORT_DIR", str(tmp_path / "cracks"))
    monkeypatch.setenv("SECOND_LOOK_INBOX", str(tmp_path / "inbox"))
