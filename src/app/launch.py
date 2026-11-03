"""Start the class demo: the web page, a temporary https link, the QR code and the live wall. Spec: spec/features/class_demo.md.

Usage, from the repository root (laptop plugged in and online):
    py src/app/launch.py
Stop with Ctrl+C: the link dies and every photo in memory is gone.
"""

import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from app.core import make_qr, tunnel_url

PORT = 8501
QR = ROOT / "data" / "results" / "app" / "qr.png"


def find_cloudflared() -> str:
    found = shutil.which("cloudflared")
    for candidate in (found, r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                      r"C:\Program Files\cloudflared\cloudflared.exe"):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("cloudflared not found: winget install --id Cloudflare.cloudflared")


def start_tunnel(timeout: float = 60) -> tuple[subprocess.Popen, str]:
    """A Cloudflare quick tunnel: no account, a random https link, alive while this process runs."""
    proc = subprocess.Popen([find_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
                            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    found: list[str] = []

    def read():
        for line in proc.stderr:  # keep reading so the pipe never fills up
            if not found and (url := tunnel_url(line)):
                found.append(url)

    threading.Thread(target=read, daemon=True).start()
    end = time.time() + timeout
    while not found and time.time() < end and proc.poll() is None:
        time.sleep(0.2)
    if not found:
        proc.terminate()
        sys.exit("The tunnel did not start: check the internet connection and try again.")
    return proc, found[0]


def port_in_use(port: int = PORT) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_app(timeout: float = 90) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(1)
    return False


def main() -> None:
    if port_in_use():
        sys.exit(f"Second Look is already running in another window (port {PORT}). Use the link printed there, "
                 "or close that window with Ctrl+C and start again.")
    tunnel, url = start_tunnel()
    key = secrets.token_urlsafe(9)
    env = dict(os.environ, SECOND_LOOK_URL=url, SECOND_LOOK_WALL_KEY=key)
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "app" / "second_look.py"),
                            "--server.port", str(PORT), "--server.headless", "true",
                            "--server.maxUploadSize", "10", "--browser.gatherUsageStats", "false",
                            "--theme.base", "dark"],  # dark widgets on the dark crack background
                           cwd=ROOT, env=env)
    if not wait_for_app():
        for p in (app, tunnel):
            p.terminate()
        sys.exit("The app did not start; nothing is online. See the messages above.")
    make_qr(url, QR)  # only once the app answers, so the saved QR never points to a dead link
    wall = f"http://localhost:{PORT}/?wall"  # opens only on this laptop; the key is a fallback (spec)
    print(f"\nStudents scan the QR or open:  {url}\nLive wall (this laptop only):  {wall}\n"
          f"Live wall for a teammate (send privately, never on screen):  {url}/?wall={key}\n"
          f"QR on its own (show it when students should scan):  http://localhost:{PORT}/?qr\nQR code saved:  {QR}\n"
          "Stop with Ctrl+C.\n", flush=True)
    webbrowser.open(wall)
    try:
        while app.poll() is None and tunnel.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (app, tunnel):
            p.terminate()
        print("Stopped: the link is closed and the wall is empty.")


if __name__ == "__main__":
    main()
