"""Start the class demo: the web page, its public https link, the QR code and the live wall. Spec: spec/features/class_demo.md.

The link is the team's fixed ngrok address (the same QR every time, D40); without ngrok it falls back to a random
Cloudflare quick-tunnel link.

Usage, from the repository root (laptop plugged in and online):
    py src/app/launch.py
Stop with Ctrl+C: the link dies and every photo in memory is gone.
"""

import json
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
# The team's free ngrok dev domain (D40): the same link, and so the same QR, at every start. Override with SECOND_LOOK_NGROK_URL.
FIXED_URL = os.environ.get("SECOND_LOOK_NGROK_URL", "https://cranium-bladder-tapioca.ngrok-free.dev")
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def find_ngrok() -> str | None:
    found = shutil.which("ngrok")
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for candidate in [found, *winget.glob("Ngrok.Ngrok_*/ngrok.exe")]:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def ngrok_online() -> bool:
    """True when an ngrok agent on this laptop already serves the fixed link."""
    try:
        with urllib.request.urlopen(NGROK_API, timeout=2) as r:
            return any(t.get("public_url") == FIXED_URL for t in json.load(r).get("tunnels", []))
    except (OSError, ValueError):
        return False


def start_ngrok(timeout: float = 30) -> subprocess.Popen | None:
    """The fixed link through ngrok (account set up with `ngrok config add-authtoken`); None if it cannot start."""
    ngrok = find_ngrok()
    if not ngrok:
        return None
    proc = subprocess.Popen([ngrok, "http", str(PORT), f"--url={FIXED_URL}", "--log", "stdout"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + timeout
    while time.time() < end and proc.poll() is None:
        if ngrok_online():
            return proc
        time.sleep(0.5)
    proc.terminate()
    return None


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
    tunnel, url = None, FIXED_URL
    if not ngrok_online():  # an ngrok already serving the fixed link is reused, not started twice
        tunnel = start_ngrok()
        if tunnel is None:
            print("The fixed ngrok link did not start (is ngrok installed and its authtoken set?). "
                  "Falling back to a random Cloudflare link: the QR in the slides will NOT work this time.", flush=True)
            tunnel, url = start_tunnel()
    key = secrets.token_urlsafe(9)
    env = dict(os.environ, SECOND_LOOK_URL=url, SECOND_LOOK_WALL_KEY=key)
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "app" / "second_look.py"),
                            "--server.port", str(PORT), "--server.headless", "true",
                            "--server.maxUploadSize", "10", "--browser.gatherUsageStats", "false",
                            "--theme.base", "dark"],  # dark widgets on the dark crack background
                           cwd=ROOT, env=env)
    procs = [p for p in (app, tunnel) if p]
    if not wait_for_app():
        for p in procs:
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
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        print("Stopped: the link is closed and the wall is empty.")


if __name__ == "__main__":
    main()
