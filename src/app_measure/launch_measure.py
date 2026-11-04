"""Start the copy with bank-card measurement for phones on the same Wi-Fi: https link, QR, live wall.
Spec: spec/features/app_measure.md. The original launcher (src/app/launch.py) and the team's fixed ngrok link are not
used, so both apps can run side by side (original on port 8501, this copy on 8503).

Usage, from the repository root:
    python3 src/app_measure/launch_measure.py
The phone must be on the same Wi-Fi. The certificate is self-made, so the phone warns once ("not private"):
Safari "Show Details" > "visit this website", Chrome "Advanced" > "Proceed". Stop with Ctrl+C.
"""

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import ssl
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
from app_measure.core_measure import make_qr

PORT = 8503
LOCAL = ROOT / "data" / "results" / "app_measure"  # data/ is ignored by git: certificate and QR stay local
CERT, KEY, CERT_IP = LOCAL / "cert.pem", LOCAL / "key.pem", LOCAL / "cert_ip.txt"
QR = LOCAL / "qr.png"
QR_PUBLIC = LOCAL / "qr_public.png"


def lan_ip() -> str:
    """This laptop's address on the Wi-Fi (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]


def ensure_cert(ip: str) -> None:
    """A self-signed certificate for this address; made again when the address changes."""
    if CERT.exists() and KEY.exists() and CERT_IP.exists() and CERT_IP.read_text().strip() == ip:
        return
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-keyout", str(KEY),
                    "-out", str(CERT), "-subj", "/CN=Second Look measure",
                    "-addext", f"subjectAltName=IP:{ip},DNS:localhost,IP:127.0.0.1"],
                   check=True, capture_output=True)
    CERT_IP.write_text(ip)


def port_in_use(port: int = PORT) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_app(timeout: float = 90) -> bool:
    ctx = ssl._create_unverified_context()  # our own self-signed certificate, on this laptop
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"https://127.0.0.1:{PORT}/_stcore/health", timeout=2, context=ctx) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(1)
    return False


TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def start_tunnel(timeout: float = 60) -> tuple[subprocess.Popen, str]:
    """A Cloudflare quick tunnel to this copy: no account, a random public https link with a proper certificate,
    alive while this process runs. Not the team's fixed ngrok link (D40), so the original's QR is not touched."""
    exe = shutil.which("cloudflared")
    if not exe:
        sys.exit("cloudflared not found: brew install cloudflared")
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"https://localhost:{PORT}", "--no-tls-verify"],
                            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    found: list[str] = []

    def read():
        for line in proc.stderr:  # keep reading so the pipe never fills up
            if not found and (m := TUNNEL_URL.search(line)):
                found.append(m.group(0))

    threading.Thread(target=read, daemon=True).start()
    end = time.time() + timeout
    while not found and time.time() < end and proc.poll() is None:
        time.sleep(0.2)
    if not found:
        proc.terminate()
        sys.exit("The public link did not start: check the internet connection and try again.")
    return proc, found[0]


TEAM_NGROK_URL = os.environ.get("SECOND_LOOK_NGROK_URL", "https://cranium-bladder-tapioca.ngrok-free.dev")  # D40
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def start_ngrok(timeout: float = 30) -> tuple[subprocess.Popen, str]:
    """The team's fixed ngrok link (the QR in the deck, D40) pointed at this copy. Needs the team's authtoken on
    this laptop (`ngrok config add-authtoken <token>`, never in git), and the link must not be served elsewhere:
    ngrok allows one laptop at a time. Jerome's launcher and files are not used or changed."""
    exe = shutil.which("ngrok")
    if not exe:
        sys.exit("ngrok not found: brew install ngrok")
    proc = subprocess.Popen([exe, "http", f"https://localhost:{PORT}", f"--url={TEAM_NGROK_URL}", "--log", "stdout"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log: list[str] = []
    threading.Thread(target=lambda: [log.append(line) for line in proc.stdout], daemon=True).start()
    end = time.time() + timeout
    while time.time() < end and proc.poll() is None:
        try:
            with urllib.request.urlopen(NGROK_API, timeout=2) as r:
                if any(t.get("public_url") == TEAM_NGROK_URL for t in json.load(r).get("tunnels", [])):
                    return proc, TEAM_NGROK_URL
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    proc.terminate()
    text = " ".join(log)
    if "ERR_NGROK_4018" in text or "authtoken" in text.lower():
        sys.exit("ngrok has no authtoken: ask Jerome for the team's token, then  ngrok config add-authtoken <token>")
    if "ERR_NGROK_334" in text or "already online" in text:
        sys.exit("The team link is still served by another laptop (Jerome's app?): stop it there first.")
    sys.exit("The team ngrok link did not start:\n" + text[-1500:])


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the measurement copy of the class demo.")
    ap.add_argument("--public", action="store_true",
                    help="also open a public link (Cloudflare quick tunnel) for people on other networks")
    ap.add_argument("--ngrok", action="store_true",
                    help="serve this copy under the team's fixed ngrok link, the QR in the deck (D40)")
    args = ap.parse_args()
    if port_in_use():
        sys.exit(f"The measurement copy is already running (port {PORT}). Use the link printed there, or stop it "
                 "with Ctrl+C and start again.")
    if not (ROOT / "models" / "unet_crack.pt").exists():
        sys.exit("models/unet_crack.pt is missing: download it from the model-v3 release (README, Run the app).")
    ip = lan_ip()
    ensure_cert(ip)
    url = f"https://{ip}:{PORT}"
    tunnel = None
    if args.ngrok:
        tunnel, url = start_ngrok()  # the deck's QR now opens this copy
    elif args.public:
        tunnel, url = start_tunnel()  # the QR and the QR page then show the public link
    key = secrets.token_urlsafe(9)
    env = dict(os.environ, SECOND_LOOK_URL=url, SECOND_LOOK_WALL_KEY=key)
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                            str(ROOT / "src" / "app_measure" / "second_look_measure.py"),
                            "--server.port", str(PORT), "--server.address", "0.0.0.0", "--server.headless", "true",
                            "--server.sslCertFile", str(CERT), "--server.sslKeyFile", str(KEY),
                            "--server.maxUploadSize", "10", "--browser.gatherUsageStats", "false",
                            "--theme.base", "dark"], cwd=ROOT, env=env)
    procs = [p for p in (app, tunnel) if p]
    if not wait_for_app():
        for p in procs:
            p.terminate()
        sys.exit("The app did not start. See the messages above.")
    qr = QR_PUBLIC if tunnel else QR
    make_qr(url, qr)
    if args.ngrok:
        print("\nThe deck's QR (team ngrok link) now opens this copy. The free ngrok plan shows a one-time "
              "'You are about to visit' page: tap 'Visit Site'.", flush=True)
    if tunnel:
        print(f"\nPublic link, any network (send it, or the QR):  {url}\n"
              f"Same Wi-Fi as this laptop also works:  https://{ip}:{PORT}\n"
              f"Live wall for a teammate (send privately: anyone with it sees the shared photos):  {url}/?wall={key}\n",
              flush=True)
    else:
        print(f"\nPhones on this Wi-Fi scan the QR or open:  {url}\n"
              "The phone warns once about the certificate (made on this laptop): continue to the site.\n", flush=True)
    print(f"Live wall (this laptop only):  https://localhost:{PORT}/?wall\n"
          f"QR on its own:  https://localhost:{PORT}/?qr\nQR code saved:  {qr}\nStop with Ctrl+C.\n", flush=True)
    webbrowser.open(f"https://localhost:{PORT}/?qr")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        print("Stopped: the links are closed and the wall is empty.")


if __name__ == "__main__":
    main()
