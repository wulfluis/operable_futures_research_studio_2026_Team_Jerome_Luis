"""Test inbox: simulated residents (D59). Spec: spec/features/class_demo.md, "Simulated residents".

Runs inside the app only when the folder exists. Each case folder holds crack.jpg, optionally element.jpg, and
case.json ({"card_placed": bool, "kind": str}); it goes through the same pipeline as a phone upload, is shared on the
wall, marked TEST, and then moves to done/.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from app.core import PhotoRejected, analyse, analyse_element

INBOX = Path(__file__).resolve().parents[2] / "data" / "inbox"
POLL_S = 1.0


def process_case(case: Path, wall, model, device, gpu, element_model, scene_model, spalling_model=None,
                 parts_model=None) -> dict:
    """One simulated resident. Returns what happened (for the log); never raises.
    parts_model: SAM 3.1's segmenter (D70), or None."""
    info = {"case": case.name}
    try:
        meta = json.loads((case / "case.json").read_text(encoding="utf-8")) if (case / "case.json").exists() else {}
        r = analyse((case / "crack.jpg").read_bytes(), True, True, wall, model, device, gpu, scene_model,
                    bool(meta.get("card_placed")), test=True)
        r.real_photos = str(meta.get("kind", "")).lower().startswith(("real", "field"))  # D113
        info.update(number=r.wall_number, statement=r.statement, measured=r.measured)
        if (case / "element.jpg").exists() and r.statement in ("Crack found", "Expert needed"):
            try:
                a = analyse_element((case / "element.jpg").read_bytes(), r, wall, element_model, device, gpu,
                                    scene_model, crack_model=model, spalling_model=spalling_model,
                                    parts_model=parts_model)
                info.update(element=a.element, level=(r.triage or {}).get("level"))
            except PhotoRejected as e:
                info["element_refused"] = str(e)
    except (OSError, ValueError, PhotoRejected) as e:
        info["error"] = f"{type(e).__name__}: {e}"
    done = case.parent / "done"
    done.mkdir(exist_ok=True)
    shutil.move(str(case), str(done / f"{case.name}_{int(time.time())}"))
    return info


def ready_cases(inbox: Path) -> list[Path]:
    """Case folders whose case.json is written (the simulator writes it last), oldest name first."""
    return sorted(p for p in inbox.iterdir() if p.is_dir() and p.name != "done" and not p.name.endswith(".part")
                  and (p / "case.json").exists())


def start(wall, model, device, gpu, element_model, scene_model, spalling_model=None,
          inbox: Path = INBOX, parts_loader=None) -> threading.Thread:
    """A background thread that empties the inbox whenever the folder exists (it does nothing while it does not).
    parts_loader: a function giving SAM 3.1 (D70) or None, called when a case arrives (it loads once)."""

    def loop():
        while True:
            for case in (ready_cases(inbox) if inbox.is_dir() else []):
                info = process_case(case, wall, model, device, gpu, element_model, scene_model, spalling_model,
                                    parts_loader() if parts_loader is not None else None)
                print("test inbox:", info, flush=True)
            time.sleep(POLL_S)

    t = threading.Thread(target=loop, name="second-look-test-inbox", daemon=True)
    t.start()
    return t
