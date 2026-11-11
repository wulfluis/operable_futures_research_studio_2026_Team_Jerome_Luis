"""A local page to try SAM 3.1 on your own photos: upload a photo, type what to look for (reading list [81][82]).

Offline test only, not part of the app; nothing here sets a level (rule 5). Runs on this laptop only
(127.0.0.1), so photos never leave it, and nothing is saved unless you press the save button
(then into `data/results/sam3/`, git-ignored).

    streamlit run src/experiments/sam3_ui.py --server.address 127.0.0.1 --server.port 8510
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam3_test as s  # noqa: E402

st.set_page_config(page_title="SAM 3.1 test", layout="wide")


@st.cache_resource(show_spinner="Loading SAM 3.1 (3.5 GB, about 10 s) ...")
def predictor(device: str, conf: float):
    return s.make_predictor(device, conf)


st.title("SAM 3.1: find things by name")
st.caption("Local test page. The photo stays on this laptop. SAM 3.1 finds things you can name; it does not "
           "judge loads, materials under paint or risk.")

with st.sidebar:
    preset = st.radio("Words", ["Elements (our list of 8)", "My own words"])
    if preset.startswith("Elements"):
        words = [w for ws in s.ELEMENTS.values() for w in ws]
        st.markdown("\n".join(f"- **{e}**: {' / '.join(ws)}" for e, ws in s.ELEMENTS.items()))
        elements = True
    else:
        text = st.text_input("What to look for (comma between words, English works best)",
                             "wall, door frame, crack, credit card")
        words = [w.strip() for w in text.split(",") if w.strip()]
        elements = False
    conf = st.slider("Lowest confidence shown", 0.05, 0.9, 0.25, 0.05,
                     help="Lower shows more finds, including wrong ones.")
    device = st.selectbox("Processor", ["mps", "cpu"], help="mps is the laptop's graphics chip, faster.")

photo_file = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
if not s.WEIGHTS.exists():
    st.error(f"Weights missing: {s.WEIGHTS}")
    st.stop()
if photo_file is None or not words:
    st.info("Upload a photo and give at least one word.")
    st.stop()

photo = ImageOps.exif_transpose(Image.open(photo_file)).convert("RGB")
p = predictor(device, 0.05)  # finds down to 0.05; the slider filters, so moving it needs no new run

key = (photo_file.name, photo_file.size, tuple(words), device)
if st.session_state.get("key") != key:
    with st.spinner("Looking ..."):
        t = time.perf_counter()
        p.set_image(np.asarray(photo)[:, :, ::-1].copy())
        r = p(text=words)[0]
        found = []
        if r.masks is not None and len(r.masks):
            for m, c, cf in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy()):
                if m.shape != (photo.height, photo.width):
                    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(photo.size)) > 127
                found.append((words[c], m, float(cf)))
        st.session_state.update(key=key, found=found, seconds=time.perf_counter() - t)

found = [f for f in st.session_state["found"] if f[2] >= conf]
labels = words
if elements:
    found, labels = s.as_elements(found), list(s.ELEMENTS)
picture = s.draw(photo, found, labels)

left, right = st.columns([3, 2])
left.image(picture, width="stretch")
with right:
    st.metric("Time", f"{st.session_state['seconds']:.1f} s")
    rows = []
    for name in labels:
        hits = sorted((c for n, _, c in found if n == name), reverse=True)
        share = float(np.mean(np.any([m for n, m, _ in found if n == name] or [np.zeros((1, 1), bool)], axis=0)))
        rows.append({"looked for": name, "found": len(hits),
                     "confidences": ", ".join(f"{c:.2f}" for c in hits[:8]) or "–",
                     "share of photo": f"{share:.0%}"})
    st.table(rows)
    if elements:
        st.caption(f"Wall: {s.WALL_NOTE}.")
    st.caption("Confidence is SAM's own score, not a probability that it is right. Check every find by eye.")
    buf = io.BytesIO()
    picture.save(buf, format="PNG")
    st.download_button("Download the picture", buf.getvalue(), f"{Path(photo_file.name).stem}_sam3.png")
    if st.button("Save in data/results/sam3/"):
        s.OUT.mkdir(parents=True, exist_ok=True)
        picture.save(s.OUT / f"{Path(photo_file.name).stem}_ui.png")
        st.success("Saved (git-ignored).")
