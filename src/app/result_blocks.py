"""The phone page's result, in reading order (D115 to D122): exposed steel first, what to do now, the crack to explore
(with corrections back to the app), the photo in steps, before and after, a few short questions, the knock, and how
sure the app is today. Spec: spec/features/class_demo.md ("Result, in order")."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components_v1
import streamlit.components.v2 as components_v2

from app import crack_view, questions
from app.core import auto_triage, causes_seen, steel_seen
from app.sequence import steps
from triage.advice import NOTE as ADVICE_NOTE, advice

ROOT = Path(__file__).resolve().parents[2]
CORRECTIONS_DIR = ROOT / "data" / "corrections"  # never committed (data/ is ignored); numbers only, no photo
CAUSE_WORDS = {"water": "water or damp", "roots": "roots", "pipe": "a pipe"}

# D119: the crack viewer inside a two-way component: corrections drawn in the viewer come back to Python.
_BRIDGE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  let frame = parentElement.querySelector('iframe');
  if (!frame) {
    frame = document.createElement('iframe');
    frame.style.cssText = 'width:100%;border:0;display:block;border-radius:18px;background:#050607';
    frame.setAttribute('title', 'The crack to explore');
    parentElement.appendChild(frame);
    window.addEventListener('message', function (e) {
      if (e.source === frame.contentWindow && e.data && e.data.type === 'sl-corrections' && frame._send)
        frame._send('corrections', {removed: e.data.removed || [], added: e.data.added || []});
    });
  }
  frame._send = setTriggerValue;
  frame.style.height = data.height + 'px';
  if (frame.dataset.key !== data.key) { frame.dataset.key = data.key; frame.srcdoc = data.html; }
}
"""
_viewer = None


def _bridge():
    global _viewer
    if _viewer is None:
        _viewer = components_v2.component("sl_crack_viewer", js=_BRIDGE_JS)
    return _viewer


def viewer(html: str, key: str, height: int) -> dict | None:
    """The viewer; returns the corrections the sender just sent, else None."""
    r = _bridge()(data={"html": html, "key": hashlib.sha1(html[-4000:].encode()).hexdigest()[:16], "height": height},
                  key=key, on_corrections_change=lambda: None)
    return getattr(r, "corrections", None) if r is not None else None


def save_corrections(result, corr: dict, wall=None) -> dict:
    """D119: kept with the result (and its wall entry) and as a file in data/corrections/ for retraining; the level
    is not changed by them, an engineer checks them."""
    entry = {"at": round(time.time(), 1), "removed": [int(i) for i in corr.get("removed") or []],
             "added": [{"kind": a.get("kind"), "pts": a.get("pts")} for a in corr.get("added") or []
                       if a.get("kind") in ("crack", "spalling")]}
    result.corrections.append(entry)
    try:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{int(entry['at'] * 1000)}_{result.wall_number or 'private'}.json"
        (CORRECTIONS_DIR / name).write_text(json.dumps({"wall_number": result.wall_number, **entry}), encoding="utf-8")
    except OSError:
        pass
    if wall is not None and result.wall_number is not None:
        wall.update(result.wall_number, result.to_dict())
    return entry


def explore(result, key: str, height: int = 560, wall=None) -> None:
    """The crack to explore (D83, D118), with corrections (D119)."""
    cache = st.session_state.setdefault("explore", {})
    rev = (key, bool(result.scale_estimate), result.link_down is not None, (result.triage or {}).get("element"),
           bool(result.growth))
    if cache.get(key, (None,))[0] != rev:
        cache[key] = (rev, crack_view.viewer_html(crack_view.photo_jpeg(result.photo), crack_view.view_data(result),
                                                  crack_view.photo_jpeg(crack_view.mask_image(result.overlay), 1400),
                                                  engineer=False))  # the resident's phone: no edits, no heatmap
    corr = viewer(cache[key][1], f"viewer_{key}", height)
    seen = st.session_state.setdefault("corr_seen", {})
    if corr and seen.get(key) != json.dumps(corr, sort_keys=True):
        seen[key] = json.dumps(corr, sort_keys=True)
        e = save_corrections(result, corr, wall)
        st.success(f"Thank you: {len(e['removed'])} crack(s) marked as wrong, {len(e['added'])} drawn. An engineer "
                   "checks them, and they help the models learn. The level does not change by itself.")


def save_entry_corrections(entry, corr: dict, wall) -> dict:
    """D119 on the engineer's wall: the engineer's corrections, kept with the result and in data/corrections/."""
    e = {"at": round(time.time(), 1), "by": "engineer", "removed": [int(i) for i in corr.get("removed") or []],
         "added": [{"kind": a.get("kind"), "pts": a.get("pts")} for a in corr.get("added") or []
                   if a.get("kind") in ("crack", "spalling")]}
    stats = dict(entry.stats)
    stats["corrections"] = list(stats.get("corrections") or []) + [e]
    wall.update(entry.number, stats)
    try:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        (CORRECTIONS_DIR / f"{int(e['at'] * 1000)}_{entry.number}_engineer.json").write_text(
            json.dumps({"wall_number": entry.number, **e}), encoding="utf-8")
    except OSError:
        pass
    return e


# the phone's own location, asked only when the resident taps the button (and the phone asks for permission)
_GEO_JS = """
export default function (component) {
  const { parentElement, setTriggerValue } = component;
  if (parentElement.querySelector('button')) return;
  const b = document.createElement('button');
  b.textContent = 'Use my phone\\'s location';
  b.style.cssText = 'width:100%;min-height:46px;border-radius:999px;border:1px solid #0c9aab;background:#fff;color:#0c7a88;' +
    'font:600 15px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;cursor:pointer';
  const msg = document.createElement('p');
  msg.style.cssText = 'font:12.5px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#5f6672;margin:6px 2px 0';
  b.onclick = function () {
    if (!navigator.geolocation) { msg.textContent = 'This phone does not give its location here.'; return; }
    msg.textContent = 'Asking the phone…';
    navigator.geolocation.getCurrentPosition(function (p) {
      msg.textContent = 'Location added.';
      setTriggerValue('pos', {lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy});
    }, function (e) { msg.textContent = 'No location: ' + (e.code === 1 ? 'permission was not given.' : 'the phone could not find it.'); },
    {enableHighAccuracy: true, timeout: 15000});
  };
  parentElement.appendChild(b); parentElement.appendChild(msg);
}
"""
_geo = None


def where(result, key: str, wall=None) -> None:
    """The place for the engineer's map: from the photo's record, else the phone's location if the resident taps."""
    global _geo
    from app.place import valid
    if result.place:
        src = "the photo" if result.place.get("source") == "photo" else "your phone"
        st.markdown(f'<p class="note">📍 Location shared with the engineers (from {src}).</p>', unsafe_allow_html=True)
        return
    st.markdown('<p class="note">📍 The photo carries no location (phones often remove it). Add your phone\'s '
                'location for the engineers\' map:</p>', unsafe_allow_html=True)
    if _geo is None:
        _geo = components_v2.component("sl_geo", js=_GEO_JS)
    r = _geo(key=f"geo_{key}", on_pos_change=lambda: None)
    pos = getattr(r, "pos", None) if r is not None else None
    if pos and valid(pos.get("lat"), pos.get("lon")):
        result.place = {"lat": round(float(pos["lat"]), 6), "lon": round(float(pos["lon"]), 6), "source": "phone",
                        "accuracy_m": None if pos.get("acc") is None else round(float(pos["acc"]))}
        if wall is not None and result.wall_number is not None:
            wall.update(result.wall_number, result.to_dict())
        st.rerun()


def steel_alert(result) -> None:
    """D115: exposed steel first, before anything else on the page."""
    if steel_seen(result):
        st.markdown('<div class="sl-alert"><b>⚠ Exposed steel seen: critical.</b> A steel bar is visible, so the '
                    'concrete cover is gone. An engineer must look, whatever the crack looks like.</div>',
                    unsafe_allow_html=True)
    cs = [CAUSE_WORDS[c] for c in causes_seen(result) if c in CAUSE_WORDS]
    if cs:
        st.markdown(f'<div class="sl-alert cause"><b>Possible {", ".join(cs)} near the crack.</b> It may come '
                    'before the crack: tell the engineer.</div>', unsafe_allow_html=True)


def todo(result) -> None:
    """D121: what to do now, beside the level."""
    from element.predict import matrix_element
    t = result.triage or {}
    el = t.get("element") or (matrix_element(result.element) if result.element is not None else None)
    steps_ = advice(t.get("level"), steel=steel_seen(result), causes=tuple(causes_seen(result)), element=el,
                    answers=result.answers)
    if not steps_:
        return
    st.markdown('<p class="m-head">What to do now</p><ul class="todo">'
                + "".join(f"<li>{s}</li>" for s in steps_) + f'</ul><p class="note">{ADVICE_NOTE}</p>',
                unsafe_allow_html=True)


def steps_strip(result) -> None:
    """D122: the photo, the crack over it, the crack alone."""
    s = steps(result)
    if len(s) < 2:
        return
    with st.expander("The photo, step by step"):
        cols = st.columns(len(s))
        for c, (cap, jpg) in zip(cols, s):
            c.image(jpg, caption=cap, width="stretch")


def before_after(result, wall) -> None:
    """D122: the earlier photo of the same crack beside this one, with the measured change."""
    if result.previous is None or wall is None:
        return
    before = wall.get(result.previous)
    if before is None or before.zoom is None or result.photo is None:
        return
    with st.expander(f"Before and now: result #{result.previous} and this photo", expanded=True):
        components_v1.html(before_after_html(before.zoom, crack_view.photo_jpeg(result.photo, 1400)), height=420)
        if result.growth:
            st.markdown(f'<p class="note">{result.growth["reason"]}</p>', unsafe_allow_html=True)
        st.markdown('<p class="note">The two photos are shown as taken: line them up by the card. The change in mm '
                    'counts only beyond both photos\' errors.</p>', unsafe_allow_html=True)


BA_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#050607;font:12px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#eef1f5}
#b{position:relative;height:100%;overflow:hidden;border-radius:18px;border:1px solid rgba(255,255,255,.12)}
#b img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#now{position:absolute;inset:0;clip-path:inset(0 0 0 50%)}
#cut{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#22d3ee;box-shadow:0 0 12px #22d3ee}
input{position:absolute;left:5%;right:5%;bottom:12px;width:90%}
.t{position:absolute;top:10px;padding:3px 9px;border-radius:999px;background:rgba(12,14,18,.84);border:1px solid rgba(255,255,255,.12)}
</style></head><body><div id="b"><img src="__BEFORE__" alt="Before"><div id="now"><img src="__NOW__" alt="Now"></div><div id="cut"></div>
<span class="t" style="left:10px">Before</span><span class="t" style="right:10px">Now</span>
<input type="range" min="0" max="100" value="50" aria-label="Slide between before and now"></div>
<script>var r=document.querySelector("input");r.oninput=function(){document.getElementById("now").style.clipPath="inset(0 0 0 "+r.value+"%)";
document.getElementById("cut").style.left=r.value+"%";};</script></body></html>"""


def before_after_html(before: bytes, now: bytes) -> str:
    b64 = lambda b: "data:image/jpeg;base64," + base64.b64encode(b).decode()  # noqa: E731
    return BA_PAGE.replace("__BEFORE__", b64(before)).replace("__NOW__", b64(now))


def ask(result, key: str, wall=None) -> None:
    """D120: a few short questions, one at a time; every answer is kept as training data."""
    a = result.answers
    q = questions.next_question(a)
    done = len([k for k in a if k in questions.BY_KEY])
    total = len(questions.QUESTIONS)
    with st.container(key=f"ask_{key}"):
        st.markdown(f'<p class="m-head">A few questions · {done} of {total}</p>', unsafe_allow_html=True)
        if q is None:
            st.markdown('<p class="note">Thank you. Your answers are with the result.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="lead-s">{q["text"]}</p>', unsafe_allow_html=True)
            opts = q["options"] + [(questions.DONT_KNOW, "I don't know")]
            cols = st.columns(len(opts))
            for c, (val, words) in zip(cols, opts):
                if c.button(words, key=f"q_{key}_{q['key']}_{val}", width="stretch"):
                    a[q["key"]] = val
                    if result.triage and not result.triage.get("steel_first"):
                        auto_triage(result, wall)  # a "yes" to a sign can only raise the level
                    elif wall is not None and result.wall_number is not None:
                        wall.update(result.wall_number, result.to_dict())
                    st.rerun()
        if a:
            with st.expander("Your answers"):
                for text, ans in questions.words(a):
                    st.markdown(f'<p class="note">{text} <b>{ans}</b></p>', unsafe_allow_html=True)
                if st.button("Start the questions again", key=f"q_reset_{key}"):
                    a.clear()
                    st.rerun()


def knock(result, key: str, wall=None) -> None:
    """D120: knock on the wall next to the crack and record it; the numbers are kept, not the sound."""
    with st.expander("Knock on the wall (optional)"):
        st.markdown('<p class="note">Knock firmly with a knuckle or a spoon right next to the crack, 2 or 3 times, '
                    'and record it. Engineers listen for a hollow sound. The app keeps the numbers of the sound, not '
                    'the recording.</p>', unsafe_allow_html=True)
        rec = st.audio_input("Record the knock", key=f"knock_{key}")
        if rec is None:
            return
        f = questions.knock(rec.getvalue())
        if f is None:
            st.markdown('<p class="note">No clear knock heard: knock louder, closer to the phone.</p>',
                        unsafe_allow_html=True)
            return
        if result.tap != f:
            result.tap = f
            if wall is not None and result.wall_number is not None:
                wall.update(result.wall_number, result.to_dict())
        st.markdown(f'<p class="m-line">{f["knocks"]} knock(s) · pitch {f["pitch_hz"]} Hz · rings {f["ring_ms"]:.0f} ms'
                    f'</p><p class="note">{f["note"]}</p>', unsafe_allow_html=True)


def how_sure() -> None:
    """D120: the app's confidence, said openly: measured numbers only."""
    with st.expander("How sure is the app today?"):
        st.markdown("".join(f'<p class="note"><b>{k}:</b> {v}.</p>' for k, v in questions.TODAY)
                    + '<p class="note">It gets more accurate as engineers confirm or correct results.</p>',
                    unsafe_allow_html=True)
