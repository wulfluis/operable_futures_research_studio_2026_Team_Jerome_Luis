"""The 3D view on the wall's result page (D57): the model Grasshopper built, turning slowly until someone drags it.
Spec: spec/features/rhino_3d.md. three.js 0.128.0 from jsDelivr (an exact version; needs the internet)."""

import json

THREE = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
HEIGHT = 560

PAGE = """<div id="sl3d" style="position:relative;width:100%;height:__H__px;background:#f6f8fa;border:1px solid #d1d9e0;border-radius:6px;overflow:hidden">
<div id="sl3d-label" style="position:absolute;left:14px;top:12px;z-index:2;font:600 13px/1.45 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
color:#1f2328;pointer-events:none;max-width:70%"></div>
<div style="position:absolute;right:14px;bottom:10px;z-index:2;font:600 12px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#59636e;
pointer-events:none">Drag to turn · scroll to zoom · double-click: crack / whole element / drawing sheet</div>
</div>
<script src="__THREE__"></script>
<script src="__ORBIT__"></script>
<script>
(function () {
  const M = __DATA__;
  const INK = {"#ffffff": "#1f2328", "#d9d9d9": "#59636e", "#5a5a5a": "#afb8c1", "#4ea1ff": "#0969da"};
  const box = document.getElementById("sl3d");
  document.getElementById("sl3d-label").innerHTML = M.label.map(function (s, i) {
    return i === 0 ? "<b style='color:#1f2328'>" + s + "</b>" : s; }).join("<br>");
  if (!window.THREE) { box.insertAdjacentHTML("beforeend",
    "<p style='color:#d1242f;padding:60px 14px'>The 3D view needs the internet (three.js).</p>"); return; }
  const W = function () { return box.clientWidth; }, H = __H__;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf6f8fa);
  const cam = new THREE.PerspectiveCamera(35, W() / H, 1, 200000);
  const ren = new THREE.WebGLRenderer({antialias: true});
  ren.setPixelRatio(window.devicePixelRatio || 1); ren.setSize(W(), H); box.appendChild(ren.domElement);
  const P = function (p) { return new THREE.Vector3(p[0], p[2], -p[1]); };  // Rhino: Z up; three.js: Y up
  const model = new THREE.Group(), crack = new THREE.Group(), sheet = new THREE.Group();
  function textSprite(s, h, col) {  // a label that always faces the camera, h mm tall (D81)
    const c = document.createElement("canvas"), x = c.getContext("2d"), px = 64;
    x.font = "600 " + px + "px -apple-system, Segoe UI, Helvetica, Arial, sans-serif";
    c.width = Math.ceil(x.measureText(s).width) + 8; c.height = px + 16;
    x.font = "600 " + px + "px -apple-system, Segoe UI, Helvetica, Arial, sans-serif";
    x.fillStyle = "#" + col.getHexString(); x.textBaseline = "top"; x.fillText(s, 4, 6);
    const tex = new THREE.CanvasTexture(c); tex.minFilter = THREE.LinearFilter;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, depthTest: false, transparent: true}));
    sp.scale.set(h * c.width / c.height, h, 1); sp.center.set(0, 0); return sp;
  }
  M.objects.forEach(function (o) {
    const col = new THREE.Color(INK[o.color] || o.color);  // Rhino's colours, readable on white (D76)
    const into = o.group === "sheet" ? sheet : null;
    if (o.kind === "text") {
      const sp = textSprite(o.text, o.height || 8, col); sp.position.copy(P(o.point));
      (into || (o.name === "station_label" ? crack : model)).add(sp); return;
    }
    if (o.kind === "mesh") {
      const pos = [];
      o.faces.forEach(function (f) { f.forEach(function (i) { const v = P(o.vertices[i]); pos.push(v.x, v.y, v.z); }); });
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.computeVertexNormals();
      const mat = o.name === "element"
        ? new THREE.MeshLambertMaterial({color: col, transparent: true, opacity: 0.55, side: THREE.DoubleSide})
        : new THREE.MeshBasicMaterial({color: col, side: THREE.DoubleSide});
      (o.name === "element" ? model : crack).add(new THREE.Mesh(g, mat));
    } else if (o.kind === "segments") {
      const pos = [];
      o.segments.forEach(function (s) { s.forEach(function (p) { const v = P(p); pos.push(v.x, v.y, v.z); }); });
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      (into || (o.name === "stations" || o.name === "tension" ? crack : model))
        .add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: col})));
    } else {
      const g = new THREE.BufferGeometry().setFromPoints(o.points.map(P));
      (into || (o.name === "crack_line" || o.name === "crack_zone" ? crack : model))
        .add(new THREE.Line(g, new THREE.LineBasicMaterial({color: col})));
    }
  });
  scene.add(model); scene.add(crack); scene.add(sheet);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6); sun.position.set(1, 2, 1.5); scene.add(sun);
  const view = new THREE.Vector3(0.9, 0.55, 1.0).normalize();  // in front (+Z here is the element's front), above, right
  const ctl = new THREE.OrbitControls(cam, ren.domElement);
  ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 1.2;
  function frame(obj, k, dir) {  // look at obj from dir (default: front, above, right), k times its size away
    const b = new THREE.Box3().setFromObject(obj), cc = b.getCenter(new THREE.Vector3());
    const s = Math.max(b.getSize(new THREE.Vector3()).length(), 150);
    ctl.target.copy(cc); cam.position.copy(cc).addScaledVector(dir || view, s * k);
  }
  let view_i = 0;  // double-click: the crack, the whole element, the drawing sheet (when there is one)
  const views = [function () { frame(crack, 2.2); }, function () { frame(scene, 1.25); }]
    .concat(sheet.children.length ? [function () { frame(sheet, 1.7, new THREE.Vector3(0, 0, 1)); }] : []);
  frame(crack, 2.2);  // opens on the crack: at real size it is small next to its element
  ctl.addEventListener("start", function () { ctl.autoRotate = false; });  // dragging takes over for good
  ren.domElement.addEventListener("dblclick", function () {  // switch: the crack <-> the whole element
    view_i = (view_i + 1) % views.length; ctl.autoRotate = false; views[view_i]();
  });
  window.addEventListener("resize", function () { cam.aspect = W() / H; cam.updateProjectionMatrix(); ren.setSize(W(), H); });
  (function loop() { requestAnimationFrame(loop); ctl.update(); ren.render(scene, cam); })();
})();
</script>"""


def viewer_html(model: dict) -> str:
    """The page for one model (NNN.mesh.json). Only numbers and the label go in; '</' is escaped so no text in the
    data can end the script early."""
    data = json.dumps({"label": model.get("label") or [], "objects": model.get("objects") or []}).replace("</", "<\\/")
    return (PAGE.replace("__DATA__", data).replace("__THREE__", THREE).replace("__ORBIT__", ORBIT)
            .replace("__H__", str(HEIGHT)))
