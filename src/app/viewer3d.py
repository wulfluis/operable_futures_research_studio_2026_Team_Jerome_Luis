"""The 3D view on the wall's result page (D57): the model Grasshopper built, turning slowly until someone drags it.
Spec: spec/features/rhino_3d.md. three.js 0.128.0 from jsDelivr (an exact version; needs the internet)."""

import json

THREE = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
HEIGHT = 560

PAGE = """<div id="sl3d" style="position:relative;width:100%;height:__H__px;background:#000;border:1px solid #262626">
<div id="sl3d-label" style="position:absolute;left:14px;top:12px;z-index:2;font:600 13px/1.45 'Source Sans Pro',sans-serif;
color:#d9d9d9;pointer-events:none;max-width:70%"></div>
<div style="position:absolute;right:14px;bottom:10px;z-index:2;font:600 12px 'Source Sans Pro',sans-serif;color:#8c8c8c;
pointer-events:none">Drag to turn · scroll to zoom · double-click: crack / whole element</div>
</div>
<script src="__THREE__"></script>
<script src="__ORBIT__"></script>
<script>
(function () {
  const M = __DATA__;
  const box = document.getElementById("sl3d");
  document.getElementById("sl3d-label").innerHTML = M.label.map(function (s, i) {
    return i === 0 ? "<b style='color:#fff;text-transform:uppercase'>" + s + "</b>" : s; }).join("<br>");
  if (!window.THREE) { box.insertAdjacentHTML("beforeend",
    "<p style='color:#d62728;padding:60px 14px'>The 3D view needs the internet (three.js).</p>"); return; }
  const W = function () { return box.clientWidth; }, H = __H__;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
  const cam = new THREE.PerspectiveCamera(35, W() / H, 1, 200000);
  const ren = new THREE.WebGLRenderer({antialias: true});
  ren.setPixelRatio(window.devicePixelRatio || 1); ren.setSize(W(), H); box.appendChild(ren.domElement);
  const P = function (p) { return new THREE.Vector3(p[0], p[2], -p[1]); };  // Rhino: Z up; three.js: Y up
  const model = new THREE.Group(), crack = new THREE.Group();
  M.objects.forEach(function (o) {
    const col = new THREE.Color(o.color);
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
      model.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: col})));
    } else {
      const g = new THREE.BufferGeometry().setFromPoints(o.points.map(P));
      (o.name === "crack_line" || o.name === "crack_zone" ? crack : model)
        .add(new THREE.Line(g, new THREE.LineBasicMaterial({color: col})));
    }
  });
  scene.add(model); scene.add(crack);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6); sun.position.set(1, 2, 1.5); scene.add(sun);
  const view = new THREE.Vector3(0.9, 0.55, 1.0).normalize();  // in front (+Z here is the element's front), above, right
  const ctl = new THREE.OrbitControls(cam, ren.domElement);
  ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 1.2;
  function frame(obj, k) {  // look at obj from the front, above and right, k times its size away
    const b = new THREE.Box3().setFromObject(obj), cc = b.getCenter(new THREE.Vector3());
    const s = Math.max(b.getSize(new THREE.Vector3()).length(), 150);
    ctl.target.copy(cc); cam.position.copy(cc).addScaledVector(view, s * k);
  }
  let onCrack = true;
  frame(crack, 2.2);  // opens on the crack: at real size it is small next to its element
  ctl.addEventListener("start", function () { ctl.autoRotate = false; });  // dragging takes over for good
  ren.domElement.addEventListener("dblclick", function () {  // switch: the crack <-> the whole element
    onCrack = !onCrack; ctl.autoRotate = false;
    if (onCrack) { frame(crack, 2.2); } else { frame(scene, 1.25); }
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
