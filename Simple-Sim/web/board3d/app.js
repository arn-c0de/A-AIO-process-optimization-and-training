// 3D-Board-Renderer viewer and defect editor (served by tools/board3d_viewer.py on 127.0.0.1).
//
// The GLB comes from the same Blender scene as the photoreal renders: every part is a node `cmp_<refdes>` at its
// nominal pose. Edits are kept as a state per part ({dx, dy, dRot, dz, tomb, removed, excess}, board mm / deg) and
// turned into the defect ops of simple_sim/board3d/scene.py; "Fotorealistisch rendern" sends them to Cycles.
// glTF is y-up: board (x, y) <-> three (x, -z); a board rotation (CCW from above) is a rotation about +y.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const $ = sel => document.querySelector(sel);
const status = text => { $('#status').textContent = text; };
const TOMB_TILT_DEG = 70;
const EPS = 1e-4;

const board3d = await (await fetch('/api/board3d')).json();
const byRef = new Map(board3d.components.map(c => [c.refdes, c]));
$('#title').textContent = `${board3d.product} ${board3d.side} · ${board3d.source}`;
$('#refdes').append(...board3d.components.filter(c => board3d.stages.includes(c.stage)).map(c => new Option(`${c.package}`, c.refdes)));

// ---------------------------------------------------------------- scene
// Photo look: image-based light (a soft studio room, PMREM), one key light with soft shadows, ambient occlusion
// (GTAO) in the gaps between parts, Khronos PBR-neutral tone mapping (keeps the photo textures' colours), the
// board as a 1.6 mm laminate on a desk. Units are mm (glTF metres read as mm: the scene is consistently scaled).
const view = $('#view');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
view.append(renderer.domElement);
const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.02).texture;
scene.environmentIntensity = 0.45;
scene.background = new THREE.Color(0x23262b);
const cx = board3d.widthMm / 2, cz = -board3d.heightMm / 2, span = Math.max(board3d.widthMm, board3d.heightMm);
const key = new THREE.DirectionalLight(0xfff4e8, 1.9);      // warm ceiling lamp, slightly off axis
key.position.set(cx - 0.35 * span, 1.6 * span, cz + 0.55 * span);
key.target.position.set(cx, 0, cz);
key.castShadow = true;
key.shadow.mapSize.set(4096, 4096);
Object.assign(key.shadow.camera, { left: -0.75 * span, right: 0.75 * span, top: 0.75 * span, bottom: -0.75 * span, near: 1, far: 4 * span });
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.02;
key.shadow.radius = 3;
scene.add(key, key.target);
const fill = new THREE.DirectionalLight(0xdfe8ff, 0.35);    // cool window fill from the other side, no shadow
fill.position.set(cx + span, 0.8 * span, cz - span);
scene.add(fill);
const BOARD_T = board3d.board?.thicknessMm ?? 1.6;
const desk = new THREE.Mesh(new THREE.PlaneGeometry(12 * span, 12 * span),
  new THREE.MeshStandardMaterial({ color: 0x3a3631, roughness: 0.85, metalness: 0 }));
desk.rotation.x = -Math.PI / 2;
desk.position.set(cx, -BOARD_T - 0.02, cz);
desk.receiveShadow = true;
scene.add(desk);
const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 4000);
camera.position.set(cx, 190, cz + 70);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(cx, 0, cz);
orbit.enableDamping = true;
orbit.dampingFactor = 0.12;
orbit.maxPolarAngle = 0.49 * Math.PI;                      // stay above the desk
orbit.update();
const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSize(0.7);
gizmo.addEventListener('dragging-changed', e => { orbit.enabled = !e.value; });
scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);

const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { samples: 4, type: THREE.HalfFloatType }));
composer.addPass(new RenderPass(scene, camera));
const ao = new GTAOPass(scene, camera, 1, 1);
ao.updateGtaoMaterial({ radius: 1.2, distanceExponent: 1.5, thickness: 1.5, scale: 1.0, samples: 16 });
ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16 });
ao.blendIntensity = 0.85;
composer.addPass(ao);
composer.addPass(new OutputPass());
let quality = 'photo';                                       // 'photo' | 'fast'
function setQuality(q) {
  quality = q;
  ao.enabled = q === 'photo';
  renderer.shadowMap.enabled = q === 'photo';
  scene.traverse(o => { if (o.material) [o.material].flat().forEach(m => { m.needsUpdate = true; }); });
  document.querySelectorAll('button.quality').forEach(b => b.classList.toggle('on', b.dataset.quality === q));
}

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

status('lade 3D-Szene …');
const gltf = await new GLTFLoader().loadAsync(board3d.glb, e => e.total && status(`lade 3D-Szene … ${Math.round(100 * e.loaded / e.total)} %`));
scene.add(gltf.scene);

// Materials: Blender's export is tuned for Cycles with the light baked into the photo textures (matte, no
// specular, joints as emission). For a live view under other light they get their physical look back.
const maxAniso = renderer.capabilities.getMaxAnisotropy();
const metal = (m, rough) => Object.assign(m, { metalness: 1, roughness: rough, envMapIntensity: 1.2 });
const swapped = new Map();
function lookOf(mat) {
  if (swapped.has(mat)) return swapped.get(mat);
  let out = mat;
  const n = mat.name || '';
  for (const t of [mat.map, mat.emissiveMap]) if (t) { t.anisotropy = maxAniso; t.needsUpdate = true; }
  if (n.startsWith('mat_substrate')) {
    // solder mask: glossy lacquer over the photo (clearcoat), the photo already holds the copper/silkscreen detail
    out = new THREE.MeshPhysicalMaterial({ name: n, map: mat.map, roughness: 0.6, metalness: 0, clearcoat: 0.3,
      clearcoatRoughness: 0.45, side: THREE.FrontSide });
  } else if (n.startsWith('mat_photo_joint')) {
    // solder joints: the scanned tin as the metal's colour, shiny
    out = metal(new THREE.MeshStandardMaterial({ name: n, map: mat.emissiveMap }), 0.32);
  } else if (n.startsWith('mat_photo')) {
    out = new THREE.MeshStandardMaterial({ name: n, map: mat.map, roughness: 0.62, metalness: 0 });  // moulded epoxy
  } else if (/^mat_(lead|solder|can|endcap|tin|pin|metal)/.test(n)) {
    metal(mat, n.startsWith('mat_can') ? 0.22 : 0.3);
  } else if (n.startsWith('mat_fr4_edge')) {
    Object.assign(mat, { roughness: 0.8, metalness: 0 });
  }
  swapped.set(mat, out);
  return out;
}
gltf.scene.traverse(o => {
  if (!o.isMesh) return;
  o.material = Array.isArray(o.material) ? o.material.map(lookOf) : lookOf(o.material);
  o.castShadow = !o.name.startsWith('board');
  o.receiveShadow = true;
});
if (!gltf.scene.getObjectByName('board_edge')) {             // older GLB without the laminate: add it here
  const edge = new THREE.Mesh(new THREE.BoxGeometry(board3d.widthMm, BOARD_T, board3d.heightMm),
    new THREE.MeshStandardMaterial({ color: 0x1d2b17, roughness: 0.8 }));
  edge.position.set(cx, -BOARD_T / 2 - 0.003, cz);
  edge.receiveShadow = edge.castShadow = true;
  scene.add(edge);
}
const parts = new Map();                     // refdes -> Object3D (cmp_ node)
gltf.scene.traverse(o => {
  if (!o.name.startsWith('cmp_')) return;
  const ref = o.name.slice(4);
  const c = byRef.get(ref) ?? [...byRef.values()].find(x => THREE.PropertyBinding.sanitizeNodeName(x.refdes) === ref);
  if (!c) return;
  o.userData.refdes = c.refdes;
  o.userData.nom = { pos: o.position.clone(), quat: o.quaternion.clone() };
  parts.set(c.refdes, o);
});
status(`${parts.size} Bauteile · klicke ein Bauteil an`);

// ---------------------------------------------------------------- state
const edits = new Map();                     // refdes -> {dx, dy, dRot, dz, tomb, removed, excess}
const blank = () => ({ dx: 0, dy: 0, dRot: 0, dz: 0, tomb: 0, removed: false, excess: false });
const stateOf = ref => edits.get(ref) ?? blank();
let selected = null;
let mode = 'translate';
const selBox = new THREE.BoxHelper(undefined, 0xffd23f);
selBox.visible = false;
scene.add(selBox);
const marks = new Map();                     // refdes -> BoxHelper of edited parts

function apply(ref) {
  const o = parts.get(ref), st = stateOf(ref), c = byRef.get(ref), n = o.userData.nom;
  o.visible = !st.removed;
  o.position.copy(n.pos).add(new THREE.Vector3(st.dx, st.dz, -st.dy));
  o.quaternion.copy(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(st.dRot))).multiply(n.quat);
  if (st.tomb) {                             // pivot about the anchored end, the other end (sign) lifts
    o.updateMatrixWorld(true);
    const half = (c.body?.l ?? 1) / 2;
    const pivot = new THREE.Vector3(-st.tomb * half, 0, 0).applyMatrix4(o.matrixWorld);
    const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(o.quaternion).normalize();
    const angle = st.tomb * THREE.MathUtils.degToRad(TOMB_TILT_DEG);
    o.position.sub(pivot).applyAxisAngle(axis, angle).add(pivot);
    o.rotateOnWorldAxis(axis, angle);
  }
  o.updateMatrixWorld(true);
  const edited = isEdited(st);
  if (edited && !marks.has(ref)) { const b = new THREE.BoxHelper(o, 0xff8c2a); marks.set(ref, b); scene.add(b); }
  if (!edited && marks.has(ref)) { scene.remove(marks.get(ref)); marks.delete(ref); }
  marks.get(ref)?.setFromObject(o);
  if (!edited) edits.delete(ref);
}
const isEdited = st => st.removed || st.excess || st.tomb || [st.dx, st.dy, st.dRot, st.dz].some(v => Math.abs(v) > EPS);

function readBack(ref) {                      // gizmo moved the node: back to board mm / deg
  const o = parts.get(ref), n = o.userData.nom, st = { ...stateOf(ref) };
  const d = o.position.clone().sub(n.pos);
  st.dx = d.x; st.dy = -d.z; st.dz = Math.max(0, d.y);
  const rel = o.quaternion.clone().multiply(n.quat.clone().invert());
  st.dRot = THREE.MathUtils.radToDeg(2 * Math.atan2(rel.y, rel.w));
  st.dRot = ((st.dRot + 540) % 360) - 180;
  edits.set(ref, st);
  apply(ref);
  refresh();
}

function ops() {
  const out = [];
  for (const [ref, st] of edits) {
    const r = v => Math.round(v * 1000) / 1000;
    if (st.removed) { out.push({ op: 'remove', refdes: ref }); continue; }
    if ([st.dx, st.dy, st.dRot].some(v => Math.abs(v) > EPS)) out.push({ op: 'move', refdes: ref, dx: r(st.dx), dy: r(st.dy), dRot: r(st.dRot) });
    if (st.dz > EPS) out.push({ op: 'lift', refdes: ref, dz: r(st.dz) });
    if (st.tomb) out.push({ op: 'tombstone', refdes: ref, tiltDeg: TOMB_TILT_DEG, raisedSign: st.tomb });
    if (st.excess) out.push({ op: 'excess', refdes: ref });
  }
  return out;
}

// ---------------------------------------------------------------- collisions (2D oriented boxes, board mm)
function box(c, st) {
  const p = c.pose, a = THREE.MathUtils.degToRad(p.rotDeg + st.dRot);
  return { x: p.x + st.dx, y: p.y + st.dy, hl: c.body.l / 2, hw: c.body.w / 2, a };
}
function overlap(b1, b2, tol = 0.02) {       // separating axis test on the 4 box axes
  const axes = [b1.a, b1.a + Math.PI / 2, b2.a, b2.a + Math.PI / 2];
  for (const t of axes) {
    const ux = Math.cos(t), uy = Math.sin(t);
    const proj = b => Math.abs(b.hl * Math.cos(b.a - t)) + Math.abs(b.hw * Math.sin(b.a - t));
    const dist = Math.abs((b2.x - b1.x) * ux + (b2.y - b1.y) * uy);
    if (dist > proj(b1) + proj(b2) - tol) return false;
  }
  return true;
}
function collisions() {
  const hits = [];
  for (const [ref, st] of edits) {
    const c = byRef.get(ref);
    if (st.removed || st.tomb || !c.body || !c.pose) continue;
    const b1 = box(c, st);
    for (const o of board3d.components) {
      if (o.refdes === ref || !o.body || !o.pose || o.body.h === 0 || !board3d.stages.includes(o.stage) || o.dnp) continue;
      const so = stateOf(o.refdes);
      if (so.removed) continue;
      if (Math.hypot(o.pose.x - b1.x, o.pose.y - b1.y) > 2 * Math.max(b1.hl, b1.hw, o.body.l, o.body.w)) continue;
      if (overlap(b1, box(o, so))) hits.push(`${ref} ↔ ${o.refdes}`);
    }
  }
  return hits;
}

// ---------------------------------------------------------------- UI
function select(ref) {
  selected = ref && parts.has(ref) ? ref : null;
  gizmo.detach();
  selBox.visible = !!selected;
  if (selected) {
    selBox.setFromObject(parts.get(selected));
    if (!stateOf(selected).tomb && !stateOf(selected).removed) gizmo.attach(parts.get(selected));
    setMode(mode);
  }
  refresh();
}
function setMode(m) {
  mode = m;
  document.querySelectorAll('button.mode').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  gizmo.setMode(m === 'rotate' ? 'rotate' : 'translate');
  gizmo.showX = m === 'translate';
  gizmo.showZ = m === 'translate';
  gizmo.showY = m !== 'translate';
}
function refresh() {
  const c = selected && byRef.get(selected), st = selected ? stateOf(selected) : null;
  const f = v => (Math.round(v * 1000) / 1000).toFixed(3);
  $('#info').innerHTML = '';
  if (c) {
    const rows = [['Bauteil', c.refdes], ['Gehäuse', c.package], ['Familie', `${c.family}${c.familyStatus === 'builder' ? '' : ' (Box)'}`],
      ['Stufe', c.stage], ['Körper', c.body ? `${f(c.body.l)} × ${f(c.body.w)} × ${c.body.h ?? '?'} mm` : '–'],
      ['Lage', `${f(c.pose.x)}, ${f(c.pose.y)} mm · ${f(c.pose.rotDeg)}°`],
      ['Änderung', `Δ ${f(st.dx)}, ${f(st.dy)} mm · ${f(st.dRot)}° · z ${f(st.dz)}`]];
    for (const [k, v] of rows) $('#info').append(Object.assign(document.createElement('dt'), { textContent: k }), Object.assign(document.createElement('dd'), { textContent: v }));
  }
  const list = ops();
  $('#count').textContent = list.length;
  $('#ops').replaceChildren(...list.map(op => {
    const li = document.createElement('li');
    li.textContent = op.op === 'move' ? `${op.refdes}: verschoben ${op.dx}, ${op.dy} mm, ${op.dRot}°`
      : op.op === 'lift' ? `${op.refdes}: angehoben ${op.dz} mm` : op.op === 'tombstone' ? `${op.refdes}: Grabstein ${op.tiltDeg}°`
      : op.op === 'remove' ? `${op.refdes}: fehlt` : `${op.refdes}: zu viel Lot`;
    li.onclick = () => { select(op.refdes); frame(op.refdes); };
    return li;
  }));
  const hits = collisions();
  $('#collide').hidden = !hits.length;
  $('#collide').textContent = hits.length ? `Kollision (physikalisch nicht möglich): ${hits.join(', ')}` : '';
  for (const b of ['#tomb', '#remove', '#excess', '#reset']) $(b).disabled = !selected;
  $('#render').disabled = !list.length;
}
function frame(ref) {
  const o = parts.get(ref);
  if (!o) return;
  const p = new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);
  const off = camera.position.clone().sub(orbit.target).setLength(25);
  orbit.target.copy(p);
  camera.position.copy(p).add(off);
  orbit.update();
}
function toggle(key, fn) {
  if (!selected) return;
  const st = { ...stateOf(selected) };
  fn(st);
  edits.set(selected, st);
  apply(selected);
  select(selected);
}

gizmo.addEventListener('objectChange', () => { if (selected) { readBack(selected); selBox.setFromObject(parts.get(selected)); } });
document.querySelectorAll('button.mode').forEach(b => { b.onclick = () => setMode(b.dataset.mode); });
$('#tomb').onclick = () => toggle('tomb', st => { st.tomb = st.tomb === 0 ? 1 : st.tomb === 1 ? -1 : 0; });
$('#remove').onclick = () => toggle('removed', st => { st.removed = !st.removed; });
$('#excess').onclick = () => toggle('excess', st => { st.excess = !st.excess; });
$('#reset').onclick = () => { if (selected) { edits.delete(selected); apply(selected); select(selected); } };
$('#resetAll').onclick = () => { for (const ref of [...edits.keys()]) { edits.delete(ref); apply(ref); } select(selected); };
$('#search').onchange = e => { const ref = e.target.value.trim(); if (parts.has(ref)) { select(ref); frame(ref); } };

let down = null;
renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || gizmo.dragging) return;
  const r = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  for (const hit of ray.intersectObject(gltf.scene, true)) {
    let o = hit.object;
    while (o && !o.userData.refdes) o = o.parent;
    if (o && o.visible) { select(o.userData.refdes); $('#search').value = o.userData.refdes; return; }
  }
  select(null);
});
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 'escape') select(null);
  else if (k === 'w') setMode('translate');
  else if (k === 'e') setMode('rotate');
  else if (k === 'l') setMode('lift');
  else if (k === 't') $('#tomb').click();
  else if (k === 'delete') $('#remove').click();
  else if (k === 'x') $('#excess').click();
  else if (k === 'r') $('#reset').click();
});

async function post(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function renderWindow() {
  const pts = [...edits.keys()].map(ref => { const c = byRef.get(ref), st = stateOf(ref); return [c.pose.x + st.dx, c.pose.y + st.dy, Math.max(c.body?.l ?? 1, c.body?.w ?? 1)]; });
  const x0 = Math.min(...pts.map(p => p[0] - p[2])) - 3, x1 = Math.max(...pts.map(p => p[0] + p[2])) + 3;
  const y0 = Math.min(...pts.map(p => p[1] - p[2])) - 3, y1 = Math.max(...pts.map(p => p[1] + p[2])) + 3;
  const w = Math.min(50, Math.max(8, x1 - x0)), h = Math.min(50, Math.max(8, y1 - y0));
  return [(x0 + x1 - w) / 2, (y0 + y1 - h) / 2, w, h].map(v => Math.round(v * 100) / 100);
}
$('#save').onclick = async () => {
  try { const r = await post('/api/defects', { name: $('#name').value, ops: ops() }); status(`gespeichert: ${r.saved}`); }
  catch (err) { status(`Fehler: ${err.message}`); }
};
$('#render').onclick = async () => {
  const btn = $('#render');
  btn.disabled = true;
  const t0 = performance.now();
  status('Cycles rendert … (CPU: etwa 10–60 s)');
  try {
    const r = await post('/api/render', { name: $('#name').value, ops: ops(), window: renderWindow(), samples: Number($('#samples').value) });
    const m = r.report.metrics;
    $('#result').hidden = false;
    $('#result img').src = `${r.compare}?t=${Date.now()}`;
    $('#result figcaption').textContent = `echt | Render | Δ E · ${r.report.components} Bauteile · ${r.report.seconds} s · SSIM ${m.ssim}`;
    status(`fertig in ${Math.round((performance.now() - t0) / 1000)} s`);
  } catch (err) { status(`Fehler: ${err.message}`); }
  btn.disabled = false;
};
$('#result img').onclick = () => $('#result').classList.toggle('big');

select(null);
// views: like the flatbed scanner (straight down), a phone held at an angle, low across the board
function viewPreset(kind) {
  const t = new THREE.Vector3(cx, 0, cz), d = 1.25 * span / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / 2;
  const dir = kind === 'top' ? new THREE.Vector3(0, 1, 0.0001) : kind === 'low' ? new THREE.Vector3(-0.35, 0.32, 0.88) : new THREE.Vector3(-0.25, 0.85, 0.55);
  orbit.target.copy(t);
  camera.position.copy(t).add(dir.normalize().multiplyScalar(kind === 'low' ? 0.75 * d : d));
  orbit.update();
}
document.querySelectorAll('button.view').forEach(b => { b.onclick = () => viewPreset(b.dataset.view); });
document.querySelectorAll('button.quality').forEach(b => { b.onclick = () => setQuality(b.dataset.quality); });
$('#shot').onclick = () => {
  composer.render();
  const a = document.createElement('a');
  a.href = renderer.domElement.toDataURL('image/png');
  a.download = `board3d-${board3d.product}-${board3d.side}.png`;
  a.click();
};
viewPreset('angle');
setQuality(quality);
renderer.setAnimationLoop(() => { orbit.update(); if (quality === 'photo') composer.render(); else renderer.render(scene, camera); });
