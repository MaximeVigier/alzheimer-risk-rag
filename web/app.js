// Alzheimer Risk RAG — front vanilla JS + Three.js (pas de build step, import via CDN
// + importmap déclaré dans index.html, nécessaire car OrbitControls.js importe "three"
// en spécificateur nu).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---- Configuration ----------------------------------------------------
const API_BASE_URL = "http://localhost:8000"; // adapte si l'API tourne ailleurs
const DATA_URL = "data/umap3d_fixed.json";
const STRATEGY = "fixed";

const CATEGORY_COLORS = {
  sleep: 0x0d9488,
  physical_activity: 0x2563eb,
  diet_nutrition: 0xea580c,
  cardiovascular_metabolic: 0xdc2626,
  lifestyle_environment: 0x7c3aed,
  genetic: 0xca8a04,
  other: 0x64748b,
};
const HIGHLIGHT_COLOR = 0xe11d48;

const BASE_SIZE = 0.045;
const HIGHLIGHT_SIZE = 0.18;
const DIM_OPACITY = 0.12;
const BASE_OPACITY = 0.8;

// ---- État global --------------------------------------------------------
let points = [];              // métadonnées brutes du JSON
let pointCloud = null;        // THREE.Points
let pointMaterial = null;     // ShaderMaterial (uniforms fog exposés pour ajustement futur)
let geometry = null;
let colorAttr = null;
let sizeAttr = null;
let opacityAttr = null;
let positionsArr = null;      // référence directe au Float32Array des positions (coords centrées)
let highlightedChunkIds = new Set();
let highlightedIndices = [];   // indices des points en surbrillance (évite de re-scanner à chaque frame)
let highlightCentroid = null;  // centre du cluster retrouvé (coords locales, centrées)
let highlightStartTime = 0;    // performance.now() du dernier surlignage, pour la pulsation

// ---- DOM refs -------------------------------------------------------
const canvas = document.getElementById("viz-canvas");
const legendEl = document.getElementById("legend");
const vizInfoEl = document.getElementById("viz-info");
const loadingOverlay = document.getElementById("loading-overlay");
const chatHistoryEl = document.getElementById("chat-history");
const chatFormEl = document.getElementById("chat-form");
const questionInputEl = document.getElementById("question-input");
const askButtonEl = document.getElementById("ask-button");
const apiStatusEl = document.getElementById("api-status");
const apiStatusTextEl = document.getElementById("api-status-text");
const vizLinesEl = document.getElementById("viz-lines");
const vizLabelsEl = document.getElementById("viz-labels");

// Étiquettes de sources actives (une par publication surlignée) : chacune pointe, via un
// trait, vers le point correspondant dans le nuage. Recalculées à chaque nouvelle réponse.
let activeLabels = []; // { index, pmid, title, year, url, lineEl, labelEl }

// =========================================================================
// Three.js setup
// =========================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f6f8);

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(4, 3, 6);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));

function resizeRenderer() {
  const panel = document.getElementById("viz-panel");
  const w = panel.clientWidth;
  const h = panel.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeRenderer);

// Custom vertex shader material for per-point size/color/opacity (via attributes).
// Le fog est géré "à la main" via des uniforms dédiés (fogNearU/fogFarU/fogColorU) plutôt
// que material.fog=true : l'injection automatique de Three.js pour un ShaderMaterial custom
// est fragile (namespaces de varyings en conflit) et produisait un écran noir silencieux.
const VERTEX_SHADER = `
  attribute float pointSize;
  attribute vec3 pointColor;
  attribute float pointOpacity;
  varying vec3 vColor;
  varying float vOpacity;
  varying float vFogDepth;
  void main() {
    vColor = pointColor;
    vOpacity = pointOpacity;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
    vFogDepth = -mvPosition.z;
  }
`;
const FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vOpacity;
  varying float vFogDepth;
  uniform vec3 fogColorU;
  uniform float fogNearU;
  uniform float fogFarU;
  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    float edge = smoothstep(0.5, 0.4, d);
    // léger cerclage plus sombre pour détacher le point du fond clair (sinon les couleurs
    // saturées "flottent" sans contour sur du blanc et paraissent floues).
    float ring = smoothstep(0.42, 0.5, d) * 0.35;
    vec3 finalColor = mix(vColor, vColor * 0.55, ring);
    float fogFactor = clamp((vFogDepth - fogNearU) / (fogFarU - fogNearU), 0.0, 1.0);
    finalColor = mix(finalColor, fogColorU, fogFactor * 0.55);
    gl_FragColor = vec4(finalColor, vOpacity * edge);
  }
`;

function buildPointCloud(pts) {
  geometry = new THREE.BufferGeometry();

  const positions = new Float32Array(pts.length * 3);
  const colors = new Float32Array(pts.length * 3);
  const sizes = new Float32Array(pts.length);
  const opacities = new Float32Array(pts.length);

  const center = { x: 0, y: 0, z: 0 };
  for (const p of pts) {
    center.x += p.x; center.y += p.y; center.z += p.z;
  }
  center.x /= pts.length; center.y /= pts.length; center.z /= pts.length;

  let maxDist = 0;
  const dists = [];
  pts.forEach((p, i) => {
    const dx = p.x - center.x, dy = p.y - center.y, dz = p.z - center.z;
    positions[i * 3] = dx;
    positions[i * 3 + 1] = dy;
    positions[i * 3 + 2] = dz;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    maxDist = Math.max(maxDist, d);
    dists.push(d);

    const hex = CATEGORY_COLORS[p.category] ?? CATEGORY_COLORS.other;
    const c = new THREE.Color(hex);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;

    sizes[i] = BASE_SIZE;
    opacities[i] = BASE_OPACITY;
  });

  // Cadre la caméra sur le 90e centile de distance au centre, pas sur le max : un nuage UMAP
  // a presque toujours quelques outliers isolés, et cadrer sur eux (maxDist) dézoome tellement
  // que le cluster dense (là où sont 90% des points) apparaît minuscule au milieu d'un grand
  // vide. Le 90e centile fait déborder les outliers du cadre, c'est le compromis voulu.
  dists.sort((a, b) => a - b);
  const p90 = dists[Math.floor(dists.length * 0.9)] || maxDist;
  const dist = Math.max(p90 * 1.1, 0.8);
  camera.position.set(dist * 0.6, dist * 0.45, dist * 0.75);
  camera.near = dist / 100;
  camera.far = Math.max(dist * 20, maxDist * 3);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();

  positionsArr = positions;
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  colorAttr = new THREE.BufferAttribute(colors, 3);
  sizeAttr = new THREE.BufferAttribute(sizes, 1);
  opacityAttr = new THREE.BufferAttribute(opacities, 1);
  geometry.setAttribute("pointColor", colorAttr);
  geometry.setAttribute("pointSize", sizeAttr);
  geometry.setAttribute("pointOpacity", opacityAttr);

  const material = new THREE.ShaderMaterial({
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    uniforms: {
      fogColorU: { value: new THREE.Color(0xf5f6f8) },
      fogNearU: { value: dist * 0.9 },
      fogFarU: { value: dist * 2.6 },
    },
  });
  pointMaterial = material;

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Pulsation des points surlignés pendant les ~1.8s qui suivent un nouveau retrieval :
  // sans ça, le changement de couleur/taille est un "cut" instantané qu'on remarque à peine
  // au milieu de 1200 points. Le battement de taille attire l'œil sur le bon cluster.
  if (highlightedIndices.length > 0 && sizeAttr) {
    const elapsed = (performance.now() - highlightStartTime) / 1000;
    const pulseWindow = 1.8;
    const pulse = elapsed < pulseWindow
      ? 1 + 0.35 * Math.sin(elapsed * Math.PI * 2 * 2.2) * (1 - elapsed / pulseWindow)
      : 1;
    for (const i of highlightedIndices) {
      sizeAttr.array[i] = HIGHLIGHT_SIZE * pulse;
    }
    sizeAttr.needsUpdate = true;
  }

  updateLabelPositions();

  renderer.render(scene, camera);
}

// =========================================================================
// Mise en surbrillance du retrieval
// =========================================================================
function applyHighlight(chunkIds, sourcesMeta = []) {
  highlightedChunkIds = new Set(chunkIds);
  const hasHighlight = highlightedChunkIds.size > 0;
  const hc = new THREE.Color(HIGHLIGHT_COLOR);

  highlightedIndices = [];
  const centroid = { x: 0, y: 0, z: 0 };
  const seenPmids = new Set();
  const labelList = [];
  const metaByChunk = new Map(sourcesMeta.map((s) => [s.chunk_id, s]));
  const metaByPmid = new Map(sourcesMeta.map((s) => [s.pmid, s]));

  points.forEach((p, i) => {
    const isHit = highlightedChunkIds.has(p.chunk_id);
    if (!hasHighlight) {
      sizeAttr.array[i] = BASE_SIZE;
      opacityAttr.array[i] = BASE_OPACITY;
      const hex = CATEGORY_COLORS[p.category] ?? CATEGORY_COLORS.other;
      const c = new THREE.Color(hex);
      colorAttr.array[i * 3] = c.r; colorAttr.array[i * 3 + 1] = c.g; colorAttr.array[i * 3 + 2] = c.b;
    } else if (isHit) {
      sizeAttr.array[i] = HIGHLIGHT_SIZE;
      opacityAttr.array[i] = 1.0;
      colorAttr.array[i * 3] = hc.r; colorAttr.array[i * 3 + 1] = hc.g; colorAttr.array[i * 3 + 2] = hc.b;
      highlightedIndices.push(i);
      centroid.x += positionsArr[i * 3];
      centroid.y += positionsArr[i * 3 + 1];
      centroid.z += positionsArr[i * 3 + 2];

      // Une étiquette par publication (pas par chunk) : plusieurs chunks du même papier
      // ne doivent pas empiler des étiquettes identiques.
      if (!seenPmids.has(p.pmid)) {
        seenPmids.add(p.pmid);
        const meta = metaByChunk.get(p.chunk_id) || metaByPmid.get(p.pmid);
        if (meta) {
          labelList.push({ index: i, pmid: meta.pmid, title: meta.title, year: meta.year, url: meta.url });
        }
      }
    } else {
      sizeAttr.array[i] = BASE_SIZE;
      opacityAttr.array[i] = DIM_OPACITY;
      const hex = CATEGORY_COLORS[p.category] ?? CATEGORY_COLORS.other;
      const c = new THREE.Color(hex);
      colorAttr.array[i * 3] = c.r; colorAttr.array[i * 3 + 1] = c.g; colorAttr.array[i * 3 + 2] = c.b;
    }
  });

  sizeAttr.needsUpdate = true;
  opacityAttr.needsUpdate = true;
  colorAttr.needsUpdate = true;

  if (highlightedIndices.length > 0) {
    centroid.x /= highlightedIndices.length;
    centroid.y /= highlightedIndices.length;
    centroid.z /= highlightedIndices.length;
    highlightCentroid = centroid;
    highlightStartTime = performance.now();
    animateCameraTo(centroid);
  } else {
    highlightCentroid = null;
  }

  renderLabels(labelList);
}

// =========================================================================
// Étiquettes de sources (callouts avec ligne de rappel vers le point du nuage)
// =========================================================================
function renderLabels(labelList) {
  vizLinesEl.innerHTML = "";
  vizLabelsEl.innerHTML = "";
  activeLabels = [];

  labelList.forEach((item, slot) => {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    vizLinesEl.appendChild(line);

    const div = document.createElement("div");
    div.className = "viz-label";
    div.style.left = `${LABEL_ANCHOR_X}px`;
    div.style.top = `${LABEL_ANCHOR_Y_START + slot * LABEL_ANCHOR_GAP}px`;
    const titleHtml = item.url
      ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">${escapeHtml(item.title || item.pmid)}</a>`
      : `<span>${escapeHtml(item.title || item.pmid)}</span>`;
    div.innerHTML = `${titleHtml}<div class="meta">PMID ${escapeHtml(item.pmid)}${item.year ? " · " + escapeHtml(item.year) : ""}</div>`;
    vizLabelsEl.appendChild(div);

    activeLabels.push({ index: item.index, slot, lineEl: line, labelEl: div });
  });

  updateLabelPositions();
}

// Projette la position 3D (courante, donc valable pendant la rotation de la caméra) d'un
// point vers des coordonnées écran relatives au panneau.
function projectToScreen(index) {
  const v = new THREE.Vector3(
    positionsArr[index * 3], positionsArr[index * 3 + 1], positionsArr[index * 3 + 2]
  );
  v.project(camera);
  const w = renderer.domElement.clientWidth;
  const h = renderer.domElement.clientHeight;
  return {
    x: (v.x * 0.5 + 0.5) * w,
    y: (-v.y * 0.5 + 0.5) * h,
    behind: v.z > 1,
  };
}

// Les étiquettes restent empilées dans une colonne fixe en haut à gauche du panneau (sobre,
// pas de chevauchement) ; seul le trait de rappel bouge en direct pour suivre le point
// pendant que l'utilisateur fait tourner la scène.
const LABEL_ANCHOR_X = 18;
const LABEL_ANCHOR_Y_START = 18;
const LABEL_ANCHOR_GAP = 72;

function updateLabelPositions() {
  if (activeLabels.length === 0) return;
  const panelRect = renderer.domElement.getBoundingClientRect();
  activeLabels.forEach(({ index, lineEl, labelEl }) => {
    const proj = projectToScreen(index);

    // Le trait part du vrai bord de la boîte (mesuré après layout), pas d'un point fixe
    // arbitraire — sinon il s'arrête au milieu du texte ou dans le vide selon la longueur
    // du titre affiché.
    const box = labelEl.getBoundingClientRect();
    const originX = box.right - panelRect.left;
    const originY = box.top + box.height / 2 - panelRect.top;

    labelEl.style.display = proj.behind ? "none" : "block";
    lineEl.setAttribute("x1", originX);
    lineEl.setAttribute("y1", originY);
    lineEl.setAttribute("x2", proj.x);
    lineEl.setAttribute("y2", proj.y);
    lineEl.style.display = proj.behind ? "none" : "block";
  });
}

// Recentre doucement la cible de l'orbit control sur le cluster retrouvé (au lieu d'un
// jump-cut) : donne l'impression que la caméra "regarde vers" la réponse, sans pour autant
// casser la rotation/zoom que l'utilisateur avait mis en place.
function animateCameraTo(target) {
  const start = controls.target.clone();
  const end = new THREE.Vector3(target.x, target.y, target.z);
  const duration = 900;
  const t0 = performance.now();

  function step() {
    const t = Math.min(1, (performance.now() - t0) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    controls.target.lerpVectors(start, end, eased);
    controls.update();
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// =========================================================================
// Chargement des données UMAP
// =========================================================================
async function loadUmapData() {
  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error(`Impossible de charger ${DATA_URL} (${res.status})`);
  const json = await res.json();
  points = json.points || [];
  return points;
}

function buildLegend() {
  legendEl.innerHTML = '<div class="legend-title">Catégories</div>';
  const present = new Set(points.map((p) => p.category));
  Object.entries(CATEGORY_COLORS).forEach(([cat, hex]) => {
    if (!present.has(cat)) return;
    const row = document.createElement("div");
    row.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = `#${hex.toString(16).padStart(6, "0")}`;
    const label = document.createElement("span");
    label.textContent = cat.replace(/_/g, " ");
    row.appendChild(swatch);
    row.appendChild(label);
    legendEl.appendChild(row);
  });
}

async function initViz() {
  resizeRenderer();
  try {
    await loadUmapData();
    buildPointCloud(points);
    buildLegend();
    vizInfoEl.textContent = `${points.length} chunks affichés (UMAP 384 -> 3 dimensions). ` +
      "Glisser = rotation, molette = zoom, clic droit = pan.";
    loadingOverlay.classList.add("hidden");
  } catch (err) {
    vizInfoEl.textContent = `Erreur de chargement des données : ${err.message}`;
    loadingOverlay.textContent = `Erreur : ${err.message}`;
  }
  animate();
}

// =========================================================================
// Chat / appel API
// =========================================================================
const SUGGESTED_QUESTIONS = [
  "Le sommeil influence-t-il le risque d'Alzheimer ?",
  "Quel est le lien entre activité physique et déclin cognitif ?",
  "L'alimentation méditerranéenne réduit-elle le risque ?",
  "Quels facteurs cardiovasculaires sont associés à la maladie ?",
];

function renderSuggestions() {
  const wrap = document.createElement("div");
  wrap.className = "suggestions";
  wrap.innerHTML = '<div class="label">Exemples de questions</div>';
  SUGGESTED_QUESTIONS.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion-chip";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      wrap.remove();
      handleAsk(q);
    });
    wrap.appendChild(btn);
  });
  chatHistoryEl.appendChild(wrap);
}

function addMessage(html, className) {
  const div = document.createElement("div");
  div.className = `msg ${className}`;
  div.innerHTML = html;
  chatHistoryEl.appendChild(div);
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  return div;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`status ${res.status}`);
    apiStatusEl.className = "ok";
    apiStatusTextEl.textContent = "API connectée";
  } catch (err) {
    apiStatusEl.className = "down";
    apiStatusTextEl.textContent = "API indisponible (lance-la sur " + API_BASE_URL + ")";
  }
}

async function handleAsk(query) {
  addMessage(escapeHtml(query), "user");
  askButtonEl.disabled = true;

  const thinkingDiv = addMessage(
    '<div class="label">RAG</div>Recherche en cours...',
    "msg answer"
  );

  try {
    const res = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, strategy: STRATEGY }),
    });

    if (!res.ok) {
      let detail = `Erreur HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body.detail) detail = body.detail;
      } catch (_) { /* ignore */ }
      thinkingDiv.remove();
      addMessage(`<div class="label">Erreur</div>${escapeHtml(detail)}`, "msg error");
      apiStatusEl.className = "down";
      apiStatusTextEl.textContent = "API en erreur";
      return;
    }

    const data = await res.json();
    thinkingDiv.remove();

    const refusedClass = data.refused ? " refused" : "";
    // Les sources ne sont plus listées ici : elles apparaissent comme étiquettes pointant
    // vers le nuage 3D (cf. renderLabels), ça évite la redondance et garde le chat léger.
    addMessage(
      `<div class="label">Réponse${data.refused ? " (refusée)" : ""}</div>${escapeHtml(data.answer)}`,
      `msg answer${refusedClass}`
    );

    highlightFromSources(data.sources || []);
    apiStatusEl.className = "ok";
    apiStatusTextEl.textContent = "API connectée";
  } catch (err) {
    thinkingDiv.remove();
    addMessage(
      `<div class="label">Erreur</div>Impossible de contacter l'API (${escapeHtml(err.message)}). ` +
      `Vérifie qu'elle tourne sur ${escapeHtml(API_BASE_URL)}.`,
      "msg error"
    );
    apiStatusEl.className = "down";
    apiStatusTextEl.textContent = "API indisponible";
  } finally {
    askButtonEl.disabled = false;
  }
}

// Relie les sources de la réponse API aux points du nuage UMAP. Utilise chunk_id si présent
// (lien exact), sinon retombe sur une correspondance par pmid (un pmid peut couvrir
// plusieurs chunks, donc tous les chunks de ce pmid sont surlignés dans ce cas).
function highlightFromSources(sources) {
  const chunkIds = sources.map((s) => s.chunk_id).filter(Boolean);
  if (chunkIds.length > 0) {
    applyHighlight(chunkIds, sources);
    return;
  }
  const pmids = new Set(sources.map((s) => s.pmid));
  const matched = points.filter((p) => pmids.has(p.pmid)).map((p) => p.chunk_id);
  applyHighlight(matched, sources);
}

chatFormEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = questionInputEl.value.trim();
  if (!query) return;
  const sugg = document.querySelector(".suggestions");
  if (sugg) sugg.remove();
  questionInputEl.value = "";
  handleAsk(query);
});

// =========================================================================
// Démarrage
// =========================================================================
initViz();
checkApiHealth();
renderSuggestions();
setInterval(checkApiHealth, 20000);
