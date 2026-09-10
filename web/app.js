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
  sleep: 0x4fd1c5,
  physical_activity: 0x63b3ed,
  diet_nutrition: 0xf6ad55,
  cardiovascular_metabolic: 0xf87171,
  lifestyle_environment: 0xb794f4,
  genetic: 0xf6e05e,
  other: 0x718096,
};
const HIGHLIGHT_COLOR = 0xff2d95;

const BASE_SIZE = 0.045;
const HIGHLIGHT_SIZE = 0.16;
const DIM_OPACITY = 0.18;
const BASE_OPACITY = 0.85;

// ---- État global --------------------------------------------------------
let points = [];              // métadonnées brutes du JSON
let pointCloud = null;        // THREE.Points
let geometry = null;
let colorAttr = null;
let sizeAttr = null;
let opacityAttr = null;
let highlightedChunkIds = new Set();

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

// =========================================================================
// Three.js setup
// =========================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);

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
const VERTEX_SHADER = `
  attribute float pointSize;
  attribute vec3 pointColor;
  attribute float pointOpacity;
  varying vec3 vColor;
  varying float vOpacity;
  void main() {
    vColor = pointColor;
    vOpacity = pointOpacity;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;
const FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vOpacity;
  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    float edge = smoothstep(0.5, 0.35, d);
    gl_FragColor = vec4(vColor, vOpacity * edge);
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

  pts.forEach((p, i) => {
    positions[i * 3] = p.x - center.x;
    positions[i * 3 + 1] = p.y - center.y;
    positions[i * 3 + 2] = p.z - center.z;

    const hex = CATEGORY_COLORS[p.category] ?? CATEGORY_COLORS.other;
    const c = new THREE.Color(hex);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;

    sizes[i] = BASE_SIZE;
    opacities[i] = BASE_OPACITY;
  });

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
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// =========================================================================
// Mise en surbrillance du retrieval
// =========================================================================
function applyHighlight(chunkIds) {
  highlightedChunkIds = new Set(chunkIds);
  const hasHighlight = highlightedChunkIds.size > 0;
  const hc = new THREE.Color(HIGHLIGHT_COLOR);

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
    let sourcesHtml = "";
    if (data.sources && data.sources.length) {
      sourcesHtml = '<div class="sources">' + data.sources.map((s) => `
        <div class="source-item">
          <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>
          <span class="meta">PMID ${escapeHtml(s.pmid)}${s.year ? " · " + escapeHtml(s.year) : ""}</span>
        </div>`).join("") + "</div>";
    }

    addMessage(
      `<div class="label">Réponse${data.refused ? " (refusée)" : ""}</div>${escapeHtml(data.answer)}${sourcesHtml}`,
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
    applyHighlight(chunkIds);
    return;
  }
  const pmids = new Set(sources.map((s) => s.pmid));
  const matched = points.filter((p) => pmids.has(p.pmid)).map((p) => p.chunk_id);
  applyHighlight(matched);
}

chatFormEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = questionInputEl.value.trim();
  if (!query) return;
  questionInputEl.value = "";
  handleAsk(query);
});

// =========================================================================
// Démarrage
// =========================================================================
initViz();
checkApiHealth();
setInterval(checkApiHealth, 20000);
