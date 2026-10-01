"use strict";

const $ = (id) => document.getElementById(id);
let selectedClass = null;
let lastState = null;
let molById = {};

// Each tab has its own server-side session, keyed by a random id kept in
// sessionStorage (so it survives a reload but is not shared with other tabs).
// crypto.getRandomValues, unlike randomUUID, also works over plain http.
const SID_KEY = "icarus-session-id";
const newSid = () =>
  Array.from(crypto.getRandomValues(new Uint8Array(16)),
             (b) => b.toString(16).padStart(2, "0")).join("");
let SID = sessionStorage.getItem(SID_KEY) || newSid();
sessionStorage.setItem(SID_KEY, SID);

// A duplicated tab inherits sessionStorage, and with it the id. Ask the other open
// tabs whether one already uses this id; if so, start a fresh session instead.
const sidReady = new Promise((resolve) => {
  if (!("BroadcastChannel" in window)) return resolve();
  const ch = new BroadcastChannel("icarus-sessions");
  let probing = true;
  ch.onmessage = (e) => {
    const { type, sid } = e.data || {};
    if (sid !== SID) return;
    if (type === "probe" && !probing) ch.postMessage({ type: "taken", sid });
    else if (type === "taken" && probing) {
      SID = newSid();
      sessionStorage.setItem(SID_KEY, SID);
    }
  };
  ch.postMessage({ type: "probe", sid: SID });
  setTimeout(() => { probing = false; resolve(); }, 150);
});

async function api(path, body) {
  await sidReady;
  const opts = { method: body === undefined ? "GET" : "POST", headers: { "X-Session-Id": SID } };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    toast(data.error || "Request failed");
    throw new Error(data.error || res.statusText);
  }
  return data;
}

let toastTimer = null;
function toast(msg) {
  if (!msg) return;
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 4000);
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ChEBI definitions carry inline HTML (<i>, <sub>, <sup>, …). Escape everything,
// then re-enable a safe formatting-only allowlist — so italics/sub/superscripts
// render while any scripts or other markup stay inert.
function sanitizeChebiHtml(s) {
  if (!s) return "";
  let e = escapeHtml(s);
  ["i", "b", "sub", "sup", "em", "strong"].forEach((t) => {
    e = e.replace(new RegExp(`&lt;(/?${t})&gt;`, "gi"), "<$1>");
  });
  return e.replace(/&lt;br\s*\/?&gt;/gi, "<br>");
}

function chebiLink(id) {
  const u = `https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:${encodeURIComponent(id)}`;
  return `<a class="chebi-link" href="${u}" target="_blank" rel="noopener">CHEBI:${escapeHtml(String(id))} ↗</a>`;
}

// Small stable hash so the depiction URL is content-addressed: session molecule
// ids (m1, m2, …) are reused across resets/restarts, and without this the browser
// would serve a stale cached structure for a reused id (name/structure mismatch).
function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}
const depictUrl = (m, w, h) =>
  `/api/depict?id=${encodeURIComponent(m.id)}&sid=${SID}&w=${w}&h=${h}&v=${hashStr(m.smiles || m.id)}`;

// ── rendering ────────────────────────────────────────────────────────────
function render(state) {
  lastState = state;
  molById = {};
  state.molecules.forEach((m) => (molById[m.id] = m));
  const pos = state.molecules.filter((m) => m.label === "pos");
  const neg = state.molecules.filter((m) => m.label === "neg");

  renderLlmModel(state.llm_model);
  $("pos-count").textContent = pos.length;
  $("neg-count").textContent = neg.length;
  $("counts").textContent =
    `${state.counts.pos} positive · ${state.counts.neg} negative`;

  renderStructGrid($("pos-list"), pos, state.has_rule);
  renderStructGrid($("neg-list"), neg, state.has_rule);

  renderConcept(state.concept);

  if (state.has_rule) {
    $("rule-area").classList.remove("hidden");
    const parts = state.rule_parts ||
      { main: { rule: state.current_rule, nl: null }, aux: [] };
    // Reload the block editor only when the selected rule actually changed, so an
    // in-progress edit isn't clobbered on every state refresh (adding examples etc.).
    const sig = `${state.selected_rule_id}|${state.has_rule}`;
    if (sig !== lastRuleSig) {
      lastRuleSig = sig;
      Blocks.load(parts.main.rule || "");
      editorDirty = false;
      syncAdvancedFromBlocks(true);
    }
    $("rule-nl").textContent = parts.main.nl || "—";
    renderAux(parts.aux || []);
    renderReport(state.report);
    updateEditedFlag();
  } else {
    lastRuleSig = null;
    $("rule-area").classList.add("hidden");
  }
  renderHistory(state);
}

// The block editor is canonical for the main hypothesis; these track when to
// (re)load it and whether the user has unsaved edits (blocks, aux, or Prolog).
let lastRuleSig = null;
let editorDirty = false;

function syncAdvancedFromBlocks(force) {
  const ta = $("rule-text");
  if (force || document.activeElement !== ta) ta.value = Blocks.compile();
}

function updateEditedFlag() {
  $("edited-flag").textContent =
    editorDirty ? "(unsaved edits — Apply to save as a new rule)" : "";
}

// ── rule history ────────────────────────────────────────────────────────────
const ORIGIN_LABELS = { popper: "Popper", aleph: "Aleph", llm: "LLM", edited: "Edited" };

function relTime(ts) {
  if (!ts) return "";
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function confSummary(c) {
  return `<span class="rh-conf-cells">
    <span class="rh-c good">TP ${c.TP}</span>
    <span class="rh-c good">TN ${c.TN}</span>
    <span class="rh-c bad">FP ${c.FP}</span>
    <span class="rh-c bad">FN ${c.FN}</span></span>`;
}

// Collapsed by default; label/re-evaluate visibility depend on the current
// collapse state plus the latest history summary.
let historyMeta = { count: 0, anyStale: false };

function updateHistoryChrome() {
  const collapsed = $("rule-history-list").classList.contains("hidden");
  const { count, anyStale } = historyMeta;
  $("rh-toggle").textContent =
    `${collapsed ? "▸ Show" : "▾ Hide"} rule history (${count})${anyStale ? " ⚠" : ""}`;
  $("reeval-btn").classList.toggle("hidden", collapsed || !anyStale);
}

$("rh-toggle").onclick = () => {
  $("rule-history-list").classList.toggle("hidden");
  updateHistoryChrome();
};

function renderHistory(state) {
  const area = $("rule-history-area"), list = $("rule-history-list");
  const hist = state.rule_history || [];
  if (!hist.length) { area.classList.add("hidden"); list.innerHTML = ""; return; }
  area.classList.remove("hidden");
  historyMeta = { count: hist.length, anyStale: hist.some((h) => h.stale) };
  list.innerHTML = hist.map((h) => {
    const selected = h.id === state.selected_rule_id;
    const label = ORIGIN_LABELS[h.origin] || h.origin;
    const conf = h.confusion
      ? confSummary(h.confusion) +
        (h.stale ? `<span class="rh-stale">⚠ examples changed since — re-evaluate</span>` : "")
      : `<span class="muted">not evaluated</span>`;
    return `<div class="rh-item ${selected ? "selected" : ""}" data-id="${h.id}">
      <div class="rh-item-head">
        <span class="rh-origin origin-${h.origin}">${label}</span>
        <span class="muted rh-time">${relTime(h.created_at)}</span>
        ${selected ? `<span class="rh-selected">● selected</span>`
                   : `<button class="rh-select linkbtn">Select</button>`}
      </div>
      <div class="rh-rule mono">${escapeHtml(h.rule)}</div>
      <div class="rh-conf">${conf}</div>
    </div>`;
  }).join("");
  updateHistoryChrome();
}

// The concept definition (seeded from a ChEBI class, editable) — shown once a
// concept exists; the edited text is what the LLM backend receives.
function renderConcept(concept) {
  const box = $("concept-box");
  if (!concept || (!concept.name && !concept.definition)) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  $("concept-name").textContent = concept.name
    ? (concept.chebi_id ? `${concept.name} (CHEBI:${concept.chebi_id})` : concept.name)
    : "";
  const ta = $("concept-def");
  if (document.activeElement !== ta) ta.value = concept.definition || "";
}

// Persist an edited definition when the user leaves the box.
$("concept-def").addEventListener("change", async () => {
  try {
    await api("/api/concept/definition", { definition: $("concept-def").value });
    loadPredicates();  // retrieval of generated predicates runs against the definition
  } catch (e) { /* toast already shown */ }
});

function renderStructGrid(container, mols, hasRule) {
  container.innerHTML = "";
  if (!mols.length) {
    container.innerHTML = `<div class="muted" style="grid-column:1/-1">No molecules yet.</div>`;
    return;
  }
  mols.forEach((m) => {
    const other = m.label === "pos" ? "neg" : "pos";
    const swap = m.label === "pos" ? "→ −" : "→ +";
    const badge = hasRule && m.outcome
      ? `<span class="card-badge badge ${m.outcome}">${m.outcome}</span>` : "";
    const explain = hasRule && m.prediction === true
      ? `<button class="explain-btn" data-id="${m.id}" title="Why is this positive?">?</button>` : "";
    const card = document.createElement("div");
    card.className = "struct-card";
    card.draggable = true;
    card.dataset.id = m.id;
    card.dataset.label = m.label;
    card.innerHTML = `
      ${badge}
      <div class="card-ctrls">
        <button class="mini-btn swap ${other}" data-id="${m.id}" data-to="${other}" title="Move to ${other === "pos" ? "positive" : "negative"}">${swap}</button>
        <button class="mini-btn x" data-id="${m.id}" title="remove">×</button>
      </div>
      <div class="thumb"><img loading="lazy" draggable="false" src="${depictUrl(m, 120, 100)}" alt=""></div>
      ${explain}
      <div class="cap" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</div>`;
    card.addEventListener("click", (e) => {
      if (e.target.closest(".card-ctrls")) return;
      if (e.target.closest(".explain-btn")) { openExplain(m.id, m.name); return; }
      openModal(m.id);
    });
    container.appendChild(card);
  });
}

function renderReport(report) {
  if (!report) { $("confusion").innerHTML = ""; $("mismatch-area").innerHTML = ""; return; }
  const c = report.confusion;
  $("confusion").innerHTML = `
    ${confCell("TP", c.TP, "good")}
    ${confCell("TN", c.TN, "good")}
    ${confCell("FP", c.FP, "bad")}
    ${confCell("FN", c.FN, "bad")}`;

  const mm = report.mismatches;
  if (mm.length === 0) {
    $("mismatch-area").innerHTML =
      `<div class="report-box"><h4>✓ All examples classified as told.</h4></div>`;
  } else {
    const rows = mm.map((m) =>
      `<div class="mismatch"><span class="badge ${m.outcome}">${m.outcome}</span>
       ${escapeHtml(m.name)} <span class="muted">— labelled ${m.label}, rule says ${m.outcome === "FP" ? "positive" : "negative"}</span></div>`
    ).join("");
    $("mismatch-area").innerHTML =
      `<div class="report-box warn"><h4>${mm.length} molecule(s) not classified as told</h4>${rows}</div>`;
  }
}

function confCell(label, n, cls) {
  return `<div class="conf-cell ${cls}"><span class="n">${n}</span><span class="l">${label}</span></div>`;
}

// ── auxiliary predicates (LLM rules) ────────────────────────────────────────
// The main hypothesis is shown on its own; the auxiliary predicates it builds on
// get their own boxes + NL, hidden behind a toggle by default.
function auxToggleLabel(n, collapsed) {
  return `${collapsed ? "▸ Show" : "▾ Hide"} ${n} auxiliary predicate${n > 1 ? "s" : ""}`;
}

function renderAux(aux) {
  const area = $("aux-area"), list = $("aux-list");
  if (!aux || !aux.length) {
    area.classList.add("hidden");
    list.innerHTML = "";
    return;
  }
  area.classList.remove("hidden");
  $("aux-toggle").textContent = auxToggleLabel(aux.length, list.classList.contains("hidden"));
  // Don't clobber a box the user is currently editing.
  if (list.contains(document.activeElement)) return;
  list.innerHTML = aux.map((a) => `
    <div class="aux-item">
      <div class="aux-name mono">${escapeHtml(a.name)}</div>
      <div class="rule-split">
        <textarea class="rule-text aux-rule" data-name="${escapeHtml(a.name)}" rows="2">${escapeHtml(a.rule || "")}</textarea>
        <div class="rule-nl-box">
          <div class="rule-nl-label">In words</div>
          <div class="rule-nl">${escapeHtml(a.nl || "—")}</div>
        </div>
      </div>
    </div>`).join("");
}

$("aux-toggle").onclick = () => {
  const list = $("aux-list");
  const collapsed = list.classList.toggle("hidden");
  $("aux-toggle").textContent =
    auxToggleLabel(list.querySelectorAll(".aux-item").length, collapsed);
};

// The full rule sent on apply = the auxiliary definitions plus the main hypothesis
// compiled from the blocks (the block editor is canonical for the main hypothesis).
function assembledRule() {
  const aux = Array.from(document.querySelectorAll(".aux-rule"))
    .map((t) => t.value.trim()).filter(Boolean);
  const main = Blocks.compile().trim();
  return [...aux, main].filter(Boolean).join("\n\n");
}

// ── detail modal ───────────────────────────────────────────────────────────
function detailRow(k, v, { mono = false, html = false } = {}) {
  if (!v) return "";
  return `<div class="detail-row"><div class="k">${k}</div><div class="v ${mono ? "mono" : ""}">${html ? v : escapeHtml(v)}</div></div>`;
}
function openModal(id) {
  const m = molById[id];
  if (!m) return;
  const other = m.label === "pos" ? "neg" : "pos";
  const moveLabel = other === "pos" ? "→ Move to positive" : "→ Move to negative";
  $("modal-struct").innerHTML =
    `<img src="${depictUrl(m, 420, 220)}" alt=""><span class="zoom-hint">hover to zoom · scroll to adjust</span>`;
  $("modal-body").innerHTML = `
    <h3>${escapeHtml(m.name)}</h3>
    ${m.chebi_id ? detailRow("ChEBI ID", chebiLink(m.chebi_id), { html: true }) : ""}
    ${detailRow("SMILES", m.smiles, { mono: true })}
    ${detailRow("InChI", m.inchi, { mono: true })}
    ${detailRow("Definition", sanitizeChebiHtml(m.definition), { html: true })}
    <div class="modal-actions">
      <button class="action-btn ${other}" data-act="move">${moveLabel}</button>
      <button class="action-btn remove" data-act="remove">× Remove</button>
    </div>`;
  $("modal-body").querySelector('[data-act="move"]').onclick = () => {
    $("modal").classList.add("hidden");
    setLabel(m.id, other);
  };
  $("modal-body").querySelector('[data-act="remove"]').onclick = () => {
    $("modal").classList.add("hidden");
    removeMolecule(m.id);
  };
  $("modal").classList.remove("hidden");
}
// Hover-zoom on the structure: magnify in place around the cursor (Amazon-style).
// The depiction is an SVG, so it stays crisp at any scale. Wheel adjusts the zoom.
let structZoom = 2.5;
(() => {
  const box = $("modal-struct");
  const img = () => box.querySelector("img");
  const apply = (e) => {
    const el = img();
    if (!el || !el.offsetWidth) return;
    const r = box.getBoundingClientRect();
    // offset* are untransformed layout coords, relative to the box
    const x = (e.clientX - r.left - el.offsetLeft) / el.offsetWidth;
    const y = (e.clientY - r.top - el.offsetTop) / el.offsetHeight;
    const clamp = (v) => Math.min(1, Math.max(0, v)) * 100;
    el.style.transformOrigin = `${clamp(x)}% ${clamp(y)}%`;
    el.style.transform = `scale(${structZoom})`;
  };
  box.addEventListener("mouseenter", (e) => { box.classList.add("zooming"); apply(e); });
  box.addEventListener("mousemove", apply);
  box.addEventListener("mouseleave", () => {
    box.classList.remove("zooming");
    const el = img();
    if (el) el.style.transform = "";
  });
  box.addEventListener("wheel", (e) => {
    e.preventDefault();
    structZoom = Math.min(8, Math.max(1.25, structZoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    apply(e);
  }, { passive: false });
})();
$("modal-close").onclick = () => $("modal").classList.add("hidden");
$("modal").addEventListener("click", (e) => {
  if (e.target.id === "modal") $("modal").classList.add("hidden");
});

// ── explanation modal (chebILP + xclingo) ──────────────────────────────────
async function openExplain(id, name) {
  $("explain-title").textContent = `Why is “${name}” positive?`;
  $("explain-content").innerHTML = `<div class="muted">Building explanation…</div>`;
  $("explain-modal").classList.remove("hidden");
  try {
    const data = await api(`/api/explain?id=${encodeURIComponent(id)}`);
    $("explain-content").innerHTML = `
      <div class="explain-struct"><img src="${data.image}" alt=""></div>
      <div class="explain-text">${escapeHtml(data.text)}</div>`;
  } catch (e) {
    $("explain-content").innerHTML = `<div class="report-box warn">${escapeHtml(e.message)}</div>`;
  }
}
$("explain-close").onclick = () => $("explain-modal").classList.add("hidden");
$("explain-modal").addEventListener("click", (e) => {
  if (e.target.id === "explain-modal") $("explain-modal").classList.add("hidden");
});

// ── ChEBI class autocomplete ───────────────────────────────────────────────
let acTimer = null;
$("class-search").addEventListener("input", (e) => {
  const q = e.target.value.trim();
  selectedClass = null;
  $("fill-btn").disabled = true;
  clearTimeout(acTimer);
  if (q.length < 2) { $("class-results").classList.add("hidden"); return; }
  acTimer = setTimeout(async () => {
    const data = await api(`/api/classes/search?q=${encodeURIComponent(q)}`);
    const box = $("class-results");
    box.innerHTML = "";
    data.results.forEach((r) => {
      const div = document.createElement("div");
      div.className = "ac-item";
      div.innerHTML = `${escapeHtml(r.name)}<span class="ac-id">CHEBI:${r.id}</span><span class="ac-n">${r.n_mol} mol</span>`;
      div.onclick = () => {
        selectedClass = r;
        $("class-search").value = r.name;
        $("selected-class").textContent = `Selected: ${r.name} (CHEBI:${r.id}, ${r.n_mol} molecules)`;
        $("fill-btn").disabled = false;
        box.classList.add("hidden");
      };
      box.appendChild(div);
    });
    box.classList.toggle("hidden", data.results.length === 0);
  }, 220);
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".autocomplete")) $("class-results").classList.add("hidden");
});

$("fill-btn").onclick = async () => {
  if (!selectedClass) return;
  const data = await api("/api/session/from_class", {
    chebi_id: selectedClass.id,
    max_pos: parseInt($("max-pos").value),
    max_neg: parseInt($("max-neg").value),
  });
  // The generated predicates offered as blocks depend on the class.
  loadPredicates().finally(() => { render(data); toast(data.message); });
};

// ── add molecules via SMILES/InChI ─────────────────────────────────────────
document.querySelectorAll(".add-mol").forEach((btn) => {
  btn.onclick = async () => {
    const label = btn.dataset.label;
    const input = $(label === "pos" ? "pos-input" : "neg-input");
    const text = input.value.trim();
    if (!text) return;
    const data = await api("/api/session/add", { text, label });
    input.value = "";
    render(data); toast(data.message);
  };
});

// ── optimistic relabel / remove ─────────────────────────────────────────────
// Apply the change to the local state and re-render immediately, then sync with
// the backend. Predictions don't depend on labels, so outcomes, counts and the
// report can be recomputed locally. Only the reply to the latest mutation is
// rendered, so a slow earlier reply can't overwrite a newer local change.
let mutationSeq = 0;

function outcomeOf(label, predicted) {
  if (label === "pos") return predicted ? "TP" : "FN";
  if (label === "neg") return predicted ? "FP" : "TN";
  return predicted ? "pos_pred" : "neg_pred";
}

function recomputeLocal(state) {
  const mols = state.molecules;
  const nPos = mols.filter((m) => m.label === "pos").length;
  const nNeg = mols.filter((m) => m.label === "neg").length;
  state.counts = { pos: nPos, neg: nNeg, unlabeled: mols.length - nPos - nNeg };
  if (!state.has_rule) return;
  const conf = { TP: 0, FP: 0, TN: 0, FN: 0 };
  const mismatches = [];
  mols.forEach((m) => {
    m.outcome = outcomeOf(m.label, !!m.prediction);
    if (m.label !== "pos" && m.label !== "neg") return;
    conf[m.outcome] += 1;
    if (m.outcome === "FP" || m.outcome === "FN") {
      mismatches.push({ id: m.id, name: m.name, smiles: m.smiles, label: m.label, outcome: m.outcome });
    }
  });
  state.report = { confusion: conf, mismatches };
}

async function mutateMolecule(path, body, applyLocal) {
  const seq = ++mutationSeq;
  if (lastState) {
    applyLocal(lastState);
    recomputeLocal(lastState);
    render(lastState);
  }
  try {
    const data = await api(path, body);
    if (seq === mutationSeq) render(data);
  } catch (e) {
    // The backend rejected it — resync to its truth (toast already shown).
    if (seq === mutationSeq) api("/api/state").then(render).catch(() => {});
  }
}

function setLabel(id, label) {
  return mutateMolecule("/api/session/set_label", { id, label }, (s) => {
    const m = s.molecules.find((x) => x.id === id);
    if (m) m.label = label;
  });
}

function removeMolecule(id) {
  return mutateMolecule("/api/session/remove", { id }, (s) => {
    s.molecules = s.molecules.filter((x) => x.id !== id);
  });
}

// ── list interactions (swap / remove) ──────────────────────────────────────
document.addEventListener("click", (e) => {
  const swap = e.target.closest(".swap");
  const x = e.target.closest(".x");
  if (x && x.classList.contains("mini-btn")) {
    e.stopPropagation();
    removeMolecule(x.dataset.id);
  } else if (swap && swap.classList.contains("mini-btn")) {
    e.stopPropagation();
    setLabel(swap.dataset.id, swap.dataset.to);
  }
});

// Drag example cards between the positive and negative columns.
document.addEventListener("dragstart", (e) => {
  const card = e.target.closest && e.target.closest(".struct-card");
  if (!card) return;
  e.dataTransfer.setData("text/x-icarus-mol", JSON.stringify({ id: card.dataset.id, label: card.dataset.label }));
  e.dataTransfer.effectAllowed = "move";
  card.classList.add("drag-src");
});
document.addEventListener("dragend", (e) => {
  const card = e.target.closest && e.target.closest(".struct-card");
  if (card) card.classList.remove("drag-src");
  document.querySelectorAll(".example-col.drop-target").forEach((c) => c.classList.remove("drop-target"));
});
[["pos-col", "pos"], ["neg-col", "neg"]].forEach(([cls, label]) => {
  const col = document.querySelector(`.${cls}`);
  col.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes("text/x-icarus-mol")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    col.classList.add("drop-target");
  });
  col.addEventListener("dragleave", (e) => {
    if (!col.contains(e.relatedTarget)) col.classList.remove("drop-target");
  });
  col.addEventListener("drop", async (e) => {
    e.preventDefault();
    col.classList.remove("drop-target");
    const raw = e.dataTransfer.getData("text/x-icarus-mol");
    if (!raw) return;
    const { id, label: from } = JSON.parse(raw);
    if (from === label) return;
    setLabel(id, label);
  });
});

// ── suggestions (one-by-one / tinder) ───────────────────────────────────────
let queue = [];
let qIdx = 0;

$("suggest-btn").onclick = async () => {
  $("suggest-btn").textContent = "Finding…";
  $("suggest-btn").disabled = true;
  $("tinder-empty").textContent = "";
  try {
    const data = await api("/api/suggest", { n: parseInt($("suggest-n").value) });
    queue = data.candidates || [];
    qIdx = 0;
    showCurrent();
    revealSuggestions();
  } finally {
    $("suggest-btn").textContent = "Find similar molecules";
    $("suggest-btn").disabled = false;
  }
};

function showCurrent() {
  const t = $("tinder");
  if (qIdx >= queue.length) {
    t.classList.add("hidden");
    $("suggest-progress").textContent = "";
    $("tinder-empty").textContent = queue.length
      ? "Done with this batch — add positives and fetch again to refine."
      : "No suggestions (add at least one positive first).";
    return;
  }
  const c = queue[qIdx];
  t.classList.remove("hidden");
  $("suggest-progress").textContent = `${qIdx + 1} / ${queue.length}`;
  $("t-sim").textContent = c.similarity;
  $("t-struct").innerHTML = `<img src="/api/depict?chebi_id=${c.chebi_id}&w=320&h=250" alt="">`;
  $("t-name").textContent = c.name;
  $("t-id").innerHTML = chebiLink(c.chebi_id);
  $("t-def").innerHTML = sanitizeChebiHtml(c.definition || "");
  let pred = "";
  if (c.prediction === true) pred = `<span class="badge pos_pred">rule predicts: positive</span>`;
  else if (c.prediction === false) pred = `<span class="badge neg_pred">rule predicts: negative</span>`;
  $("t-pred").innerHTML = pred;
}

async function decide(label) {
  const c = queue[qIdx];
  if (label) {
    await api("/api/suggest/accept", { chebi_id: c.chebi_id, label });
    const state = await api("/api/state");
    render(state);
  }
  qIdx++;
  showCurrent();
  // Accepting grows the examples above, pushing the card down — bring it back.
  revealSuggestions();
}

// Show the end of the suggestions card, keeping the candidate card's top visible.
function revealSuggestions() {
  const t = $("tinder");
  const section = t.closest("section");
  revealEnd(section, t.classList.contains("hidden") ? section : t);
}
$("t-pos").onclick = () => swipeOut("pos");
$("t-neg").onclick = () => swipeOut("neg");
$("t-skip").onclick = () => decide(null);

// Swipe the suggestion card: left = positive, right = negative.
const SWIPE_THRESHOLD = 110;
let swipe = null;
let swiping = false;

function setSwipeOffset(dx) {
  const card = $("tinder-card");
  card.style.transform = `translateX(${dx}px) rotate(${dx / 18}deg)`;
  const stage = $("tinder-stage");
  stage.querySelector(".swipe-zone.pos").classList.toggle("armed", dx <= -SWIPE_THRESHOLD);
  stage.querySelector(".swipe-zone.neg").classList.toggle("armed", dx >= SWIPE_THRESHOLD);
}

function resetSwipe() {
  $("tinder-card").classList.remove("dragging");
  $("tinder-stage").classList.remove("swiping");
  setSwipeOffset(0);
}

async function swipeOut(label) {
  if (swiping || qIdx >= queue.length) return;
  swiping = true;
  const card = $("tinder-card");
  card.classList.remove("dragging");
  $("tinder-stage").classList.remove("swiping");
  setSwipeOffset(label === "pos" ? -500 : 500);
  card.style.opacity = "0";
  try {
    await new Promise((r) => setTimeout(r, 220));
    await decide(label);
  } finally {
    card.style.transition = "none";
    card.style.opacity = "";
    setSwipeOffset(0);
    card.offsetHeight; // flush so the reset does not animate
    card.style.transition = "";
    swiping = false;
  }
}

$("tinder-card").addEventListener("pointerdown", (e) => {
  if (swiping || e.button !== 0 || e.target.closest("a")) return;
  swipe = { x: e.clientX, id: e.pointerId, dx: 0 };
  $("tinder-card").setPointerCapture(e.pointerId);
  $("tinder-card").classList.add("dragging");
  $("tinder-stage").classList.add("swiping");
});
$("tinder-card").addEventListener("pointermove", (e) => {
  if (!swipe || e.pointerId !== swipe.id) return;
  swipe.dx = e.clientX - swipe.x;
  setSwipeOffset(swipe.dx);
});
function endSwipe(e) {
  if (!swipe || e.pointerId !== swipe.id) return;
  const dx = swipe.dx;
  swipe = null;
  if (Math.abs(dx) >= SWIPE_THRESHOLD) swipeOut(dx < 0 ? "pos" : "neg");
  else resetSwipe();
}
$("tinder-card").addEventListener("pointerup", endSwipe);
$("tinder-card").addEventListener("pointercancel", endSwipe);
$("tinder-card").addEventListener("dragstart", (e) => e.preventDefault());

// ── learn (background learner run with live output) ─────────────────────────
let learnPoll = null;

const METHOD_LABELS = { popper: "Popper", aleph: "Aleph", llm: "LLM" };

// Name the configured model in the method selector (and live-output title).
function renderLlmModel(model) {
  if (!model) return;
  const label = `LLM (${model})`;
  const opt = document.querySelector('#learn-method option[value="llm"]');
  if (opt) opt.textContent = label;
  METHOD_LABELS.llm = label;
}

// The timeout only bounds the ILP searches; the LLM call is bounded by the CLI.
$("learn-method").addEventListener("change", () => {
  $("timeout-wrap").style.display =
    $("learn-method").value === "llm" ? "none" : "";
});

$("learn-btn").onclick = async () => {
  const method = $("learn-method").value;
  const label = METHOD_LABELS[method] || method;
  $("learn-btn").disabled = true;
  $("learn-status").textContent = `Running ${label}…`;
  $("learn-progress").classList.remove("hidden");
  document.querySelector(".lp-title").textContent = `${label} output`;
  $("learn-tail").textContent = "starting…";
  $("learn-full").textContent = "";
  revealEnd($("learn-progress"));
  try {
    await api("/api/learn", { method, timeout: parseInt($("timeout").value) });
  } catch (err) {
    $("learn-status").textContent = "";
    $("learn-btn").disabled = false;
    return;
  }
  const run = learnPoll = {};
  setTimeout(() => pollLearn(run), 500);
};

// One poll in flight at a time (next one scheduled only after this reply), tied to
// its run. A fixed setInterval let several slow replies overlap; each saw `done`
// and re-fired the render/toast/scroll-to-rule, dragging the view back down.
async function pollLearn(run) {
  if (run !== learnPoll) return;
  let p;
  try {
    p = await fetch("/api/learn/progress", { headers: { "X-Session-Id": SID } })
      .then((r) => r.json());
  } catch (e) {
    if (run === learnPoll) setTimeout(() => pollLearn(run), 500);
    return;
  }
  if (run !== learnPoll) return;

  if (p.tail && p.tail.length) $("learn-tail").textContent = p.tail.join("\n");
  $("learn-full").textContent = p.log || "";
  autoScroll($("learn-full"));

  if (!p.done) {
    setTimeout(() => pollLearn(run), 500);
  } else {
    learnPoll = null;
    $("learn-btn").disabled = false;
    $("learn-status").textContent = "";
    if (p.error) {
      toast(p.error);
    } else {
      // A learn may have written new aux predicates (LLM) — refresh the palette,
      // then render so the returned rule loads with those blocks available.
      loadPredicates().finally(() => {
        render(p.state);
        toast(p.message);
        if (queue.length) showCurrent();  // refresh predictions on the visible candidate
        // Show the learned rule; the output panel stays visible above it if it fits.
        requestAnimationFrame(() =>
          revealEnd($("rule-area").closest("section"), $("learn-progress")));
      });
    }
  }
}

// Scroll the window down just enough to bring `el`'s bottom into view, but never
// so far that `topEl`'s top slides under the sticky header. Never scrolls up.
function revealEnd(el, topEl = el) {
  if (!el || el.classList.contains("hidden")) return;
  const header = document.querySelector("header");
  const headerH = header ? header.getBoundingClientRect().height : 0;
  const pad = 12;
  const overflow = el.getBoundingClientRect().bottom + pad - window.innerHeight;
  const room = topEl.getBoundingClientRect().top - headerH - pad;
  const dy = Math.min(overflow, room);
  if (dy > 0) window.scrollBy({ top: dy, behavior: "smooth" });
}

function autoScroll(el) {
  if (!el.classList.contains("hidden")) el.scrollTop = el.scrollHeight;
}

$("toggle-log").onclick = () => {
  const full = $("learn-full");
  const hidden = full.classList.toggle("hidden");
  $("learn-tail").classList.toggle("hidden", !hidden);
  $("toggle-log").textContent = hidden ? "show full output" : "hide full output";
  autoScroll(full);
};

// ── rule editing ─────────────────────────────────────────────────────────────
$("apply-rule-btn").onclick = async () => {
  const rule = assembledRule();
  try {
    const data = await api("/api/rule/edit", { rule });
    editorDirty = false;
    // The applied rule becomes a new selected entry; force the editor to reload it.
    lastRuleSig = null;
    render(data);
    toast(data.message);
    renderChanges(data.changed);
  } catch (e) { /* toast already shown */ }
};

// Discard unsaved edits, restoring the selected rule (no new history entry).
$("revert-rule-btn").onclick = () => {
  editorDirty = false;
  lastRuleSig = null;   // force a reload of the editor from the selected rule
  if (lastState) render(lastState);
  $("change-area").innerHTML = "";
};

// Aux-predicate text edits mark the rule dirty (main hypothesis edits come through
// the block editor's onChange). The advanced Prolog box syncs on change (blur).
$("rule-area").addEventListener("input", (e) => {
  if (e.target.matches(".aux-rule")) { editorDirty = true; updateEditedFlag(); }
});
$("rule-text").addEventListener("input", () => { editorDirty = true; updateEditedFlag(); });
$("rule-text").addEventListener("change", () => {
  // Parse hand-edited Prolog back into blocks so the two views stay in sync.
  Blocks.load($("rule-text").value);
  editorDirty = true;
  updateEditedFlag();
});

// Select a rule from the history — makes it the current, editable rule.
$("rule-history-list").addEventListener("click", async (e) => {
  const item = e.target.closest(".rh-item");
  if (!item) return;
  const id = item.dataset.id;
  if (lastState && id === lastState.selected_rule_id) return;
  const data = await api("/api/rule/select", { id });
  render(data);
  $("change-area").innerHTML = "";
});

// Re-score every history rule against the current examples (clears stale flags).
$("reeval-btn").onclick = async () => {
  const data = await api("/api/rules/reevaluate", {});
  render(data);
  toast(data.message);
};

function renderChanges(changed) {
  if (!changed || !changed.length) { $("change-area").innerHTML = ""; return; }
  const rows = changed.map((c) =>
    `<div class="change-line">${escapeHtml(c.name)}
     <span class="muted">(${c.label})</span>:
     <span class="badge ${c.from ? "pos_pred" : "neg_pred"}">${c.from ? "positive" : "negative"}</span> →
     <span class="badge ${c.to ? "pos_pred" : "neg_pred"}">${c.to ? "positive" : "negative"}</span></div>`
  ).join("");
  $("change-area").innerHTML =
    `<div class="report-box"><h4>${changed.length} classification(s) changed by your edit</h4>${rows}</div>`;
}

// ── reset ────────────────────────────────────────────────────────────────────
$("reset-btn").onclick = async () => {
  const data = await api("/api/session/reset", {});
  render(data);
  loadPredicates();  // no class any more → no class-specific generated predicates
  queue = []; qIdx = 0;
  $("tinder").classList.add("hidden");
  $("tinder-empty").textContent = "";
  $("change-area").innerHTML = "";
  $("class-search").value = "";
  $("selected-class").textContent = "";
  toast("Session cleared.");
};

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    $("modal").classList.add("hidden");
    $("explain-modal").classList.add("hidden");
  }
});

// ── block editor wiring ──────────────────────────────────────────────────────
Blocks.init({
  palette: $("block-palette"),
  canvas: $("block-canvas"),
  onChange: () => {
    // A block edit: mark dirty and mirror the compiled Prolog into the advanced box.
    editorDirty = true;
    syncAdvancedFromBlocks();
    updateEditedFlag();
  },
});

let catalogLoaded = false;
async function loadPredicates() {
  try {
    const data = await api("/api/predicates");
    Blocks.setCatalog(data.catalog, data.aux, data.target);
    catalogLoaded = true;
  } catch (e) { /* toast already shown */ }
}

// initial load
loadPredicates();
api("/api/state").then(render);
