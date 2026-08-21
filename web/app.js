"use strict";

const $ = (id) => document.getElementById(id);
let selectedClass = null;
let lastState = null;
let molById = {};

async function api(path, body) {
  const opts = { method: body === undefined ? "GET" : "POST" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
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
  `/api/depict?id=${encodeURIComponent(m.id)}&w=${w}&h=${h}&v=${hashStr(m.smiles || m.id)}`;

// ── rendering ────────────────────────────────────────────────────────────
function render(state) {
  lastState = state;
  molById = {};
  state.molecules.forEach((m) => (molById[m.id] = m));
  const pos = state.molecules.filter((m) => m.label === "pos");
  const neg = state.molecules.filter((m) => m.label === "neg");

  $("pos-count").textContent = pos.length;
  $("neg-count").textContent = neg.length;
  $("counts").textContent =
    `${state.counts.pos} positive · ${state.counts.neg} negative`;

  renderStructGrid($("pos-list"), pos, state.has_rule);
  renderStructGrid($("neg-list"), neg, state.has_rule);

  if (state.has_rule) {
    $("rule-area").classList.remove("hidden");
    const ta = $("rule-text");
    if (document.activeElement !== ta) ta.value = state.current_rule || "";
    $("edited-flag").textContent = state.edited ? "(edited — differs from learned rule)" : "";
    $("rule-nl").textContent = state.rule_nl || "—";
    renderReport(state.report);
  } else {
    $("rule-area").classList.add("hidden");
  }
}

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
    card.innerHTML = `
      ${badge}
      <div class="card-ctrls">
        <button class="mini-btn swap" data-id="${m.id}" data-to="${other}" title="move">${swap}</button>
        <button class="mini-btn x" data-id="${m.id}" title="remove">×</button>
      </div>
      <div class="thumb"><img loading="lazy" src="${depictUrl(m, 120, 100)}" alt=""></div>
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

// ── detail modal ───────────────────────────────────────────────────────────
function detailRow(k, v, { mono = false, html = false } = {}) {
  if (!v) return "";
  return `<div class="detail-row"><div class="k">${k}</div><div class="v ${mono ? "mono" : ""}">${html ? v : escapeHtml(v)}</div></div>`;
}
function openModal(id) {
  const m = molById[id];
  if (!m) return;
  $("modal-struct").innerHTML = `<img src="${depictUrl(m, 420, 220)}" alt="">`;
  $("modal-body").innerHTML = `
    <h3>${escapeHtml(m.name)}</h3>
    ${m.chebi_id ? detailRow("ChEBI ID", chebiLink(m.chebi_id), { html: true }) : ""}
    ${detailRow("SMILES", m.smiles, { mono: true })}
    ${detailRow("InChI", m.inchi, { mono: true })}
    ${detailRow("Definition", sanitizeChebiHtml(m.definition), { html: true })}`;
  $("modal").classList.remove("hidden");
}
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
  render(data); toast(data.message);
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

// ── list interactions (swap / remove) ──────────────────────────────────────
document.addEventListener("click", async (e) => {
  const swap = e.target.closest(".swap");
  const x = e.target.closest(".x");
  if (x && x.classList.contains("mini-btn")) {
    e.stopPropagation();
    const data = await api("/api/session/remove", { id: x.dataset.id });
    render(data);
  } else if (swap && swap.classList.contains("mini-btn")) {
    e.stopPropagation();
    const data = await api("/api/session/set_label", { id: swap.dataset.id, label: swap.dataset.to });
    render(data);
  }
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
}
$("t-pos").onclick = () => decide("pos");
$("t-neg").onclick = () => decide("neg");
$("t-skip").onclick = () => decide(null);

// ── learn (background Popper run with live output) ──────────────────────────
let learnPoll = null;

$("learn-btn").onclick = async () => {
  $("learn-btn").disabled = true;
  $("learn-status").textContent = "Running Popper…";
  $("learn-progress").classList.remove("hidden");
  $("learn-tail").textContent = "starting…";
  $("learn-full").textContent = "";
  try {
    await api("/api/learn", { timeout: parseInt($("timeout").value) });
  } catch (err) {
    $("learn-status").textContent = "";
    $("learn-btn").disabled = false;
    return;
  }
  learnPoll = setInterval(pollLearn, 500);
};

async function pollLearn() {
  let p;
  try {
    p = await fetch("/api/learn/progress").then((r) => r.json());
  } catch (e) { return; }

  if (p.tail && p.tail.length) $("learn-tail").textContent = p.tail.join("\n");
  $("learn-full").textContent = p.log || "";
  autoScroll($("learn-full"));

  if (p.done) {
    clearInterval(learnPoll); learnPoll = null;
    $("learn-btn").disabled = false;
    $("learn-status").textContent = "";
    if (p.error) {
      toast(p.error);
    } else {
      render(p.state);
      toast(p.message);
      if (queue.length) showCurrent();  // refresh predictions on the visible candidate
    }
  }
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
  const rule = $("rule-text").value.trim();
  try {
    const data = await api("/api/rule/edit", { rule });
    render(data);
    toast(data.message);
    renderChanges(data.changed);
  } catch (e) { /* toast already shown */ }
};

$("revert-rule-btn").onclick = async () => {
  const data = await api("/api/rule/reset", {});
  render(data);
  $("change-area").innerHTML = "";
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

// initial load
api("/api/state").then(render);
