"use strict";
// ── Visual building-block rule editor ───────────────────────────────────────
// A rule is an OR of clauses ("alternatives"). Each clause has a molecule area
// (global properties), freely-placed atom areas (per-atom properties), and bonds
// drawn between two atom areas. The model compiles to the exact
// `concept(V0) :- …` Prolog the ILP pipeline consumes, and parses back from it,
// so learned rules render as blocks and hand-edits round-trip.
//
// compileModel / parseRule / describe are DOM-free (unit-testable in Node); the
// Editor object below wires them to the palette + canvas UI.
(function (global) {

  // Built-in element symbol → name (mirrors icarus/predicates.COMMON_ELEMENTS),
  // so describe() labels elements even before the catalog loads / in Node tests.
  const ELEMENTS = {
    c: "Carbon", n: "Nitrogen", o: "Oxygen", s: "Sulfur", p: "Phosphorus",
    f: "Fluorine", cl: "Chlorine", br: "Bromine", i: "Iodine", h: "Hydrogen",
    b: "Boron", si: "Silicon", se: "Selenium",
  };

  let TARGET = "concept";

  // ── readable description of a ground predicate ────────────────────────────
  // Returns {label, scope}. `label` is the compact chip text; `scope` is a hint
  // (molecule|atom|bond) used when the predicate is added from the palette (the
  // parser places predicates by their actual argument role, not this hint).
  function describe(pred) {
    let m;
    if (ELEMENTS[pred]) return { label: ELEMENTS[pred], scope: "atom" };
    if (/^[a-z]{1,2}$/.test(pred)) return { label: pred.toUpperCase(), scope: "atom" };

    if (pred === "charge0") return { label: "charge 0", scope: "atom" };
    if (pred === "charge_p") return { label: "charge +", scope: "atom" };
    if (pred === "charge_n") return { label: "charge −", scope: "atom" };
    if ((m = /^charge_m(\d+)$/.exec(pred))) return { label: "charge −" + m[1], scope: "atom" };
    if ((m = /^charge(\d+)$/.exec(pred))) return { label: "charge +" + m[1], scope: "atom" };

    if ((m = /^has_(\d+)_hs$/.exec(pred))) return { label: m[1] + " H", scope: "atom" };
    if ((m = /^has_at_least_(\d+)_hs$/.exec(pred))) return { label: "≥" + m[1] + " H", scope: "atom" };

    if (pred === "in_ring") return { label: "in ring", scope: "atom" };
    if ((m = /^in_ring(\d+)$/.exec(pred))) return { label: m[1] + "-ring", scope: "atom" };

    if ((m = /^cip_code_([A-Za-z])$/.exec(pred))) return { label: "CIP " + m[1].toUpperCase(), scope: "atom" };
    if ((m = /^steroid_(\d+)$/.exec(pred))) return { label: "steroid " + m[1], scope: "atom" };

    if (pred === "has_bond_to") return { label: "bonded", scope: "bond" };
    if ((m = /^bSTEREO([A-Z]+)$/.exec(pred))) return { label: "stereo " + m[1].toLowerCase(), scope: "bond" };
    if (/^b[A-Z]+$/.test(pred)) return { label: pred.slice(1).toLowerCase(), scope: "bond" };

    if (pred === "aromatic" || pred === "aliphatic") return { label: pred, scope: "molecule" };
    if ((m = /^net_charge_(\w+)$/.exec(pred))) return { label: "net " + m[1], scope: "molecule" };

    // aux_* or anything unrecognised → humanised name (drop aux_ + id suffix).
    let name = pred.startsWith("aux_") ? pred.slice(4) : pred;
    name = name.replace(/_\d{4,}$/, "").replace(/_/g, " ").trim();
    return { label: name || pred, scope: "molecule" };
  }

  // Mutual-exclusivity group of a ground predicate, or null when it can freely
  // coexist with others on the same atom/molecule. An atom is exactly one element,
  // carries one formal charge, one exact H-count, one CIP code, one steroid
  // position; a molecule is aromatic xor aliphatic and has one net charge. Ring
  // membership is intentionally ungrouped — a fused-ring atom is in several rings.
  function groupOf(pred) {
    if (ELEMENTS[pred] || /^[a-z]{1,2}$/.test(pred)) return "element";
    if (/^charge(0|_p|_n|_m?\d+)$/.test(pred)) return "charge";
    if (/^has_\d+_hs$/.test(pred)) return "h_exact";
    if (/^has_at_least_\d+_hs$/.test(pred)) return "h_atleast";
    if (/^cip_code_/.test(pred)) return "cip";
    if (/^steroid_\d+$/.test(pred)) return "steroid";
    if (pred === "aromatic" || pred === "aliphatic") return "aromaticity";
    if (/^net_charge_/.test(pred)) return "net_charge";
    return null;
  }

  // Registered readable labels (from the loaded catalog/aux) win over the generic
  // describe() label, so choice options and aux descriptions read nicely.
  const LABELS = {};
  function labelFor(pred) {
    return (pred in LABELS) ? LABELS[pred] : describe(pred).label;
  }

  // ── Prolog helpers (bracket-depth aware, mirrors icarus/ilp) ──────────────
  function splitClauses(text) {
    const t = (text || "").split("\n").filter((l) => !l.trim().startsWith("%")).join("\n");
    let depth = 0, cur = "", out = [];
    for (const ch of t) {
      if ("([{".includes(ch)) depth++;
      else if (")]}".includes(ch)) depth = Math.max(0, depth - 1);
      if (ch === "." && depth === 0) { const c = cur.trim(); if (c) out.push(c + "."); cur = ""; }
      else cur += ch;
    }
    const tail = cur.trim();
    if (tail) out.push(tail.endsWith(".") ? tail : tail + ".");
    return out;
  }

  function splitTop(s, sep) {
    let depth = 0, cur = "", out = [];
    for (const ch of s) {
      if ("([{".includes(ch)) depth++;
      else if (")]}".includes(ch)) depth = Math.max(0, depth - 1);
      if (ch === sep && depth === 0) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }

  function parseLiteral(lit) {
    lit = lit.trim();
    const i = lit.indexOf("(");
    if (i < 0) return { pred: lit, args: [] };
    const pred = lit.slice(0, i).trim();
    const inner = lit.slice(i + 1, lit.lastIndexOf(")"));
    const args = inner.trim() ? splitTop(inner, ",").map((a) => a.trim()) : [];
    return { pred, args };
  }

  const IS_VAR = (a) => /^[A-Z_][A-Za-z0-9_]*$/.test(a);

  // ── compile: model → Prolog text ──────────────────────────────────────────
  function compileModel(model) {
    if (!model || !model.clauses) return "";
    const lines = [];
    for (const cl of model.clauses) {
      if (cl.rawClause != null) { lines.push(cl.rawClause.trim()); continue; }
      const varOf = {};
      (cl.atoms || []).forEach((a, i) => { varOf[a.id] = "V" + (i + 1); });
      const lits = [];
      (cl.atoms || []).forEach((a, i) => {
        const v = "V" + (i + 1);
        lits.push(`has_atom(V0,${v})`);
        (a.chips || []).forEach((ch) => lits.push(`${ch.pred}(${v})`));
      });
      (cl.molChips || []).forEach((ch) => lits.push(`${ch.pred}(V0)`));
      (cl.bonds || []).forEach((b) => {
        if (varOf[b.a] && varOf[b.b]) lits.push(`${b.pred}(${varOf[b.a]},${varOf[b.b]})`);
      });
      if (!lits.length) continue;
      lines.push(`${TARGET}(V0) :- ${lits.join(", ")}.`);
    }
    return lines.join("\n");
  }

  // ── parse: Prolog text → model ────────────────────────────────────────────
  let _id = 0;
  const newId = () => "a" + (++_id);

  function parseRule(text) {
    const model = { clauses: [] };
    for (const clauseText of splitClauses(text)) {
      if (!clauseText.includes(":-")) continue;
      const [headStr, bodyStr] = [clauseText.slice(0, clauseText.indexOf(":-")),
                                  clauseText.slice(clauseText.indexOf(":-") + 2)];
      const head = parseLiteral(headStr.trim());
      const body = bodyStr.replace(/\.\s*$/, "");
      const molVar = head.args[0];
      const literals = splitTop(body, ",").map((l) => l.trim()).filter(Boolean).map(parseLiteral);

      // A clause is "raw" (kept verbatim) when it uses something the block grammar
      // can't place: a non-target head, an arity≥3 literal, or a binary literal
      // touching the molecule variable / non-variable arguments.
      let raw = head.pred !== TARGET || !molVar;
      for (const { pred, args } of literals) {
        if (pred === "has_atom") continue;
        if (args.length === 1) {
          if (!IS_VAR(args[0])) raw = true;
        } else if (args.length === 2) {
          if (!IS_VAR(args[0]) || !IS_VAR(args[1]) || args[0] === molVar || args[1] === molVar) raw = true;
        } else { raw = true; }
      }
      if (raw) { model.clauses.push({ rawClause: clauseText.trim() }); continue; }

      // Structured clause. Atom areas = every non-molecule variable, in order of
      // first appearance (whether introduced by has_atom or used directly).
      const areaByVar = {};
      const atoms = [];
      const ensureAtom = (v) => {
        if (!areaByVar[v]) {
          const a = { id: newId(), var: v, chips: [], x: 24 + atoms.length * 172, y: 20 };
          areaByVar[v] = a; atoms.push(a);
        }
        return areaByVar[v];
      };
      literals.forEach(({ pred, args }) => {
        args.forEach((a) => { if (IS_VAR(a) && a !== molVar) ensureAtom(a); });
      });

      const molChips = [], bonds = [];
      literals.forEach(({ pred, args }) => {
        if (pred === "has_atom") return;
        if (args.length === 1) {
          if (args[0] === molVar) molChips.push({ pred });
          else areaByVar[args[0]].chips.push({ pred });
        } else if (args.length === 2) {
          bonds.push({ a: areaByVar[args[0]].id, b: areaByVar[args[1]].id, pred });
        }
      });
      atoms.forEach((a) => delete a.var);
      model.clauses.push({ atoms, molChips, bonds, _needsLayout: true });
    }
    return model;
  }

  function emptyModel() { return { clauses: [{ atoms: [], molChips: [], bonds: [] }] }; }

  // Force-directed (Fruchterman–Reingold) layout of a clause's atoms within a
  // WxH area: atoms repel each other, bonds pull their endpoints together, so a
  // star like A–B, A–C relaxes into a spread where both bonds stay visible
  // instead of collapsing onto one line. Writes top-left {x,y} back onto each
  // atom (boxW/boxH keep boxes inside the area). Deterministic (circle seed).
  function forceLayout(atoms, bonds, W, H, boxW, boxH) {
    const n = atoms.length;
    if (!n) return;
    const cx = W / 2, cy = H / 2;
    if (n === 1) { atoms[0].x = Math.round(cx - boxW / 2); atoms[0].y = Math.round(cy - boxH / 2); return; }

    const idx = {};
    atoms.forEach((a, i) => { idx[a.id] = i; });
    const R = Math.min(W, H) * 0.32;
    const pos = atoms.map((_, i) => ({
      x: cx + Math.cos((2 * Math.PI * i) / n) * R,
      y: cy + Math.sin((2 * Math.PI * i) / n) * R,
    }));
    const edges = (bonds || [])
      .filter((b) => idx[b.a] != null && idx[b.b] != null)
      .map((b) => [idx[b.a], idx[b.b]]);

    const k = Math.sqrt((W * H) / n) * 0.62;   // ideal separation
    let temp = W * 0.12;
    for (let it = 0; it < 220; it++) {
      const disp = pos.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
          let d = Math.hypot(dx, dy) || 0.01;
          const f = (k * k) / d, ux = dx / d, uy = dy / d;
          disp[i].x += ux * f; disp[i].y += uy * f;
          disp[j].x -= ux * f; disp[j].y -= uy * f;
        }
      }
      for (const [a, b] of edges) {
        let dx = pos[a].x - pos[b].x, dy = pos[a].y - pos[b].y;
        let d = Math.hypot(dx, dy) || 0.01;
        const f = (d * d) / k, ux = dx / d, uy = dy / d;
        disp[a].x -= ux * f; disp[a].y -= uy * f;
        disp[b].x += ux * f; disp[b].y += uy * f;
      }
      for (let i = 0; i < n; i++) {
        let d = Math.hypot(disp[i].x, disp[i].y) || 0.01;
        pos[i].x += (disp[i].x / d) * Math.min(d, temp);
        pos[i].y += (disp[i].y / d) * Math.min(d, temp);
      }
      temp = Math.max(temp * 0.95, 2);
    }
    // Normalise to fill the area with a margin, then convert centre→top-left.
    const xs = pos.map((p) => p.x), ys = pos.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const mL = boxW / 2 + 8, mR = boxW / 2 + 8, mT = boxH / 2 + 8, mB = boxH / 2 + 8;
    const sx = (maxX - minX) > 1 ? (W - mL - mR) / (maxX - minX) : 1;
    const sy = (maxY - minY) > 1 ? (H - mT - mB) / (maxY - minY) : 1;
    atoms.forEach((a, i) => {
      const px = mL + (pos[i].x - minX) * sx;
      const py = mT + (pos[i].y - minY) * sy;
      a.x = Math.round(px - boxW / 2);
      a.y = Math.round(py - boxH / 2);
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  //  DOM editor
  // ════════════════════════════════════════════════════════════════════════
  const Editor = {
    model: emptyModel(),
    catalog: [],
    aux: [],
    paletteEl: null,
    canvasEl: null,
    onChangeCb: null,
    loadedText: "",
    dirty: false,
    selected: null,   // {clauseIdx, kind:'mol'|'atom', atomId?} for click-to-add

    init(opts) {
      this.paletteEl = opts.palette;
      this.canvasEl = opts.canvas;
      this.onChangeCb = opts.onChange || null;
      this.canvasEl.addEventListener("click", (e) => this.onCanvasClick(e));
      this.renderCanvas();
    },

    setCatalog(catalog, aux, target) {
      this.catalog = catalog || [];
      this.aux = aux || [];
      if (target) TARGET = target;
      // Register readable labels for every ground predicate the catalog names.
      for (const b of [...this.catalog, ...this.aux]) {
        const inp = b.input || {};
        if (inp.type === "toggle" && inp.pred) LABELS[inp.pred] = b.label;
        if (inp.type === "choice") (inp.options || []).forEach((o) => { LABELS[o.pred] = b.label + ": " + o.label; });
      }
      // Element symbols → nice names.
      for (const b of this.catalog) {
        if (b.category === "Element" && b.input && b.input.type === "toggle") {
          ELEMENTS[b.input.pred] = b.label; LABELS[b.input.pred] = b.label;
        }
      }
      this.renderPalette();
    },

    changed() {
      this.dirty = true;
      if (this.onChangeCb) this.onChangeCb();
    },

    // Load a rule's text into the editor (rebuilds the model + UI).
    load(text) {
      this.loadedText = text || "";
      this.model = (text && text.trim()) ? parseRule(text) : emptyModel();
      if (!this.model.clauses.length) this.model = emptyModel();
      this.dirty = false;
      this.selected = null;
      this.renderCanvas();
    },

    compile() { return compileModel(this.model); },
    isDirty() { return this.dirty; },

    // Add a predicate chip to an area's chip list, dropping any exact duplicate and
    // any mutually-exclusive chip (same group) first — so a second element / charge
    // / H-count replaces the first rather than stacking a contradiction.
    addChip(list, pred) {
      const g = groupOf(pred);
      for (let i = list.length - 1; i >= 0; i--) {
        const p = list[i].pred;
        if (p === pred || (g && groupOf(p) === g)) list.splice(i, 1);
      }
      list.push({ pred });
    },

    // ── palette ─────────────────────────────────────────────────────────────
    renderPalette() {
      const el = this.paletteEl;
      el.innerHTML = "";
      // Bond/relation predicates aren't dragged from the palette — a bond needs two
      // atoms, so it's drawn with the 🔗 connect handle and its type is chosen on the
      // bond itself. So the palette lists only molecule- and atom-scope blocks.
      const all = [...this.catalog, ...this.aux].filter((b) => b.scope !== "bond");
      // Popular row first; the collapsible groups hold everything else (so a block
      // isn't listed twice).
      const groups = {};
      const order = [];
      for (const b of all) {
        if (b.popular) continue;
        if (!groups[b.category]) { groups[b.category] = []; order.push(b.category); }
        groups[b.category].push(b);
      }
      const popular = all.filter((b) => b.popular);
      if (popular.length) {
        const row = document.createElement("div");
        row.className = "palette-popular";
        row.innerHTML = `<div class="palette-group-title">Common</div>`;
        const wrap = document.createElement("div");
        wrap.className = "palette-blocks";
        popular.forEach((b) => wrap.appendChild(this.paletteBlock(b)));
        row.appendChild(wrap);
        el.appendChild(row);
      }
      // Collapsible advanced groups.
      const more = document.createElement("details");
      more.className = "palette-more";
      more.innerHTML = `<summary>More predicates…</summary>`;
      order.forEach((cat) => {
        const g = document.createElement("div");
        g.className = "palette-group";
        g.innerHTML = `<div class="palette-group-title">${escapeHtml(cat)}</div>`;
        const wrap = document.createElement("div");
        wrap.className = "palette-blocks";
        groups[cat].forEach((b) => wrap.appendChild(this.paletteBlock(b)));
        g.appendChild(wrap);
        more.appendChild(g);
      });
      el.appendChild(more);
    },

    // Build one palette block element. Returns the ground predicate via getPred().
    paletteBlock(b) {
      const div = document.createElement("div");
      div.className = "palette-block scope-" + b.scope;
      div.dataset.scope = b.scope;
      const inp = b.input || {};
      let control = "";
      if (inp.type === "choice") {
        const opts = (inp.options || []).map((o) =>
          `<option value="${escapeHtml(o.pred)}"${o.pred === inp.default ? " selected" : ""}>${escapeHtml(o.label)}</option>`).join("");
        control = `<select class="pb-choice">${opts}</select>`;
      } else if (inp.type === "element") {
        control = `<input class="pb-element" type="text" maxlength="2" placeholder="e.g. Cl" size="3">`;
      }
      div.innerHTML = `<span class="pb-label">${escapeHtml(b.label)}</span>${control}` +
        (b.description ? `<span class="pb-desc" title="${escapeHtml(b.description)}">ⓘ</span>` : "");

      const getPred = () => {
        if (inp.type === "toggle") return inp.pred;
        if (inp.type === "choice") return div.querySelector(".pb-choice").value;
        if (inp.type === "element") {
          const v = (div.querySelector(".pb-element").value || "").trim().toLowerCase();
          return v || null;
        }
        return null;
      };

      div.draggable = true;
      div.addEventListener("dragstart", (e) => {
        const pred = getPred();
        if (!pred) { e.preventDefault(); return; }
        e.dataTransfer.setData("text/plain", JSON.stringify({ pred, scope: b.scope }));
        e.dataTransfer.effectAllowed = "copy";
        Editor._dragScope = b.scope;  // getData is blocked during dragover, so stash it
      });
      div.addEventListener("dragend", () => { Editor._dragScope = null; });
      // Click-to-add to the selected area (accessibility / no-drag fallback).
      div.addEventListener("click", (e) => {
        if (e.target.closest(".pb-choice, .pb-element")) return;
        const pred = getPred();
        if (pred) this.addToSelected(pred, b.scope);
      });
      return div;
    },

    // ── canvas render ─────────────────────────────────────────────────────────
    renderCanvas() {
      const el = this.canvasEl;
      el.innerHTML = "";
      this.model.clauses.forEach((cl, ci) => el.appendChild(this.clauseCard(cl, ci)));
      const add = document.createElement("button");
      add.className = "add-clause ghost";
      add.textContent = "+ OR alternative";
      add.onclick = () => { this.model.clauses.push({ atoms: [], molChips: [], bonds: [] }); this.renderCanvas(); this.changed(); };
      el.appendChild(add);
      // Auto-arrange freshly loaded/reflowed clauses (canvas is now in the DOM, so
      // its width is known), then size each canvas to its atoms and draw the bonds.
      this.model.clauses.forEach((cl, ci) => {
        if (cl.rawClause) return;
        if (cl._needsLayout) { this.autoLayout(cl, ci); cl._needsLayout = false; }
        this.fitCanvas(ci);
        this.drawBonds(ci);
      });
    },

    // Run the force-directed layout for a clause and apply it to the DOM boxes.
    autoLayout(cl, ci) {
      if (!cl.atoms || cl.atoms.length < 2) {
        if (cl.atoms && cl.atoms.length === 1) { cl.atoms[0].x = 24; cl.atoms[0].y = 20; }
        return;
      }
      const canvas = this.canvasEl.querySelector(`.atom-canvas[data-ci="${ci}"]`);
      const W = Math.max(320, (canvas ? canvas.clientWidth : 0) || 520);
      const H = Math.max(240, Math.min(520, 150 + 52 * cl.atoms.length));
      forceLayout(cl.atoms, cl.bonds, W, H, 138, 78);
      if (canvas) {
        cl.atoms.forEach((a) => {
          const box = canvas.querySelector(`.atom-box[data-aid="${a.id}"]`);
          if (box) { box.style.left = a.x + "px"; box.style.top = a.y + "px"; }
        });
      }
    },

    // Grow the atom-canvas so every (variable-height) atom box fits without clipping.
    fitCanvas(ci) {
      const canvas = this.canvasEl.querySelector(`.atom-canvas[data-ci="${ci}"]`);
      if (!canvas) return;
      let bottom = 0;
      canvas.querySelectorAll(".atom-box").forEach((b) => { bottom = Math.max(bottom, b.offsetTop + b.offsetHeight); });
      canvas.style.height = Math.max(190, bottom + 44) + "px";
    },

    clauseCard(cl, ci) {
      const card = document.createElement("div");
      card.className = "clause-card";
      card.dataset.ci = ci;

      const head = document.createElement("div");
      head.className = "clause-head";
      head.innerHTML = `<span class="clause-title">${ci === 0 ? "Rule" : "OR — alternative " + (ci + 1)}</span>`;
      const actions = document.createElement("span");
      actions.className = "clause-head-actions";
      if (!cl.rawClause && (cl.atoms || []).length > 1) {
        const tidy = document.createElement("button");
        tidy.className = "ghost tidy-btn"; tidy.textContent = "⤢ Auto-arrange";
        tidy.title = "re-arrange atoms for a clearer layout";
        tidy.onclick = () => { cl._needsLayout = true; this.renderCanvas(); this.changed(); };
        actions.appendChild(tidy);
      }
      if (this.model.clauses.length > 1 || cl.rawClause) {
        const del = document.createElement("button");
        del.className = "mini-btn x"; del.title = "remove alternative"; del.textContent = "×";
        del.onclick = () => { this.model.clauses.splice(ci, 1); if (!this.model.clauses.length) this.model = emptyModel(); this.renderCanvas(); this.changed(); };
        actions.appendChild(del);
      }
      head.appendChild(actions);
      card.appendChild(head);

      if (cl.rawClause != null) {
        const raw = document.createElement("div");
        raw.className = "clause-raw";
        raw.innerHTML = `<div class="raw-note">Advanced clause (edit as Prolog):</div>`;
        const ta = document.createElement("textarea");
        ta.className = "rule-text raw-clause"; ta.rows = 2; ta.value = cl.rawClause;
        ta.addEventListener("input", () => { cl.rawClause = ta.value; this.changed(); });
        raw.appendChild(ta);
        card.appendChild(raw);
        return card;
      }

      // Molecule area.
      const mol = document.createElement("div");
      mol.className = "mol-area dropzone";
      mol.dataset.ci = ci;
      const selMol = this.selected && this.selected.ci === ci && this.selected.kind === "mol";
      if (selMol) mol.classList.add("selected");
      mol.innerHTML = `<span class="area-label">Whole molecule</span>`;
      (cl.molChips || []).forEach((ch, idx) => mol.appendChild(this.chipEl(ch.pred, () => { cl.molChips.splice(idx, 1); this.renderCanvas(); this.changed(); })));
      if (!(cl.molChips || []).length) mol.insertAdjacentHTML("beforeend", `<span class="area-hint">drop a molecule property here (e.g. aromatic)</span>`);
      mol.addEventListener("click", (e) => { if (!e.target.closest(".chip")) this.select({ ci, kind: "mol" }); });
      this.wireDrop(mol, "molecule", (pred) => { this.addChip(cl.molChips, pred); this.renderCanvas(); this.changed(); });
      card.appendChild(mol);

      // Atom canvas.
      const canvas = document.createElement("div");
      canvas.className = "atom-canvas";
      canvas.dataset.ci = ci;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "bond-layer");
      canvas.appendChild(svg);
      (cl.atoms || []).forEach((a) => canvas.appendChild(this.atomBox(cl, ci, a)));
      const addAtom = document.createElement("button");
      addAtom.className = "add-atom ghost";
      addAtom.textContent = "+ atom";
      addAtom.onclick = () => {
        const a = { id: newId(), chips: [], x: 24 + (cl.atoms.length % 4) * 172, y: 24 + Math.floor(cl.atoms.length / 4) * 130 };
        cl.atoms.push(a); this.renderCanvas(); this.changed();
      };
      canvas.appendChild(addAtom);
      // Drop an atom-scope predicate on empty canvas → new atom carrying it.
      this.wireDrop(canvas, "atom", (pred, e) => {
        if (e && e.target.closest(".atom-box")) return; // handled by the box
        const rect = canvas.getBoundingClientRect();
        const a = { id: newId(), chips: [{ pred }], x: (e ? e.clientX - rect.left - 60 : 40), y: (e ? e.clientY - rect.top - 20 : 40) };
        cl.atoms.push(a); this.renderCanvas(); this.changed();
      }, true);
      card.appendChild(canvas);
      return card;
    },

    atomBox(cl, ci, a) {
      const box = document.createElement("div");
      box.className = "atom-box dropzone";
      box.dataset.ci = ci; box.dataset.aid = a.id;
      box.style.left = (a.x || 0) + "px";
      box.style.top = (a.y || 0) + "px";
      if (this.selected && this.selected.ci === ci && this.selected.kind === "atom" && this.selected.atomId === a.id) box.classList.add("selected");

      const head = document.createElement("div");
      head.className = "atom-head";
      head.innerHTML = `<span class="atom-grip" title="drag to move">⠿ atom</span>`;
      const bondH = document.createElement("button");
      bondH.className = "atom-bond-handle"; bondH.title = "drag to another atom to bond"; bondH.textContent = "🔗";
      const rm = document.createElement("button");
      rm.className = "mini-btn x"; rm.title = "remove atom"; rm.textContent = "×";
      rm.onclick = (e) => {
        e.stopPropagation();
        cl.atoms = cl.atoms.filter((x) => x.id !== a.id);
        cl.bonds = (cl.bonds || []).filter((bd) => bd.a !== a.id && bd.b !== a.id);
        this.renderCanvas(); this.changed();
      };
      head.appendChild(bondH); head.appendChild(rm);
      box.appendChild(head);

      const chips = document.createElement("div");
      chips.className = "atom-chips";
      (a.chips || []).forEach((ch, idx) => chips.appendChild(this.chipEl(ch.pred, () => { a.chips.splice(idx, 1); this.renderCanvas(); this.changed(); })));
      if (!(a.chips || []).length) chips.innerHTML = `<span class="area-hint">drop atom properties</span>`;
      box.appendChild(chips);

      box.addEventListener("click", (e) => { if (!e.target.closest(".chip, .mini-btn, .atom-bond-handle")) this.select({ ci, kind: "atom", atomId: a.id }); });
      this.wireDrop(box, "atom", (pred) => { this.addChip(a.chips, pred); this.renderCanvas(); this.changed(); });
      this.wireAtomDrag(box, cl, ci, a);
      this.wireBondHandle(bondH, cl, ci, a);
      return box;
    },

    chipEl(pred, onRemove) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<span class="chip-label">${escapeHtml(labelFor(pred))}</span>`;
      const x = document.createElement("button");
      x.className = "chip-x"; x.textContent = "×"; x.title = "remove";
      x.onclick = (e) => { e.stopPropagation(); onRemove(); };
      chip.appendChild(x);
      return chip;
    },

    // ── selection (click-to-add) ──────────────────────────────────────────────
    select(sel) { this.selected = sel; this.renderCanvas(); },
    onCanvasClick(e) {
      if (e.target === this.canvasEl) { this.selected = null; this.renderCanvas(); }
    },
    addToSelected(pred, scope) {
      const s = this.selected;
      // An "any"-scope block goes wherever the user has selected (an atom if one is
      // selected, otherwise the molecule area).
      if (scope === "any") scope = (s && s.kind === "atom") ? "atom" : "molecule";
      if (scope === "molecule") {
        const ci = s && s.kind === "mol" ? s.ci : 0;
        const cl = this.model.clauses[ci]; if (!cl || cl.rawClause) return;
        this.addChip(cl.molChips, pred);
      } else if (scope === "atom") {
        if (s && s.kind === "atom") {
          const cl = this.model.clauses[s.ci];
          const a = cl.atoms.find((x) => x.id === s.atomId);
          if (a) this.addChip(a.chips, pred);
        } else {
          const ci = s ? s.ci : 0;
          const cl = this.model.clauses[ci] || this.model.clauses[0];
          if (!cl || cl.rawClause) return;
          cl.atoms.push({ id: newId(), chips: [{ pred }], x: 24 + (cl.atoms.length % 4) * 172, y: 24 });
        }
      }
      this.renderCanvas(); this.changed();
    },

    // ── drag/drop of palette blocks onto a zone ───────────────────────────────
    wireDrop(zone, scope, add, passEvent) {
      // An "any"-scope block (an arity-1 generated predicate whose atom-vs-molecule
      // role is unknown) is accepted by both the molecule area and atom boxes.
      const accepts = (ds) => ds === scope || ds === "any";
      zone.addEventListener("dragover", (e) => {
        if (accepts(Editor._dragScope)) { e.preventDefault(); zone.classList.add("drop-hover"); }
      });
      zone.addEventListener("dragleave", () => zone.classList.remove("drop-hover"));
      zone.addEventListener("drop", (e) => {
        zone.classList.remove("drop-hover");
        let payload; try { payload = JSON.parse(e.dataTransfer.getData("text/plain")); } catch (_) { return; }
        if (!payload || !accepts(payload.scope)) return;
        e.preventDefault(); e.stopPropagation();
        add(payload.pred, passEvent ? e : undefined);
      });
    },

    // ── freeform atom drag (reposition) ───────────────────────────────────────
    wireAtomDrag(box, cl, ci, a) {
      const grip = box.querySelector(".atom-grip");
      grip.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        const canvas = box.parentElement;
        const rect = canvas.getBoundingClientRect();
        const offX = e.clientX - box.offsetLeft, offY = e.clientY - box.offsetTop;
        const move = (ev) => {
          let x = Math.max(0, Math.min(ev.clientX - offX, rect.width - box.offsetWidth));
          let y = Math.max(0, Math.min(ev.clientY - offY, rect.height - box.offsetHeight));
          box.style.left = x + "px"; box.style.top = y + "px";
          a.x = x; a.y = y; this.drawBonds(ci);
        };
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          this.changed();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    },

    // ── bond drawing from an atom's handle ────────────────────────────────────
    wireBondHandle(handle, cl, ci, a) {
      handle.addEventListener("pointerdown", (e) => {
        e.preventDefault(); e.stopPropagation();
        const canvas = handle.closest(".atom-canvas");
        const svg = canvas.querySelector(".bond-layer");
        const rect = canvas.getBoundingClientRect();
        const start = this.atomCenter(canvas, a.id);
        const temp = document.createElementNS("http://www.w3.org/2000/svg", "line");
        temp.setAttribute("class", "bond-temp");
        temp.setAttribute("x1", start.x); temp.setAttribute("y1", start.y);
        temp.setAttribute("x2", start.x); temp.setAttribute("y2", start.y);
        svg.appendChild(temp);
        const move = (ev) => { temp.setAttribute("x2", ev.clientX - rect.left); temp.setAttribute("y2", ev.clientY - rect.top); };
        const up = (ev) => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
          temp.remove();
          const target = document.elementFromPoint(ev.clientX, ev.clientY);
          const tbox = target && target.closest(".atom-box");
          if (tbox && tbox.dataset.aid && tbox.dataset.aid !== a.id) {
            cl.bonds = cl.bonds || [];
            cl.bonds.push({ a: a.id, b: tbox.dataset.aid, pred: "has_bond_to" });
            cl._needsLayout = true;  // reflow so the new bond isn't hidden behind an atom
            this.renderCanvas(); this.changed();
          }
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    },

    atomCenter(canvas, aid) {
      const box = canvas.querySelector(`.atom-box[data-aid="${aid}"]`);
      if (!box) return { x: 0, y: 0 };
      return { x: box.offsetLeft + box.offsetWidth / 2, y: box.offsetTop + box.offsetHeight / 2 };
    },

    drawBonds(ci) {
      const canvas = this.canvasEl.querySelector(`.atom-canvas[data-ci="${ci}"]`);
      if (!canvas) return;
      const svg = canvas.querySelector(".bond-layer");
      const cl = this.model.clauses[ci];
      svg.setAttribute("width", canvas.clientWidth);
      svg.setAttribute("height", canvas.clientHeight);
      // Clear existing (keep any temp line).
      svg.querySelectorAll(".bond-group").forEach((n) => n.remove());
      (cl.bonds || []).forEach((b, idx) => {
        const p1 = this.atomCenter(canvas, b.a), p2 = this.atomCenter(canvas, b.b);
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "bond-group");
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", p1.x); line.setAttribute("y1", p1.y);
        line.setAttribute("x2", p2.x); line.setAttribute("y2", p2.y);
        line.setAttribute("class", "bond-line");
        g.appendChild(line);
        // Type label / control at the midpoint.
        const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
        const fo = document.createElementNS("http://www.w3.org/2000/svg", "foreignObject");
        fo.setAttribute("x", mx - 40); fo.setAttribute("y", my - 14);
        fo.setAttribute("width", 80); fo.setAttribute("height", 28);
        const wrap = document.createElement("div");
        wrap.className = "bond-ctrl";
        wrap.innerHTML = this.bondSelect(b.pred) + `<button class="bond-x" title="remove bond">×</button>`;
        wrap.querySelector("select").addEventListener("change", (e) => { b.pred = e.target.value; this.changed(); });
        wrap.querySelector(".bond-x").addEventListener("click", () => { cl.bonds.splice(idx, 1); this.renderCanvas(); this.changed(); });
        fo.appendChild(wrap);
        g.appendChild(fo);
        svg.appendChild(g);
      });
    },

    bondSelect(pred) {
      const opts = [
        ["has_bond_to", "any"], ["bSINGLE", "single"], ["bDOUBLE", "double"],
        ["bTRIPLE", "triple"], ["bAROMATIC", "aromatic"],
        ["bSTEREOE", "stereo E"], ["bSTEREOZ", "stereo Z"],
        ["bSTEREOCIS", "stereo cis"], ["bSTEREOTRANS", "stereo trans"],
      ];
      if (!opts.some(([p]) => p === pred)) opts.push([pred, labelFor(pred)]);
      return `<select>` + opts.map(([p, l]) => `<option value="${escapeHtml(p)}"${p === pred ? " selected" : ""}>${escapeHtml(l)}</option>`).join("") + `</select>`;
    },
  };

  function escapeHtml(s) {
    return (s == null ? "" : String(s)).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  // Public API.
  global.Blocks = {
    init: (o) => Editor.init(o),
    setCatalog: (c, a, t) => Editor.setCatalog(c, a, t),
    load: (t) => Editor.load(t),
    compile: () => Editor.compile(),
    isDirty: () => Editor.isDirty(),
    loadedText: () => Editor.loadedText,
    _pure: { compileModel, parseRule, describe, splitClauses, groupOf, forceLayout },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = global.Blocks._pure;

})(typeof window !== "undefined" ? window : globalThis);
