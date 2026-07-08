"use strict";

/* =========================================================================
 * PnmlParser + PnmlRenderer: adapted from woped-web, branch t2p-v2-test-env,
 * src/app/utilities/modelDisplayer.ts (MIT). TypeScript types stripped,
 * svg-pan-zoom replaced by a fixed-height scroll fit, and initialMarking
 * parsing added so the start token is visible. The renderer itself invents
 * no positions: nodes without coordinates are laid out by AutoLayout first.
 * ========================================================================= */

const PnmlParser = {
  parse(pnmlXml) {
    const doc = new DOMParser().parseFromString(pnmlXml, "text/xml");
    if (doc.getElementsByTagName("parsererror").length) {
      throw new Error("The backend returned a document that is not valid XML.");
    }
    const net = { places: [], transitions: [], arcs: [] };

    for (const p of Array.from(doc.getElementsByTagName("place"))) {
      const id = p.getAttribute("id");
      if (!id) continue;
      const text = p.getElementsByTagName("text")[0];
      net.places.push({
        id,
        label: (text && text.textContent) || id,
        marked: this.initialMarking(p) >= 1,
        ...this.ownPosition(p),
      });
    }
    for (const t of Array.from(doc.getElementsByTagName("transition"))) {
      const id = t.getAttribute("id");
      if (!id) continue;
      const text = t.getElementsByTagName("text")[0];
      net.transitions.push({
        id,
        label: (text && text.textContent) || "",
        ...this.ownPosition(t),
      });
    }
    const seen = new Set();
    for (const a of Array.from(doc.getElementsByTagName("arc"))) {
      const source = a.getAttribute("source") || "";
      const target = a.getAttribute("target") || "";
      const key = source + "->" + target;
      if (!source || !target || seen.has(key)) continue;
      seen.add(key);
      const waypoints = this.arcWaypoints(a);
      net.arcs.push({ source, target, ...(waypoints.length ? { waypoints } : {}) });
    }
    return net;
  },

  /** A node's OWN <graphics><position>, not the nested name/label graphics. */
  ownPosition(el) {
    const graphics = Array.from(el.children).find((c) => c.localName === "graphics");
    const position = graphics
      ? Array.from(graphics.children).find((c) => c.localName === "position")
      : undefined;
    if (!position) return {};
    const x = parseFloat(position.getAttribute("x") || "");
    const y = parseFloat(position.getAttribute("y") || "");
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : {};
  },

  arcWaypoints(el) {
    const graphics = Array.from(el.children).find((c) => c.localName === "graphics");
    if (!graphics) return [];
    const points = [];
    for (const child of Array.from(graphics.children)) {
      if (child.localName !== "position") continue;
      const x = parseFloat(child.getAttribute("x") || "");
      const y = parseFloat(child.getAttribute("y") || "");
      if (Number.isFinite(x) && Number.isFinite(y)) points.push({ x, y });
    }
    return points;
  },

  initialMarking(place) {
    for (const child of Array.from(place.children)) {
      if (child.localName !== "initialMarking") continue;
      const text = child.getElementsByTagName("text")[0];
      const value = parseInt((text && text.textContent) || "", 10);
      return Number.isFinite(value) ? value : 0;
    }
    return 0;
  },
};

/* =========================================================================
 * AutoLayout: demo-only stand-in for the pipeline's server-side layouting
 * (t2p-2.0 assign_pnml_coordinates). Applied ONLY when the document carries
 * no coordinates — the direct endpoint is geometry-free by contract. Simple
 * layered layout: BFS from the start place, columns left to right.
 * ========================================================================= */

const AutoLayout = {
  /** True only when EVERY node is positioned; a partially positioned net
   * (contract violation) is re-laid out completely rather than rendering
   * a fragment. */
  fullyPositioned(net) {
    return [...net.places, ...net.transitions].every(
      (n) => typeof n.x === "number" && typeof n.y === "number"
    );
  },

  apply(net) {
    const nodes = new Map();
    for (const n of [...net.places, ...net.transitions]) nodes.set(n.id, n);

    const outgoing = new Map();
    const incoming = new Map();
    for (const a of net.arcs) {
      if (!nodes.has(a.source) || !nodes.has(a.target)) continue;
      (outgoing.get(a.source) || outgoing.set(a.source, []).get(a.source)).push(a.target);
      (incoming.get(a.target) || incoming.set(a.target, []).get(a.target)).push(a.source);
    }

    const starts = net.places
      .filter((p) => p.marked || !(incoming.get(p.id) || []).length)
      .map((p) => p.id);
    const queue = (starts.length ? starts : [...nodes.keys()].slice(0, 1)).map(
      (id) => [id, 0]
    );
    const layer = new Map();
    while (queue.length) {
      const [id, depth] = queue.shift();
      if (layer.has(id)) continue;
      layer.set(id, depth);
      for (const next of outgoing.get(id) || []) queue.push([next, depth + 1]);
    }
    let spare = Math.max(-1, ...layer.values()) + 1;
    for (const id of nodes.keys()) if (!layer.has(id)) layer.set(id, spare++);

    const byLayer = new Map();
    for (const [id, l] of layer) (byLayer.get(l) || byLayer.set(l, []).get(l)).push(id);

    const X0 = 80, DX = 130, MID_Y = 210, DY = 100;
    for (const [l, ids] of byLayer) {
      ids.forEach((id, i) => {
        const node = nodes.get(id);
        node.x = X0 + l * DX;
        node.y = MID_Y + (i - (ids.length - 1) / 2) * DY;
      });
    }
  },
};

/* ========================================================================= */

const PnmlRenderer = {
  SVG_NS: "http://www.w3.org/2000/svg",
  PLACE_R: 25,
  NODE_H: 34,
  TRANSITION_W: 40,
  CANVAS_H: 368,

  node(tag, attrs) {
    const el = document.createElementNS(this.SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
  },

  /** Point on the node border towards (px, py), so arcs meet the shape edge. */
  clip(b, px, py) {
    const dx = px - b.cx, dy = py - b.cy;
    if (dx === 0 && dy === 0) return { x: b.cx, y: b.cy };
    if (b.shape === "circle") {
      const d = Math.hypot(dx, dy);
      return { x: b.cx + (dx * b.r) / d, y: b.cy + (dy * b.r) / d };
    }
    const sx = dx !== 0 ? b.w / 2 / Math.abs(dx) : Infinity;
    const sy = dy !== 0 ? b.h / 2 / Math.abs(dy) : Infinity;
    const s = Math.min(sx, sy);
    return { x: b.cx + dx * s, y: b.cy + dy * s };
  },

  cleanLabel(label) {
    return label.replace(/^\s*\[[^\]]*\]\s*/, "").trim() || label;
  },

  render(container, net, markerId) {
    const boxes = new Map();
    for (const p of net.places) {
      if (typeof p.x !== "number" || typeof p.y !== "number") continue;
      boxes.set(p.id, { cx: p.x, cy: p.y, shape: "circle", r: this.PLACE_R, w: 0, h: 0 });
    }
    for (const t of net.transitions) {
      if (typeof t.x !== "number" || typeof t.y !== "number") continue;
      boxes.set(t.id, {
        cx: t.x, cy: t.y, shape: "rect", r: 0, w: this.TRANSITION_W, h: this.NODE_H,
      });
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const grow = (x, y) => {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    };
    for (const b of boxes.values()) {
      const hw = b.shape === "circle" ? b.r : b.w / 2;
      const hh = b.shape === "circle" ? b.r : b.h / 2;
      grow(b.cx - hw, b.cy - hh);
      grow(b.cx + hw, b.cy + hh);
    }
    for (const arc of net.arcs) for (const wp of arc.waypoints || []) grow(wp.x, wp.y);
    for (const t of net.transitions) {
      const b = boxes.get(t.id);
      if (!b) continue;
      const halfW = (this.cleanLabel(t.label).length * 7) / 2;
      grow(b.cx - halfW, b.cy + b.h / 2 + 20);
      grow(b.cx + halfW, b.cy + b.h / 2 + 20);
    }
    if (!Number.isFinite(minX)) { minX = minY = 0; maxX = maxY = 100; }

    const pad = 30;
    const vbX = minX - pad, vbY = minY - pad;
    const vbW = maxX - minX + 2 * pad, vbH = maxY - minY + 2 * pad;

    // Height-fit with a zoom cap: a long net scrolls horizontally at full
    // shape size instead of shrinking to a sliver (same intent as woped-web's
    // height-fit re-zoom, without svg-pan-zoom), while a tiny net is not
    // blown up beyond 1.5x its natural size. Height leaves room for a
    // horizontal scrollbar inside the fixed-height canvas.
    const MAX_SCALE = 1.5;
    const scale = Math.min(this.CANVAS_H / vbH, MAX_SCALE);
    const height = Math.max(1, Math.round(vbH * scale));
    const width = Math.max(1, Math.round(vbW * scale));
    const svg = this.node("svg", {
      xmlns: this.SVG_NS,
      viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
      width, height,
    });

    const defs = this.node("defs", {});
    const marker = this.node("marker", {
      id: markerId, viewBox: "0 0 10 10", refX: 9, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: "auto",
    });
    marker.appendChild(this.node("path", { d: "M0,0 L10,5 L0,10 z", fill: "#111" }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    for (const arc of net.arcs) {
      const s = boxes.get(arc.source);
      const t = boxes.get(arc.target);
      if (!s || !t) continue;
      const mids = arc.waypoints || [];
      const afterSource = mids[0] || { x: t.cx, y: t.cy };
      const beforeTarget = mids[mids.length - 1] || { x: s.cx, y: s.cy };
      const start = this.clip(s, afterSource.x, afterSource.y);
      const end = this.clip(t, beforeTarget.x, beforeTarget.y);
      const pts = [start, ...mids, end];
      svg.appendChild(this.node("polyline", {
        points: pts.map((p) => `${p.x},${p.y}`).join(" "),
        fill: "none", stroke: "#111", "stroke-width": 1.5,
        "marker-end": `url(#${markerId})`,
      }));
    }

    for (const p of net.places) {
      const b = boxes.get(p.id);
      if (!b) continue;
      const circle = this.node("circle", {
        cx: b.cx, cy: b.cy, r: b.r, fill: "#fff", stroke: "#111", "stroke-width": 1.5,
      });
      const title = this.node("title", {});
      title.textContent = p.label;
      circle.appendChild(title);
      svg.appendChild(circle);
      if (p.marked) {
        svg.appendChild(this.node("circle", { cx: b.cx, cy: b.cy, r: 5, fill: "#111" }));
      }
    }

    for (const t of net.transitions) {
      const b = boxes.get(t.id);
      if (!b) continue;
      const rect = this.node("rect", {
        x: b.cx - b.w / 2, y: b.cy - b.h / 2, width: b.w, height: b.h,
        fill: "#fff", stroke: "#111", "stroke-width": 1.5,
      });
      const title = this.node("title", {});
      title.textContent = t.label;
      rect.appendChild(title);
      svg.appendChild(rect);
      const label = this.cleanLabel(t.label);
      if (!label) continue;
      const text = this.node("text", {
        x: b.cx, y: b.cy + b.h / 2 + 6,
        "text-anchor": "middle", "dominant-baseline": "hanging",
        "font-size": 12, fill: "#111",
      });
      // Truncate long labels so neighbouring columns don't overlap; the
      // full name stays available as the shape's tooltip.
      text.textContent = label.length > 22 ? label.slice(0, 21) + "…" : label;
      svg.appendChild(text);
    }

    container.innerHTML = "";
    container.appendChild(svg);
  },
};

/* =========================================================================
 * Backend contracts.
 * direct:   POST {origin}/generate_pnml  {user_text, provider, model}
 *           -> raw PNML (application/xml) + X-Validation-Issues header
 * pipeline: POST {live}/v2/generate/pnml {text, provider, model}
 *           -> {"result": "<pnml…>"}   (same call woped-web makes)
 * ========================================================================= */

const LIVE_BASE = "https://woped.dhbw-karlsruhe.de/t2p-2.0";
const TIMEOUT_MS = 180000;

const Api = {
  modesInfo: {
    direct: { label: "Direct PNML" },
    pipeline: { label: "Pipeline" },
  },

  async generate(mode, { text, provider, model, apiKey }) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    const started = performance.now();
    try {
      // Relative same-origin path so the page also works behind a
      // path-prefix reverse proxy (resolved against /demo).
      const url = mode === "direct" ? "generate_pnml" : LIVE_BASE + "/v2/generate/pnml";
      const body = mode === "direct"
        ? { user_text: text, provider, model }
        : { text, provider, model };
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + apiKey,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const ms = performance.now() - started;
      if (!response.ok) {
        throw Object.assign(new Error(await this.errorMessage(response, mode)), { ms });
      }
      if (mode === "direct") {
        const pnml = await response.text();
        const issuesHeader = response.headers.get("X-Validation-Issues") || "";
        const issues = issuesHeader ? issuesHeader.split("; ").filter(Boolean) : [];
        return { pnml, issues, ms };
      }
      const payload = await response.json();
      if (!payload || typeof payload.result !== "string") {
        throw Object.assign(new Error("The pipeline response carried no result."), { ms });
      }
      return { pnml: payload.result, issues: [], ms };
    } catch (err) {
      if (err.name === "AbortError") {
        throw Object.assign(new Error("The request timed out."), {
          ms: performance.now() - started,
        });
      }
      if (err instanceof TypeError) {
        throw Object.assign(
          new Error(mode === "direct"
            ? "Could not reach this connector. Is it still running?"
            : "Could not reach the live WoPeD server."),
          { ms: performance.now() - started }
        );
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  },

  /** v2 error bodies are {"error": {code, message, details?}} on both APIs. */
  async errorMessage(response, mode) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      const err = body && body.error;
      if (err && err.message) {
        message = err.message;
        if (Array.isArray(err.details) && err.details.length) {
          message += ": " + err.details.join("; ");
        }
      }
    } catch { /* non-JSON error page (e.g. proxy 502): keep the status text */ }
    if (response.status === 502 && mode === "pipeline") {
      message += " (The live server may time out on long generations.)";
    }
    return message;
  },

  /** Backend reachability probe. The pipeline backend has a dedicated
   * health endpoint; the direct backend is the server that serves this
   * page, so its /models call (needed for the dropdowns anyway) doubles
   * as the probe — /health/ready would test provider connectivity
   * instead and takes seconds. */
  async reachable(mode) {
    if (mode === "direct") return true; // refined by the models fetch below
    const response = await fetch(LIVE_BASE + "/v2/health", {
      signal: AbortSignal.timeout(8000),
    });
    return response.ok;
  },

  /** Both backends advertise provider/model pairs; endpoint paths differ. */
  async models(mode) {
    const url = mode === "direct" ? "models" : LIVE_BASE + "/v2/models";
    const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!response.ok) throw new Error("models endpoint " + response.status);
    const payload = await response.json();
    return Array.isArray(payload.models) ? payload.models : [];
  },
};

/* ========================================================================= */

const App = {
  sides: ["a", "b"],
  results: { a: null, b: null },
  modelCache: {},
  timers: { a: null, b: null },
  running: false,
  modeGeneration: { a: 0, b: 0 },
  backendUp: { a: null, b: null },

  el(id) { return document.getElementById(id); },
  input(name, side) { return this.el(`${name}-${side}`); },
  resultCard(side) { return this.el(`result-${side}`); },

  init() {
    for (const side of this.sides) {
      this.input("mode", side).addEventListener("change", () => this.onModeChange(side));
      this.input("provider", side).addEventListener("change", () => this.onProviderChange(side));
      for (const name of ["model", "key"]) {
        this.input(name, side).addEventListener("input", () => this.refreshRunButton());
      }
      this.resultCard(side).querySelector('[data-act="xml"]')
        .addEventListener("click", () => this.showXml(side));
      this.resultCard(side).querySelector('[data-act="download"]')
        .addEventListener("click", () => this.download(side));
      this.onModeChange(side);
    }
    this.el("text").addEventListener("input", () => this.refreshRunButton());
    this.el("run").addEventListener("click", () => this.runBoth());
    this.el("xml-close").addEventListener("click", () => this.el("xml-dialog").close());
    // Close the dialog on a backdrop click (the inner content stops the
    // event from reaching the dialog element itself).
    this.el("xml-dialog").addEventListener("click", (event) => {
      if (event.target === this.el("xml-dialog")) this.el("xml-dialog").close();
    });
    this.refreshRunButton();
  },

  async onModeChange(side) {
    const mode = this.input("mode", side).value;
    // Guard against a slow models fetch finishing after the user switched
    // the mode again: only the latest invocation may touch the controls.
    const generation = ++this.modeGeneration[side];
    this.backendUp[side] = null;
    this.setStatus(side, "checking", "checking backend…");
    this.refreshRunButton();
    const providerSelect = this.input("provider", side);
    const previous = providerSelect.value;
    let pairs = this.modelCache[mode];
    let reachable = Boolean(pairs);
    if (!pairs) {
      try {
        const probe = Api.reachable(mode);
        pairs = this.modelCache[mode] = await Api.models(mode);
        reachable = await probe;
      } catch {
        // Neither result is cached, so the next mode change retries.
        reachable = false;
        pairs = [
          { provider: "openai", model: "" },
          { provider: "gemini", model: "" },
        ];
      }
    }
    if (!reachable) delete this.modelCache[mode];
    if (generation !== this.modeGeneration[side]) return;
    this.backendUp[side] = reachable;
    this.setStatus(
      side,
      reachable ? "ok" : "bad",
      reachable ? "backend reachable" : "backend not reachable"
    );
    const providers = [...new Set(pairs.map((m) => m.provider))];
    providerSelect.innerHTML = "";
    for (const p of providers) {
      const option = document.createElement("option");
      option.value = option.textContent = p;
      providerSelect.appendChild(option);
    }
    if (providers.includes(previous)) providerSelect.value = previous;
    this.onProviderChange(side, pairs);
  },

  onProviderChange(side, pairs) {
    const mode = this.input("mode", side).value;
    const provider = this.input("provider", side).value;
    const datalist = this.el(`models-${side}`);
    datalist.innerHTML = "";
    const models = (pairs || this.modelCache[mode] || [])
      .filter((m) => m.provider === provider && m.model)
      .map((m) => m.model);
    for (const m of models) {
      const option = document.createElement("option");
      option.value = m;
      datalist.appendChild(option);
    }
    const modelInput = this.input("model", side);
    // Fill an empty field, and replace a value that clearly belongs to
    // another provider's advertised list, so a provider switch cannot
    // submit a mismatched pair. Hand-typed custom names are kept.
    const allKnown = Object.values(this.modelCache).flat();
    const belongsElsewhere = allKnown.some(
      (m) => m.model === modelInput.value && m.provider !== provider
    );
    if ((!modelInput.value || belongsElsewhere) && models.length) {
      modelInput.value = models[0];
    }
    this.refreshRunButton();
  },

  settings(side) {
    const key = this.input("key", side).value.trim()
      || this.input("key", "a").value.trim();
    return {
      mode: this.input("mode", side).value,
      text: this.el("text").value.trim(),
      provider: this.input("provider", side).value,
      model: this.input("model", side).value.trim(),
      apiKey: key,
    };
  },

  ready(side) {
    const s = this.settings(side);
    return Boolean(
      this.backendUp[side] && s.text && s.provider && s.model && s.apiKey
    );
  },

  setStatus(side, kind, text) {
    const note = this.el(`note-${side}`);
    note.className = "mode-note status-" + kind;
    note.textContent = text;
  },

  refreshRunButton() {
    const ok = this.sides.every((side) => this.ready(side));
    this.el("run").disabled = this.running || !ok;
    this.el("run-hint").textContent = this.running
      ? "Runs in progress…"
      : ok
        ? ""
        : this.sides.some((side) => this.backendUp[side] === false)
          ? "A selected backend is not reachable."
          : "Enter a description, model and API key to start.";
  },

  runBoth() {
    if (this.running || this.el("run").disabled) return;
    this.running = true;
    this.refreshRunButton();
    Promise.allSettled(this.sides.map((side) => this.runSide(side))).then(() => {
      this.running = false;
      this.refreshRunButton();
    });
  },

  async runSide(side) {
    const settings = this.settings(side);
    this.results[side] = null;
    this.setBanner(side, null);
    this.setActionsEnabled(side, false);
    this.showRunning(side, settings);
    try {
      const { pnml, issues, ms } = await Api.generate(settings.mode, settings);
      let net;
      try {
        net = PnmlParser.parse(pnml);
        if (net.places.length + net.transitions.length === 0) {
          throw new Error("The response contained no Petri net elements.");
        }
      } catch (parseErr) {
        parseErr.ms = ms; // keep the measured request time on parse failures
        throw parseErr;
      }
      const autoLaidOut = !AutoLayout.fullyPositioned(net);
      if (autoLaidOut) AutoLayout.apply(net);
      this.results[side] = { pnml, settings };
      this.stopTimer(side);
      this.renderResultHead(side, settings, {
        ms,
        stats: `${net.places.length} places, ${net.transitions.length} transitions, ${net.arcs.length} arcs`,
        layout: autoLaidOut ? "client auto-layout" : "backend layout",
      });
      if (issues.length) {
        this.setBanner(side, "warn", issues);
      }
      PnmlRenderer.render(
        this.resultCard(side).querySelector(".canvas"), net, `arrow-${side}`
      );
      this.setActionsEnabled(side, true);
    } catch (err) {
      this.stopTimer(side);
      this.renderResultHead(side, settings, { ms: err.ms });
      this.setBanner(side, "err", [err.message || String(err)]);
      this.resultCard(side).querySelector(".canvas").innerHTML =
        '<div class="placeholder">No result.</div>';
    }
  },

  showRunning(side, settings) {
    this.renderResultHead(side, settings, {});
    const canvas = this.resultCard(side).querySelector(".canvas");
    canvas.innerHTML =
      '<div class="placeholder"><div><div class="spinner"></div>Generating… <span data-elapsed>0.0</span> s</div></div>';
    const startedAt = performance.now();
    this.stopTimer(side);
    this.timers[side] = setInterval(() => {
      const el = canvas.querySelector("[data-elapsed]");
      if (el) el.textContent = ((performance.now() - startedAt) / 1000).toFixed(1);
    }, 100);
  },

  stopTimer(side) {
    if (this.timers[side]) clearInterval(this.timers[side]);
    this.timers[side] = null;
  },

  renderResultHead(side, settings, { ms, stats, layout } = {}) {
    const head = this.resultCard(side).querySelector(".result-head");
    head.innerHTML = "";
    const add = (cls, text) => {
      const span = document.createElement("span");
      span.className = "badge " + cls;
      span.textContent = text;
      head.appendChild(span);
    };
    const modeText = `${Api.modesInfo[settings.mode].label} (${settings.provider}/${settings.model})`;
    add("mode", modeText);
    head.firstChild.title = modeText; // full value on hover if ellipsized
    if (typeof ms === "number") add("time", (ms / 1000).toFixed(1) + " s");
    if (stats) add("stat", stats);
    if (layout) add("layout", layout);
  },

  setBanner(side, kind, items = []) {
    const slot = this.resultCard(side).querySelector(".banner-slot");
    slot.innerHTML = "";
    if (!kind) return;
    const banner = document.createElement("div");
    banner.className = "banner " + kind;
    if (kind === "warn") {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `Validation issues (${items.length})`;
      details.appendChild(summary);
      const list = document.createElement("ul");
      for (const item of items) {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      }
      details.appendChild(list);
      banner.appendChild(details);
    } else {
      banner.textContent = items.join(" — ");
    }
    slot.appendChild(banner);
  },

  setActionsEnabled(side, enabled) {
    for (const btn of this.resultCard(side).querySelectorAll(".result-actions button")) {
      btn.disabled = !enabled;
    }
  },

  showXml(side) {
    const result = this.results[side];
    if (!result) return;
    this.el("xml-title").textContent =
      `PNML (${Api.modesInfo[result.settings.mode].label})`;
    this.el("xml-body").textContent = result.pnml;
    this.el("xml-dialog").showModal();
  },

  download(side) {
    const result = this.results[side];
    if (!result) return;
    const blob = new Blob([result.pnml], { type: "application/xml" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    const safeModel = result.settings.model.replace(/[^\w.-]+/g, "_");
    link.download = `${result.settings.mode}-${safeModel}.pnml`;
    link.click();
    URL.revokeObjectURL(link.href);
  },
};

App.init();
