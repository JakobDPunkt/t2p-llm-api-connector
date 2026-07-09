"use strict";

/* =========================================================================
 * PnmlParser + PnmlRenderer: adapted from woped-web, branch t2p-v2-test-env,
 * src/app/utilities/modelDisplayer.ts (MIT). TypeScript types stripped,
 * pan/zoom via svg-pan-zoom as in woped-web, and initialMarking
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
 * no coordinates — the direct endpoint is geometry-free by contract.
 * Layered left-to-right layout via dagre (vendored @dagrejs/dagre).
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

    const g = new dagre.graphlib.Graph();
    // Node heights include the label row drawn below transitions, so rows
    // laid out by dagre cannot collide with the text of the row above.
    g.setGraph({ rankdir: "LR", nodesep: 40, ranksep: 70 });
    g.setDefaultEdgeLabel(() => ({}));
    for (const p of net.places) g.setNode(p.id, { width: 50, height: 50 });
    for (const t of net.transitions) g.setNode(t.id, { width: 40, height: 55 });
    for (const a of net.arcs) {
      if (nodes.has(a.source) && nodes.has(a.target)) g.setEdge(a.source, a.target);
    }
    dagre.layout(g);

    for (const [id, node] of nodes) {
      const { x, y } = g.node(id);
      node.x = x;
      node.y = y;
    }
    // Dagre routes edges around nodes (relevant for loops/back edges); keep
    // its bend points so the renderer draws arcs the way they were laid out.
    for (const a of net.arcs) {
      const edge = g.edge(a.source, a.target);
      const bends = edge ? edge.points.slice(1, -1) : [];
      if (bends.length) a.waypoints = bends;
    }
  },
};

/* ========================================================================= */

const PnmlRenderer = {
  SVG_NS: "http://www.w3.org/2000/svg",
  PLACE_R: 25,
  NODE_H: 34,
  TRANSITION_W: 40,

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

    const svg = this.node("svg", {
      xmlns: this.SVG_NS,
      viewBox: `${vbX} ${vbY} ${vbW} ${vbH}`,
      width: "100%",
      height: "100%",
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

    // fit + center: the whole net is visible at first render; wheel,
    // double-click, drag and the control icons take it from there.
    svgPanZoom(svg, {
      zoomEnabled: true,
      controlIconsEnabled: true,
      dblClickZoomEnabled: true,
      mouseWheelZoomEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.1,
      maxZoom: 50,
    });
  },
};

/* =========================================================================
 * Backend contracts.
 * direct:   POST {origin}/generate_pnml_direct  {user_text, provider, model}
 *           -> raw PNML (application/xml) + X-Validation-Issues header
 * pipeline: POST {live}/v2/generate/pnml {text, provider, model}
 *           -> {"result": "<pnml…>"}   (same call woped-web makes)
 * ========================================================================= */

const LIVE_BASE = "https://woped.dhbw-karlsruhe.de/t2p-2.0";
const TIMEOUT_MS = 180000;

/** "1 issue" / "3 issues" without the (s) shorthand. */
function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Distil the direct backend's attempt history into the numbers the result
 *  head and report banner need: what the model got wrong on its first,
 *  unaided attempt, how many correction passes ran, and what was delivered.
 *  Returns null for the pipeline (no attempt history). */
function generationReport(history) {
  if (!history || !Array.isArray(history.attempts) || !history.attempts.length) {
    return null;
  }
  const attempts = history.attempts;
  return {
    firstIssues: attempts[0].issues || [],
    deliveredIssues: attempts[history.deliveredIndex].issues || [],
    corrections: attempts.length - 1,
  };
}

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
      // debug=1 asks the direct backend for the full attempt history (JSON)
      // instead of bare PNML, so the page can show the model's first shot.
      const url = mode === "direct" ? "generate_pnml_direct?debug=1" : LIVE_BASE + "/v2/generate/pnml";
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
        // Debug contract: {pnml, delivered_index, attempts:[{issues, counts}]}.
        const payload = await response.json();
        const attempts = Array.isArray(payload.attempts) ? payload.attempts : [];
        const deliveredIndex = payload.delivered_index || 0;
        const issues = (attempts[deliveredIndex] || {}).issues || [];
        return { pnml: payload.pnml, issues, ms, history: { attempts, deliveredIndex } };
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
          message += ". " + err.details.join(". ");
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

  /** Both backends advertise provider/model pairs; endpoint paths differ.
   * With an apiKey (direct mode only) the connector runs live discovery
   * against the provider, so the list reflects what that key can access —
   * that call takes longer than serving the cached list. The live
   * /v2/models does not forward keys, so pipeline mode never sends one. */
  async models(mode, apiKey) {
    const url = mode === "direct" ? "models" : LIVE_BASE + "/v2/models";
    const useKey = mode === "direct" && apiKey;
    const response = await fetch(url, {
      headers: useKey ? { Authorization: "Bearer " + apiKey } : undefined,
      signal: AbortSignal.timeout(useKey ? 20000 : 8000),
    });
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
  // Per-side model list discovered with that side's API key (direct mode);
  // null falls back to the unauthenticated modelCache list.
  sidePairs: { a: null, b: null },
  keyedModelCache: {},
  keyTimers: { a: null, b: null },
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
      this.input("model", side).addEventListener("change", () => this.refreshRunButton());
      this.input("key", side).addEventListener("input", () => {
        this.refreshRunButton();
        this.scheduleModelReload(side);
      });
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
    this.sidePairs[side] = null;
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
    // Re-apply a key-based list after the base list replaced it.
    if (this.input("key", side).value.trim()) this.reloadModels(side);
  },

  /** Debounced: refetch the model list with the side's API key once the
   * user stops typing, so the dropdown shows what that key can access. */
  scheduleModelReload(side) {
    clearTimeout(this.keyTimers[side]);
    this.keyTimers[side] = setTimeout(() => this.reloadModels(side), 600);
  },

  async reloadModels(side) {
    const mode = this.input("mode", side).value;
    if (mode !== "direct") return; // the live /v2/models ignores keys
    const key = this.input("key", side).value.trim();
    const generation = ++this.modeGeneration[side];
    if (!key) {
      this.sidePairs[side] = null;
      this.onProviderChange(side);
      return;
    }
    this.setModelsNote(side, "loading model list for this key…");
    let pairs = this.keyedModelCache[key];
    if (!pairs) {
      try {
        pairs = this.keyedModelCache[key] = await Api.models(mode, key);
      } catch {
        pairs = null; // discovery call failed: keep the current list
      }
    }
    if (generation !== this.modeGeneration[side]) return;
    if (pairs) this.sidePairs[side] = pairs;
    this.onProviderChange(side);
  },

  /** Strict select fed by the models endpoint, mirroring woped-web's
   * mat-select: only advertised provider/model pairs are offered. */
  onProviderChange(side, pairs) {
    const mode = this.input("mode", side).value;
    const provider = this.input("provider", side).value;
    const models = (pairs || this.sidePairs[side] || this.modelCache[mode] || [])
      .filter((m) => m.provider === provider && m.model)
      .map((m) => m.model);
    const modelSelect = this.input("model", side);
    const previous = modelSelect.value;
    modelSelect.innerHTML = "";
    for (const m of models) {
      const option = document.createElement("option");
      option.value = option.textContent = m;
      modelSelect.appendChild(option);
    }
    if (models.includes(previous)) modelSelect.value = previous;
    this.updateModelsNote(side);
    this.refreshRunButton();
  },

  setModelsNote(side, text) {
    this.el(`models-note-${side}`).textContent = text;
  },

  /** One line under the model select telling the user where the list comes
   * from — and how to get the full one. A fallback-only list has exactly
   * one entry per provider. */
  updateModelsNote(side) {
    const mode = this.input("mode", side).value;
    const hasKey = Boolean(this.input("key", side).value.trim());
    const count = this.input("model", side).options.length;
    let text;
    if (mode === "pipeline") {
      text = count > 1
        ? `${count} models advertised by the live backend.`
        : "Only the backend's default model is advertised.";
    } else if (count > 1) {
      text = `${count} models available for this provider.`;
    } else if (hasKey) {
      text = "Key not accepted for model discovery — default model only.";
    } else {
      text = "Enter your API key to load the provider's full model list.";
    }
    this.setModelsNote(side, text);
  },

  settings(side) {
    return {
      mode: this.input("mode", side).value,
      text: this.el("text").value.trim(),
      provider: this.input("provider", side).value,
      model: this.input("model", side).value.trim(),
      apiKey: this.input("key", side).value.trim(),
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

  /** Sides run independently: a panel without its own API key (or with an
   * unreachable backend) is simply left out of the run. */
  refreshRunButton() {
    const readySides = this.sides.filter((side) => this.ready(side));
    this.el("run").disabled = this.running || readySides.length === 0;
    this.el("run").textContent =
      readySides.length === 1
        ? readySides[0] === "a" ? "Run left" : "Run right"
        : "Run both";
    this.el("run-hint").textContent = this.running
      ? "Runs in progress…"
      : readySides.length === 2
        ? ""
        : readySides.length === 1
          ? "The other panel is skipped (no API key or backend not reachable)."
          : this.sides.some((side) => this.backendUp[side] === false)
            ? "A selected backend is not reachable."
            : "Enter a description, model and API key to start.";
  },

  runBoth() {
    if (this.running || this.el("run").disabled) return;
    const sides = this.sides.filter((side) => this.ready(side));
    if (!sides.length) return;
    this.running = true;
    this.refreshRunButton();
    Promise.allSettled(sides.map((side) => this.runSide(side))).then(() => {
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
      const { pnml, issues, ms, history } = await Api.generate(settings.mode, settings);
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
      if (!AutoLayout.fullyPositioned(net)) AutoLayout.apply(net);
      this.results[side] = { pnml, settings };
      this.stopTimer(side);
      const report = generationReport(history);
      this.renderResultHead(side, settings, {
        ms,
        stats: `${net.places.length} places, ${net.transitions.length} transitions, ${net.arcs.length} arcs`,
        report,
      });
      this.setReportBanner(side, report, issues);
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

  renderResultHead(side, settings, { ms, stats, report } = {}) {
    const head = this.resultCard(side).querySelector(".result-head");
    head.innerHTML = "";

    // Card identity is the backend under test. Provider, model and layout are
    // fixed in the settings above, so they are not repeated as badges; the
    // provider/model pair stays available on hover.
    const title = document.createElement("span");
    title.className = "result-title";
    title.textContent = Api.modesInfo[settings.mode].label;
    title.title = `${settings.provider} / ${settings.model}`;
    head.appendChild(title);

    const add = (cls, text) => {
      const span = document.createElement("span");
      span.className = "badge " + cls;
      span.textContent = text;
      head.appendChild(span);
    };
    if (typeof ms === "number") add("time", (ms / 1000).toFixed(1) + " s");
    if (stats) add("stat", stats);

    // Generation quality (direct backend only). One badge, consistently
    // phrased; the detail of what went wrong lives in the report banner.
    if (report) {
      if (report.firstIssues.length === 0) {
        add("ok", "valid");
      } else if (report.deliveredIssues.length === 0) {
        add("ok", "corrected");
      } else {
        add("warn", `${plural(report.deliveredIssues.length, "issue")} unresolved`);
      }
    }
  },

  /** Show what the direct backend's generation actually did: the delivered
   *  net's remaining problems, or — when the correction loop cleaned them up —
   *  what the model's first attempt got wrong. Both are things the structural
   *  validator can see; a clean first attempt shows no banner. */
  setReportBanner(side, report, deliveredIssues) {
    // Pipeline (no attempt history): only surface remaining issues, if any.
    if (!report) {
      this.setBanner(side, deliveredIssues.length ? "warn" : null, deliveredIssues);
      return;
    }
    if (report.deliveredIssues.length) {
      this.setBanner(
        side, "warn", report.deliveredIssues,
        `${plural(report.deliveredIssues.length, "issue")} unresolved`
      );
    } else if (report.firstIssues.length) {
      this.setBanner(
        side, "info", report.firstIssues,
        `${plural(report.firstIssues.length, "issue")} corrected`
      );
    } else {
      this.setBanner(side, null);
    }
  },

  setBanner(side, kind, items = [], summaryText) {
    const slot = this.resultCard(side).querySelector(".banner-slot");
    slot.innerHTML = "";
    if (!kind) return;
    const banner = document.createElement("div");
    banner.className = "banner " + kind;
    if (kind === "warn" || kind === "info") {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = summaryText || plural(items.length, "issue");
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
      banner.textContent = items.join(". ");
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
