/**
 * compare.js — NestAI Comparison Mode
 *
 * Adds a "Compare" tab to the existing frontend that runs the same query
 * against all three search methods side-by-side:
 *   • Baseline 1 — Filter-Based (regex, no LLM)
 *   • Baseline 2 — LLM Chatbot  (single GPT-4o-mini call)
 *   • NestAI Agent              (full ReAct pipeline)
 *
 * Drop compare.js next to app.js and add one line to index.html:
 *   <script src="compare.js"></script>
 * (after app.js so window globals are available)
 */

(function () {
  "use strict";

  // ── Config ────────────────────────────────────────────────────────────────

  const ENDPOINTS = {
    filter: "/api/search/baseline-filter",
    llm:    "/api/search/baseline-llm",
    agent:  "/api/search",
  };

  const COLUMN_META = [
    {
      key:   "filter",
      label: "Baseline 1 · Filter",
      badge: "No LLM",
      color: "#e07b39",
      desc:  "Regex/keyword parsing → hard filters → price sort. Fast & deterministic.",
    },
    {
      key:   "llm",
      label: "Baseline 2 · LLM Chat",
      badge: "Single GPT call",
      color: "#7c6bdf",
      desc:  "Query + sampled listings sent as plain text to GPT-4o-mini in one turn.",
    },
    {
      key:   "agent",
      label: "NestAI Agent",
      badge: "ReAct pipeline",
      color: "#2a9d8f",
      desc:  "LangGraph ReAct loop: parse → filter → score → adapt → explain.",
    },
  ];

  const DATASET_PATH = "matched_subset_dataset.csv";

  // ── State ─────────────────────────────────────────────────────────────────

  const state = {
    results: { filter: null, llm: null, agent: null },
    loading: { filter: false, llm: false, agent: false },
    errors:  { filter: null,  llm: null,  agent: null  },
    timings: { filter: null,  llm: null,  agent: null  },
  };

  // ── DOM helpers ───────────────────────────────────────────────────────────

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "className") node.className = v;
      else if (k === "style")   Object.assign(node.style, v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
      else node.setAttribute(k, v);
    }
    for (const child of children) {
      if (child == null) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  function getApiKey() {
    // Re-use whatever key the main app already has in its settings input
    const keyInput =
      document.getElementById("api-key-input") ||
      document.querySelector('input[type="password"]') ||
      document.querySelector('input[placeholder*="API"]');
    return keyInput ? keyInput.value.trim() : "";
  }

  function getQuery() {
    const qInput =
      document.getElementById("query-input") ||
      document.querySelector('input[type="text"]') ||
      document.querySelector("textarea");
    return qInput ? qInput.value.trim() : "";
  }

  // ── API calls ─────────────────────────────────────────────────────────────

  async function fetchBaseline(method, query, apiKey) {
    const url  = ENDPOINTS[method];
    const body = { query, dataset: DATASET_PATH };
    if (method === "llm" || method === "agent") body.api_key = apiKey;

    const t0       = performance.now();
    const response = await fetch(url, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: response.statusText }));
      throw Object.assign(new Error(err.error || "Request failed"), { elapsed });
    }
    const data = await response.json();
    return { data, elapsed };
  }

  // ── Rendering ─────────────────────────────────────────────────────────────

  function renderCard(listing, meta, rank) {
    const price    = listing.price != null ? `$${listing.price.toFixed(0)}/night` : "Price N/A";
    const rating   = listing.review_rating != null ? `${Number(listing.review_rating).toFixed(2)}★` : "—";
    const beds     = listing.bedrooms != null ? `${listing.bedrooms}BR` : "?BR";
    const baths    = listing.bathrooms != null ? `${Number(listing.bathrooms).toFixed(1)}BA` : "?BA";
    const nbhd     = listing.neighborhood || "—";
    const amenities = (listing.amenities || []).slice(0, 4);

    const card = el("div", { className: "cmp-card" },
      el("div", { className: "cmp-card-rank", style: { background: meta.color } }, `#${rank}`),
      el("div", { className: "cmp-card-body" },
        el("div", { className: "cmp-card-title" }, listing.title || "Untitled"),
        el("div", { className: "cmp-card-meta" },
          el("span", { className: "cmp-tag" }, nbhd),
          el("span", { className: "cmp-tag cmp-tag--price" }, price),
          el("span", { className: "cmp-tag" }, `${beds} · ${baths}`),
          el("span", { className: "cmp-tag" }, rating),
        ),
        amenities.length
          ? el("div", { className: "cmp-amenities" },
              ...amenities.map(a => el("span", { className: "cmp-amenity" }, a))
            )
          : null,
        listing.llm_explanation
          ? el("div", { className: "cmp-explanation" }, `"${listing.llm_explanation}"`)
          : null,
        // NestAI agent explanations come via the parent result's explanations array
      )
    );
    return card;
  }

  function renderColumn(meta, colEl) {
    const s        = state;
    const loading  = s.loading[meta.key];
    const error    = s.errors[meta.key];
    const result   = s.results[meta.key];
    const elapsed  = s.timings[meta.key];

    // Clear old content (keep header)
    const header = colEl.querySelector(".cmp-col-header");
    colEl.innerHTML = "";
    colEl.appendChild(header);

    if (loading) {
      colEl.appendChild(el("div", { className: "cmp-spinner-wrap" },
        el("div", { className: "cmp-spinner" }),
        el("p", { className: "cmp-spinner-label" }, "Searching…"),
      ));
      return;
    }

    if (error) {
      colEl.appendChild(el("div", { className: "cmp-error" }, `⚠ ${error}`));
      return;
    }

    if (!result) return;

    const recs  = result.recommendations || [];
    const expl  = result.explanation || "";
    const agent_explanations = result.explanations || [];

    // Stats bar
    colEl.appendChild(el("div", { className: "cmp-stats" },
      el("span", { className: "cmp-stat" }, `${recs.length} results`),
      elapsed ? el("span", { className: "cmp-stat" }, `${elapsed}s`) : null,
      result.total_matched != null
        ? el("span", { className: "cmp-stat" }, `${result.total_matched} matched`) : null,
      result.listings_shown != null
        ? el("span", { className: "cmp-stat" }, `${result.listings_shown} shown to LLM`) : null,
    ));

    if (expl) {
      colEl.appendChild(el("div", { className: "cmp-method-expl" }, expl));
    }

    if (recs.length === 0) {
      colEl.appendChild(el("div", { className: "cmp-empty" }, "No results returned."));
      return;
    }

    const list = el("div", { className: "cmp-cards" });
    recs.forEach((listing, i) => {
      const card = renderCard(listing, meta, i + 1);
      // Attach agent explanation if available
      if (meta.key === "agent" && agent_explanations[i]) {
        const explBlock = el("div", { className: "cmp-explanation" },
          `"${agent_explanations[i]}"`);
        card.querySelector(".cmp-card-body").appendChild(explBlock);
      }
      list.appendChild(card);
    });
    colEl.appendChild(list);
  }

  function renderAllColumns() {
    COLUMN_META.forEach(meta => {
      const colEl = document.getElementById(`cmp-col-${meta.key}`);
      if (colEl) renderColumn(meta, colEl);
    });
  }

  // ── Search orchestration ──────────────────────────────────────────────────

  async function runComparison(query) {
    const apiKey = getApiKey();

    // Reset state
    for (const m of COLUMN_META) {
      state.results[m.key] = null;
      state.errors[m.key]  = null;
      state.timings[m.key] = null;
      state.loading[m.key] = true;
    }
    renderAllColumns();

    // Fire all three requests concurrently
    await Promise.allSettled(
      COLUMN_META.map(async (meta) => {
        try {
          const { data, elapsed } = await fetchBaseline(meta.key, query, apiKey);
          state.results[meta.key] = data;
          state.timings[meta.key] = elapsed;
        } catch (err) {
          state.errors[meta.key] = err.message;
          if (err.elapsed) state.timings[meta.key] = err.elapsed;
        } finally {
          state.loading[meta.key] = false;
          const colEl = document.getElementById(`cmp-col-${meta.key}`);
          if (colEl) renderColumn(meta, colEl);
        }
      })
    );
  }

  // ── Tab injection ─────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("cmp-styles")) return;
    const style = document.createElement("style");
    style.id = "cmp-styles";
    style.textContent = `
      /* ── Tab button ── */
      #cmp-tab-btn {
        cursor: pointer;
        padding: 8px 18px;
        border-radius: 20px;
        border: 2px solid #2a9d8f;
        background: transparent;
        color: #2a9d8f;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: .3px;
        transition: all .2s;
        margin-left: 10px;
      }
      #cmp-tab-btn:hover, #cmp-tab-btn.active {
        background: #2a9d8f;
        color: #fff;
      }

      /* ── Comparison panel ── */
      #cmp-panel {
        display: none;
        flex-direction: column;
        gap: 16px;
        padding: 20px;
        max-width: 1400px;
        margin: 0 auto;
        width: 100%;
        box-sizing: border-box;
      }
      #cmp-panel.visible { display: flex; }

      #cmp-header {
        text-align: center;
        margin-bottom: 8px;
      }
      #cmp-header h2 {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0 0 4px;
      }
      #cmp-header p {
        color: #666;
        font-size: 14px;
        margin: 0;
      }

      #cmp-search-row {
        display: flex;
        gap: 10px;
        align-items: center;
        background: #fff;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,.08);
      }
      #cmp-query-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 15px;
        color: #222;
        background: transparent;
      }
      #cmp-query-input::placeholder { color: #aaa; }
      #cmp-search-btn {
        padding: 8px 22px;
        background: #2a9d8f;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: background .2s;
        white-space: nowrap;
      }
      #cmp-search-btn:hover { background: #21867a; }
      #cmp-search-btn:disabled { background: #aaa; cursor: default; }

      /* ── Columns ── */
      #cmp-columns {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        align-items: start;
      }
      @media (max-width: 900px) {
        #cmp-columns { grid-template-columns: 1fr; }
      }

      .cmp-col {
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 2px 14px rgba(0,0,0,.07);
        overflow: hidden;
        min-height: 200px;
      }
      .cmp-col-header {
        padding: 14px 16px 10px;
        border-bottom: 3px solid;
      }
      .cmp-col-header h3 {
        margin: 0 0 4px;
        font-size: 15px;
        font-weight: 700;
        color: #1a1a2e;
      }
      .cmp-col-header .cmp-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        color: #fff;
        margin-bottom: 6px;
      }
      .cmp-col-header .cmp-col-desc {
        font-size: 12px;
        color: #777;
        line-height: 1.4;
      }

      /* ── Stats bar ── */
      .cmp-stats {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        padding: 8px 16px;
        background: #f8f9fa;
        border-bottom: 1px solid #eee;
      }
      .cmp-stat {
        font-size: 12px;
        color: #555;
        background: #e9ecef;
        padding: 2px 8px;
        border-radius: 10px;
      }

      /* ── Method explanation ── */
      .cmp-method-expl {
        padding: 8px 16px;
        font-size: 12px;
        color: #666;
        border-bottom: 1px solid #f0f0f0;
        font-style: italic;
        line-height: 1.5;
      }

      /* ── Cards ── */
      .cmp-cards { padding: 12px; display: flex; flex-direction: column; gap: 10px; }

      .cmp-card {
        display: flex;
        border: 1px solid #efefef;
        border-radius: 10px;
        overflow: hidden;
        background: #fafafa;
        transition: box-shadow .15s;
      }
      .cmp-card:hover { box-shadow: 0 3px 12px rgba(0,0,0,.1); }

      .cmp-card-rank {
        width: 32px;
        min-width: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-size: 12px;
        font-weight: 700;
      }

      .cmp-card-body { padding: 10px 12px; flex: 1; min-width: 0; }

      .cmp-card-title {
        font-size: 13px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .cmp-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 5px;
      }

      .cmp-tag {
        font-size: 11px;
        background: #eef0f3;
        color: #444;
        padding: 2px 7px;
        border-radius: 8px;
        white-space: nowrap;
      }
      .cmp-tag--price {
        background: #e8f5e9;
        color: #2e7d32;
        font-weight: 600;
      }

      .cmp-amenities {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        margin-bottom: 4px;
      }
      .cmp-amenity {
        font-size: 10px;
        background: #e3f2fd;
        color: #1565c0;
        padding: 1px 6px;
        border-radius: 6px;
      }

      .cmp-explanation {
        font-size: 11px;
        color: #666;
        font-style: italic;
        margin-top: 5px;
        line-height: 1.4;
      }

      /* ── Spinner ── */
      .cmp-spinner-wrap { display: flex; flex-direction: column; align-items: center; padding: 40px 20px; gap: 12px; }
      .cmp-spinner {
        width: 32px; height: 32px;
        border: 3px solid #e0e0e0;
        border-top-color: #2a9d8f;
        border-radius: 50%;
        animation: cmp-spin .8s linear infinite;
      }
      @keyframes cmp-spin { to { transform: rotate(360deg); } }
      .cmp-spinner-label { font-size: 13px; color: #888; }

      /* ── Error / empty ── */
      .cmp-error { padding: 20px; color: #c62828; font-size: 13px; }
      .cmp-empty { padding: 20px; color: #999; font-size: 13px; text-align: center; }
    `;
    document.head.appendChild(style);
  }

  function buildPanel() {
    const panel = el("div", { id: "cmp-panel" },
      el("div", { id: "cmp-header" },
        el("h2", {}, "🔍 Side-by-Side Comparison"),
        el("p",  {}, "Run the same query through all three search methods at once."),
      ),
      el("div", { id: "cmp-search-row" },
        el("input", {
          id:          "cmp-query-input",
          type:        "text",
          placeholder: "e.g. 2-bedroom with WiFi in Brooklyn under $200/night",
        }),
        el("button", {
          id:    "cmp-search-btn",
          onClick: handleSearch,
        }, "Compare All"),
      ),
      el("div", { id: "cmp-columns" },
        ...COLUMN_META.map(meta =>
          el("div", { className: "cmp-col", id: `cmp-col-${meta.key}` },
            el("div", {
              className: "cmp-col-header",
              style:     { borderBottomColor: meta.color },
            },
              el("div", { className: "cmp-badge", style: { background: meta.color } }, meta.badge),
              el("h3", {}, meta.label),
              el("div", { className: "cmp-col-desc" }, meta.desc),
            )
          )
        )
      )
    );
    return panel;
  }

  async function handleSearch() {
    const query = document.getElementById("cmp-query-input").value.trim();
    if (!query) {
      alert("Please enter a search query.");
      return;
    }
    const btn = document.getElementById("cmp-search-btn");
    btn.disabled = true;
    btn.textContent = "Searching…";
    try {
      await runComparison(query);
    } finally {
      btn.disabled = false;
      btn.textContent = "Compare All";
    }
  }

  function injectTabButton(mainContent) {
    // Try to find the existing search button or nav area to append next to
    const navArea =
      document.querySelector(".nav-tabs") ||
      document.querySelector(".tabs") ||
      document.querySelector("nav") ||
      document.querySelector("header");

    const btn = el("button", {
      id:      "cmp-tab-btn",
      onClick: () => toggleComparePanel(mainContent),
    }, "⚡ Compare Methods");

    if (navArea) {
      navArea.appendChild(btn);
    } else {
      // Fallback: prepend to body
      document.body.insertBefore(btn, document.body.firstChild);
    }
    return btn;
  }

  function toggleComparePanel(mainContent) {
    const panel  = document.getElementById("cmp-panel");
    const btn    = document.getElementById("cmp-tab-btn");
    const isOpen = panel.classList.contains("visible");

    if (isOpen) {
      panel.classList.remove("visible");
      btn.classList.remove("active");
      if (mainContent) mainContent.style.display = "";
    } else {
      panel.classList.add("visible");
      btn.classList.add("active");
      if (mainContent) mainContent.style.display = "none";
      // Pre-fill the compare query with whatever is in the main search box
      const q = getQuery();
      if (q) document.getElementById("cmp-query-input").value = q;
    }
  }

  // ── Boot ──────────────────────────────────────────────────────────────────

  function init() {
    injectStyles();

    // Identify the main content area to hide when compare is active
    const mainContent =
      document.getElementById("main-content") ||
      document.querySelector(".results-container") ||
      document.querySelector("main") ||
      null;

    // Build and insert the comparison panel
    const panel = buildPanel();
    document.body.appendChild(panel);

    // Inject the tab toggle button
    injectTabButton(mainContent);

    // Allow Enter key in compare query box
    document.getElementById("cmp-query-input").addEventListener("keydown", e => {
      if (e.key === "Enter") handleSearch();
    });

    console.log("[NestAI Compare] Comparison mode loaded.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
