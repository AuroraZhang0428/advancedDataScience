/**
 * compare.js — NestAI Comparison Mode (v2)
 * Dark theme, fixed search input, proper panel toggle.
 */

(function () {
  "use strict";

  const ENDPOINTS = {
    filter: "/api/search/baseline-filter",
    llm:    "/api/search/baseline-llm",
    agent:  "/api/search",
  };

  const COLUMNS = [
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

  const state = {
    results: { filter: null, llm: null, agent: null },
    loading: { filter: false, llm: false, agent: false },
    errors:  { filter: null, llm: null, agent: null },
    timings: { filter: null, llm: null, agent: null },
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "className") node.className = v;
      else if (k === "style") Object.assign(node.style, v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function getApiKey() {
    const stored = localStorage.getItem("nestai_api_key") || "";
    if (stored) return stored;
    const inp = document.querySelector('input[type="password"]');
    return inp ? inp.value.trim() : "";
  }

  // ── API ───────────────────────────────────────────────────────────────────

  async function fetchMethod(method, query, apiKey) {
    const body = { query, dataset: "matched_subset_dataset.csv" };
    if (method === "llm" || method === "agent") body.api_key = apiKey;
    const t0 = performance.now();
    const res = await fetch(ENDPOINTS[method], {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw Object.assign(new Error(err.error || "Request failed"), { elapsed });
    }
    return { data: await res.json(), elapsed };
  }

  // ── Card rendering ────────────────────────────────────────────────────────

  function renderCard(listing, meta, rank, agentExplanation) {
    const price  = listing.price != null ? `$${Number(listing.price).toFixed(0)}/night` : "N/A";
    const rating = listing.review_rating != null ? `${Number(listing.review_rating).toFixed(2)}★` : "—";
    const beds   = listing.bedrooms != null ? `${listing.bedrooms}BR` : "?BR";
    const baths  = listing.bathrooms != null ? `${Number(listing.bathrooms).toFixed(1)}BA` : "?BA";
    const nbhd   = listing.neighborhood || "—";
    const ams    = (listing.amenities || []).slice(0, 4);
    const expl   = listing.llm_explanation || agentExplanation || null;

    return el("div", { className: "cmp-card" },
      el("div", { className: "cmp-card-rank", style: { background: meta.color } }, `#${rank}`),
      el("div", { className: "cmp-card-body" },
        el("div", { className: "cmp-card-title" }, listing.title || "Untitled"),
        el("div", { className: "cmp-card-meta" },
          el("span", { className: "cmp-tag" }, nbhd),
          el("span", { className: "cmp-tag cmp-tag--price" }, price),
          el("span", { className: "cmp-tag" }, `${beds} · ${baths}`),
          el("span", { className: "cmp-tag" }, rating),
        ),
        ams.length ? el("div", { className: "cmp-amenities" },
          ...ams.map(a => el("span", { className: "cmp-amenity" }, a))
        ) : null,
        expl ? el("div", { className: "cmp-expl" }, `"${expl}"`) : null,
      )
    );
  }

  function renderColumn(meta) {
    const col = document.getElementById(`cmp-col-${meta.key}`);
    if (!col) return;
    const header = col.querySelector(".cmp-col-header");
    col.innerHTML = "";
    col.appendChild(header);

    if (state.loading[meta.key]) {
      col.appendChild(el("div", { className: "cmp-spinner-wrap" },
        el("div", { className: "cmp-spinner" }),
        el("p", { className: "cmp-spinner-label" }, "Searching…"),
      ));
      return;
    }
    if (state.errors[meta.key]) {
      col.appendChild(el("div", { className: "cmp-error" }, `⚠ ${state.errors[meta.key]}`));
      return;
    }
    const result = state.results[meta.key];
    if (!result) return;

    const recs    = result.recommendations || [];
    const expls   = result.explanations || [];
    const elapsed = state.timings[meta.key];

    const stats = [
      `${recs.length} results`,
      elapsed ? `${elapsed}s` : null,
      result.total_matched != null ? `${result.total_matched} matched` : null,
      result.listings_shown != null ? `${result.listings_shown} shown to LLM` : null,
    ].filter(Boolean);

    col.appendChild(el("div", { className: "cmp-stats" },
      ...stats.map(s => el("span", { className: "cmp-stat" }, s))
    ));
    if (result.explanation) {
      col.appendChild(el("div", { className: "cmp-method-expl" }, result.explanation));
    }
    if (recs.length === 0) {
      col.appendChild(el("div", { className: "cmp-empty" }, "No results returned."));
      return;
    }
    const list = el("div", { className: "cmp-cards" });
    recs.forEach((r, i) => list.appendChild(renderCard(r, meta, i + 1, expls[i])));
    col.appendChild(list);
  }

  // ── Search ────────────────────────────────────────────────────────────────

  async function runComparison(query) {
    const apiKey = getApiKey();
    for (const m of COLUMNS) {
      state.results[m.key] = null;
      state.errors[m.key]  = null;
      state.timings[m.key] = null;
      state.loading[m.key] = true;
    }
    COLUMNS.forEach(m => renderColumn(m));

    await Promise.allSettled(COLUMNS.map(async (meta) => {
      try {
        const { data, elapsed } = await fetchMethod(meta.key, query, apiKey);
        state.results[meta.key] = data;
        state.timings[meta.key] = elapsed;
      } catch (err) {
        state.errors[meta.key]  = err.message;
        if (err.elapsed) state.timings[meta.key] = err.elapsed;
      } finally {
        state.loading[meta.key] = false;
        renderColumn(meta);
      }
    }));
  }

  async function handleSearch() {
    const query = document.getElementById("cmp-query").value.trim();
    if (!query) { alert("Please enter a search query."); return; }
    const btn = document.getElementById("cmp-search-btn");
    btn.disabled = true;
    btn.textContent = "Searching…";
    try { await runComparison(query); }
    finally { btn.disabled = false; btn.textContent = "Compare All"; }
  }

  // ── Build panel ───────────────────────────────────────────────────────────

  function buildPanel() {
    return el("div", { id: "cmp-panel" },
      el("div", { className: "cmp-panel-header" },
        el("div", { className: "cmp-panel-title" },
          el("span", {}, "🔍"),
          el("h2", {}, "Side-by-Side Comparison"),
        ),
        el("p", { className: "cmp-panel-sub" },
          "Run the same query through all three search methods at once."
        ),
      ),
      el("div", { className: "cmp-search-row" },
        el("input", {
          id: "cmp-query",
          type: "text",
          placeholder: "e.g. 2BR with WiFi in Brooklyn under $200/night",
          onKeydown: e => { if (e.key === "Enter") handleSearch(); },
        }),
        el("button", { id: "cmp-search-btn", onClick: handleSearch }, "Compare All"),
      ),
      el("div", { id: "cmp-columns" },
        ...COLUMNS.map(meta =>
          el("div", { className: "cmp-col", id: `cmp-col-${meta.key}` },
            el("div", {
              className: "cmp-col-header",
              style: { borderBottomColor: meta.color },
            },
              el("span", { className: "cmp-badge", style: { background: meta.color } }, meta.badge),
              el("h3", {}, meta.label),
              el("p", { className: "cmp-col-desc" }, meta.desc),
            )
          )
        )
      )
    );
  }

  // ── Styles ────────────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("cmp-styles")) return;
    const s = document.createElement("style");
    s.id = "cmp-styles";
    s.textContent = `
      #cmp-toggle-btn {
        position: fixed; top: 16px; right: 180px; z-index: 1000;
        padding: 9px 20px;
        background: linear-gradient(135deg, #2a9d8f, #264653);
        color: #fff; border: none; border-radius: 22px;
        font-size: 14px; font-weight: 700; cursor: pointer;
        letter-spacing: .3px;
        box-shadow: 0 4px 16px rgba(42,157,143,.35);
        transition: opacity .2s, transform .2s;
      }
      #cmp-toggle-btn:hover { opacity: .88; transform: translateY(-1px); }

      #cmp-panel {
        display: none; position: fixed; inset: 0; z-index: 999;
        background: #0d1117; overflow-y: auto;
        padding: 72px 24px 48px; box-sizing: border-box;
        color: #e6edf3; font-family: inherit;
      }
      #cmp-panel.visible { display: block; }

      .cmp-panel-header { text-align: center; margin-bottom: 24px; }
      .cmp-panel-title {
        display: flex; align-items: center; justify-content: center;
        gap: 10px; margin-bottom: 6px;
      }
      .cmp-panel-title h2 { margin: 0; font-size: 26px; font-weight: 700; color: #e6edf3; }
      .cmp-panel-sub { color: #8b949e; font-size: 14px; margin: 0; }

      .cmp-search-row {
        display: flex; gap: 10px; align-items: center;
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 12px 16px; max-width: 860px; margin: 0 auto 28px;
      }
      #cmp-query {
        flex: 1; border: none; outline: none;
        font-size: 15px; color: #e6edf3; background: transparent;
      }
      #cmp-query::placeholder { color: #484f58; }
      #cmp-search-btn {
        padding: 9px 24px;
        background: linear-gradient(135deg, #2a9d8f, #264653);
        color: #fff; border: none; border-radius: 8px;
        font-size: 14px; font-weight: 700; cursor: pointer;
        white-space: nowrap; transition: opacity .2s;
      }
      #cmp-search-btn:hover { opacity: .85; }
      #cmp-search-btn:disabled { opacity: .5; cursor: default; }

      #cmp-columns {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 16px; max-width: 1300px; margin: 0 auto; align-items: start;
      }
      @media (max-width: 900px) { #cmp-columns { grid-template-columns: 1fr; } }

      .cmp-col {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 14px; overflow: hidden; min-height: 120px;
      }
      .cmp-col-header { padding: 14px 16px 12px; border-bottom: 3px solid #30363d; }
      .cmp-badge {
        display: inline-block; padding: 2px 9px; border-radius: 10px;
        font-size: 11px; font-weight: 700; color: #fff; margin-bottom: 6px;
      }
      .cmp-col-header h3 { margin: 0 0 4px; font-size: 15px; font-weight: 700; color: #e6edf3; }
      .cmp-col-desc { font-size: 12px; color: #8b949e; margin: 0; line-height: 1.4; }

      .cmp-stats {
        display: flex; flex-wrap: wrap; gap: 6px;
        padding: 8px 14px; background: #0d1117; border-bottom: 1px solid #21262d;
      }
      .cmp-stat {
        font-size: 11px; color: #8b949e; background: #21262d;
        padding: 2px 8px; border-radius: 10px;
      }
      .cmp-method-expl {
        padding: 8px 14px; font-size: 11px; color: #8b949e;
        font-style: italic; line-height: 1.5; border-bottom: 1px solid #21262d;
      }

      .cmp-cards { padding: 10px; display: flex; flex-direction: column; gap: 8px; }
      .cmp-card {
        display: flex; background: #0d1117; border: 1px solid #21262d;
        border-radius: 10px; overflow: hidden; transition: border-color .15s;
      }
      .cmp-card:hover { border-color: #388bfd; }
      .cmp-card-rank {
        width: 30px; min-width: 30px;
        display: flex; align-items: center; justify-content: center;
        color: #fff; font-size: 11px; font-weight: 700;
      }
      .cmp-card-body { padding: 10px 12px; flex: 1; min-width: 0; }
      .cmp-card-title {
        font-size: 13px; font-weight: 600; color: #e6edf3; margin-bottom: 5px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .cmp-card-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 5px; }
      .cmp-tag {
        font-size: 11px; background: #21262d; color: #8b949e;
        padding: 2px 7px; border-radius: 8px; white-space: nowrap;
      }
      .cmp-tag--price { background: #1a3a2a; color: #3fb950; font-weight: 600; }
      .cmp-amenities { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 4px; }
      .cmp-amenity {
        font-size: 10px; background: #1a2332; color: #79c0ff;
        padding: 1px 6px; border-radius: 6px;
      }
      .cmp-expl { font-size: 11px; color: #8b949e; font-style: italic; margin-top: 4px; line-height: 1.4; }

      .cmp-spinner-wrap { display: flex; flex-direction: column; align-items: center; padding: 40px 20px; gap: 12px; }
      .cmp-spinner {
        width: 28px; height: 28px;
        border: 3px solid #21262d; border-top-color: #2a9d8f;
        border-radius: 50%; animation: cmp-spin .8s linear infinite;
      }
      @keyframes cmp-spin { to { transform: rotate(360deg); } }
      .cmp-spinner-label { font-size: 13px; color: #8b949e; }
      .cmp-error { padding: 20px; color: #f85149; font-size: 13px; }
      .cmp-empty { padding: 20px; color: #484f58; font-size: 13px; text-align: center; }

      #cmp-close {
        display: none; position: fixed; top: 18px; left: 20px; z-index: 1001;
        background: #21262d; border: 1px solid #30363d; color: #e6edf3;
        border-radius: 8px; padding: 7px 14px; font-size: 13px; cursor: pointer;
      }
      #cmp-close:hover { background: #30363d; }
    `;
    document.head.appendChild(s);
  }

  // ── Boot ──────────────────────────────────────────────────────────────────

  function init() {
    injectStyles();

    // Remove old button if it exists
    const old = document.getElementById("cmp-tab-btn");
    if (old) old.remove();

    const panel    = buildPanel();
    const toggleBtn = el("button", { id: "cmp-toggle-btn" }, "⚡ Compare Methods");
    const closeBtn  = el("button", { id: "cmp-close" }, "← Back to NestAI");

    document.body.appendChild(panel);
    document.body.appendChild(toggleBtn);
    document.body.appendChild(closeBtn);

    function toggle() {
      const open = panel.classList.toggle("visible");
      closeBtn.style.display  = open ? "block" : "none";
      toggleBtn.textContent   = open ? "✕ Close" : "⚡ Compare Methods";
    }

    toggleBtn.addEventListener("click", toggle);
    closeBtn.addEventListener("click", toggle);

    console.log("[NestAI Compare v2] loaded.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
