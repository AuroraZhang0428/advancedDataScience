/**
 * compare.js — NestAI "How It Works" info panel
 * Floating button opens a static explanation page comparing the 3 methods.
 * No search inputs, no API calls — informational only.
 */

(function () {
  "use strict";

  // ── Styles ────────────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("cmp-styles")) return;
    const s = document.createElement("style");
    s.id = "cmp-styles";
    s.textContent = `
      /* ── Toggle button ── */
      #cmp-toggle-btn {
        position: fixed; top: 16px; right: 24px; z-index: 1000;
        display: flex; align-items: center; gap: 7px;
        padding: 9px 18px;
        background: linear-gradient(135deg, #2a9d8f 0%, #1a6b62 100%);
        color: #fff; border: none; border-radius: 22px;
        font-size: 13px; font-weight: 700; font-family: inherit;
        cursor: pointer; letter-spacing: .3px;
        box-shadow: 0 4px 18px rgba(42,157,143,.35);
        transition: opacity .2s, transform .2s, box-shadow .2s;
      }
      #cmp-toggle-btn:hover {
        opacity: .92; transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(42,157,143,.45);
      }
      #cmp-toggle-btn .cmp-btn-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: rgba(255,255,255,.55);
        animation: cmp-pulse 2s ease-in-out infinite;
      }
      @keyframes cmp-pulse {
        0%, 100% { opacity: .55; transform: scale(1); }
        50%       { opacity: 1;   transform: scale(1.3); }
      }

      /* ── Full-screen panel ── */
      #cmp-panel {
        display: none; position: fixed; inset: 0; z-index: 999;
        background: #07080d; overflow-y: auto;
        font-family: inherit;
      }
      #cmp-panel.visible { display: block; animation: cmp-fade-in .3s ease; }
      @keyframes cmp-fade-in { from { opacity: 0; } to { opacity: 1; } }

      /* ── Close button ── */
      #cmp-close {
        position: fixed; top: 18px; left: 20px; z-index: 1001;
        display: none; align-items: center; gap: 6px;
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
        color: #e6edf3; border-radius: 10px;
        padding: 8px 16px; font-size: 13px; font-weight: 600;
        font-family: inherit; cursor: pointer; transition: background .2s;
      }
      #cmp-close:hover { background: rgba(255,255,255,.12); }

      /* ── Panel inner layout ── */
      .cmp-inner {
        max-width: 1200px; margin: 0 auto;
        padding: 80px 32px 72px;
      }

      /* ── Hero section ── */
      .cmp-hero {
        text-align: center; margin-bottom: 64px;
      }
      .cmp-hero-badge {
        display: inline-flex; align-items: center; gap: 7px;
        background: rgba(42,157,143,.12); border: 1px solid rgba(42,157,143,.25);
        border-radius: 50px; padding: 5px 14px; margin-bottom: 22px;
        font-size: 12px; font-weight: 700; color: #2a9d8f; letter-spacing: .06em; text-transform: uppercase;
      }
      .cmp-hero h1 {
        font-size: clamp(28px, 4vw, 44px); font-weight: 800;
        color: #e6edf3; margin: 0 0 16px; letter-spacing: -.02em; line-height: 1.15;
      }
      .cmp-hero h1 span { color: #2a9d8f; }
      .cmp-hero p {
        font-size: 17px; color: #8b949e; max-width: 580px;
        margin: 0 auto; line-height: 1.7;
      }

      /* ── Section title ── */
      .cmp-section-title {
        text-align: center; margin-bottom: 36px;
      }
      .cmp-section-title h2 {
        font-size: 22px; font-weight: 700; color: #e6edf3; margin: 0 0 8px;
      }
      .cmp-section-title p { font-size: 14px; color: #8b949e; margin: 0; }

      /* ── Method cards grid ── */
      .cmp-methods {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 20px; margin-bottom: 64px;
      }
      @media (max-width: 860px) { .cmp-methods { grid-template-columns: 1fr; } }

      .cmp-method-card {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 16px; padding: 28px 24px;
        position: relative; overflow: hidden;
        transition: border-color .25s, transform .25s;
      }
      .cmp-method-card:hover { border-color: #30363d; transform: translateY(-3px); }
      .cmp-method-card.cmp-card-winner {
        border-color: rgba(42,157,143,.35);
        background: linear-gradient(160deg, rgba(42,157,143,.06) 0%, #0d1117 60%);
      }
      .cmp-method-card.cmp-card-winner::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(42,157,143,.12), transparent);
        pointer-events: none;
      }

      .cmp-winner-chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(42,157,143,.15); border: 1px solid rgba(42,157,143,.3);
        border-radius: 50px; padding: 3px 11px; margin-bottom: 18px;
        font-size: 11px; font-weight: 700; color: #2a9d8f; text-transform: uppercase; letter-spacing: .05em;
      }

      .cmp-method-icon {
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; margin-bottom: 16px;
      }
      .cmp-method-card h3 {
        font-size: 16px; font-weight: 700; color: #e6edf3; margin: 0 0 6px;
      }
      .cmp-method-card .cmp-method-tag {
        display: inline-block; font-size: 11px; font-weight: 700;
        padding: 2px 9px; border-radius: 8px; margin-bottom: 14px;
      }
      .cmp-method-card p {
        font-size: 13.5px; color: #8b949e; line-height: 1.65; margin: 0 0 20px;
      }

      .cmp-pros-cons { display: flex; flex-direction: column; gap: 6px; }
      .cmp-trait {
        display: flex; align-items: flex-start; gap: 8px;
        font-size: 12.5px; line-height: 1.5;
      }
      .cmp-trait-icon { flex-shrink: 0; font-size: 13px; margin-top: 1px; }
      .cmp-trait-text { color: #8b949e; }
      .cmp-trait.good .cmp-trait-text { color: #c9d1d9; }

      /* ── Comparison table ── */
      .cmp-table-wrap {
        background: #0d1117; border: 1px solid #21262d; border-radius: 16px;
        overflow: hidden; margin-bottom: 64px;
      }
      .cmp-table {
        width: 100%; border-collapse: collapse;
      }
      .cmp-table th {
        padding: 14px 20px; font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: .06em;
        border-bottom: 1px solid #21262d; text-align: left;
      }
      .cmp-table th:first-child { color: #484f58; }
      .cmp-table th.th-filter { color: #e07b39; }
      .cmp-table th.th-llm    { color: #7c6bdf; }
      .cmp-table th.th-agent  { color: #2a9d8f; }
      .cmp-table td {
        padding: 13px 20px; font-size: 13px; color: #8b949e;
        border-bottom: 1px solid #161b22;
      }
      .cmp-table td:first-child { color: #c9d1d9; font-weight: 600; }
      .cmp-table tr:last-child td { border-bottom: none; }
      .cmp-table tr:hover td { background: rgba(255,255,255,.02); }
      .cmp-check { color: #3fb950; font-size: 15px; }
      .cmp-cross  { color: #f85149; font-size: 15px; }
      .cmp-partial { color: #d29922; font-size: 13px; font-weight: 600; }

      /* ── How to use section ── */
      .cmp-howto {
        background: linear-gradient(135deg, rgba(42,157,143,.07) 0%, rgba(124,107,223,.07) 100%);
        border: 1px solid rgba(42,157,143,.18);
        border-radius: 16px; padding: 36px 40px; margin-bottom: 0;
        text-align: center;
      }
      .cmp-howto h2 { font-size: 20px; font-weight: 700; color: #e6edf3; margin: 0 0 12px; }
      .cmp-howto p  { font-size: 14px; color: #8b949e; margin: 0 0 28px; line-height: 1.7; }
      .cmp-steps {
        display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
      }
      .cmp-step {
        display: flex; align-items: center; gap: 10px;
        background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
        border-radius: 12px; padding: 12px 18px;
        font-size: 13px; color: #c9d1d9; font-weight: 500;
      }
      .cmp-step-num {
        width: 24px; height: 24px; border-radius: 50%;
        background: rgba(42,157,143,.2); border: 1px solid rgba(42,157,143,.4);
        color: #2a9d8f; font-size: 12px; font-weight: 800;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      }
    `;
    document.head.appendChild(s);
  }

  // ── Build panel ───────────────────────────────────────────────────────────

  function buildPanel() {
    const panel = document.createElement("div");
    panel.id = "cmp-panel";
    panel.innerHTML = `
      <div class="cmp-inner">

        <!-- Hero -->
        <div class="cmp-hero">
          <div class="cmp-hero-badge">⚡ Method Comparison</div>
          <h1>Why <span>NestAI</span> finds better apartments</h1>
          <p>Most tools either blindly filter by price or dump listings into a chatbot. NestAI uses a multi-step AI agent that reasons, adapts, and explains — just like a real leasing consultant.</p>
        </div>

        <!-- Method Cards -->
        <div class="cmp-section-title">
          <h2>The Three Approaches</h2>
          <p>Same query. Three very different strategies.</p>
        </div>

        <div class="cmp-methods">

          <!-- Baseline 1 -->
          <div class="cmp-method-card">
            <div class="cmp-method-icon" style="background:rgba(224,123,57,.1)">🔍</div>
            <h3>Filter-Based Search</h3>
            <span class="cmp-method-tag" style="background:rgba(224,123,57,.12);color:#e07b39">Baseline 1 · No AI</span>
            <p>Parses your query with regex patterns and hard filters, then sorts results by price. What you'd get from a basic property portal.</p>
            <div class="cmp-pros-cons">
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Instant results, fully deterministic</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">No semantic understanding of nuance</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">Fails on vague requests ("quiet, remote-work friendly")</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">Returns nothing when filters are too strict</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">No ranking beyond price — no explanations</span></div>
            </div>
          </div>

          <!-- Baseline 2 -->
          <div class="cmp-method-card">
            <div class="cmp-method-icon" style="background:rgba(124,107,223,.1)">💬</div>
            <h3>Standard LLM Chatbot</h3>
            <span class="cmp-method-tag" style="background:rgba(124,107,223,.12);color:#7c6bdf">Baseline 2 · Single GPT Call</span>
            <p>Samples up to 60 listings and sends them as plain text to GPT-4o-mini in one shot. Simulates pasting your query into ChatGPT.</p>
            <div class="cmp-pros-cons">
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Understands natural language queries</span></div>
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Can reason about trade-offs in context</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">Only sees a random sample of 60 listings</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">No constraint enforcement or scoring pipeline</span></div>
              <div class="cmp-trait"><span class="cmp-trait-icon">✗</span><span class="cmp-trait-text">Cannot adapt when initial results are weak</span></div>
            </div>
          </div>

          <!-- NestAI -->
          <div class="cmp-method-card cmp-card-winner">
            <div class="cmp-winner-chip">⭐ Our System</div>
            <div class="cmp-method-icon" style="background:rgba(42,157,143,.12)">🏠</div>
            <h3>NestAI Agent</h3>
            <span class="cmp-method-tag" style="background:rgba(42,157,143,.12);color:#2a9d8f">LangGraph ReAct Pipeline</span>
            <p>A LangGraph ReAct agent that parses structured preferences, applies hard constraints, scores with a multi-dimensional formula, and adapts when results fall short.</p>
            <div class="cmp-pros-cons">
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Searches the entire dataset — no sampling limit</span></div>
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Multi-dimensional scoring: price, reviews, location, amenities, lifestyle</span></div>
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Autonomously relaxes constraints when needed</span></div>
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Enriches shortlist with live Google Maps data</span></div>
              <div class="cmp-trait good"><span class="cmp-trait-icon">✓</span><span class="cmp-trait-text">Explains every recommendation with factual reasoning</span></div>
            </div>
          </div>

        </div>

        <!-- Comparison table -->
        <div class="cmp-section-title">
          <h2>Feature Comparison</h2>
        </div>
        <div class="cmp-table-wrap">
          <table class="cmp-table">
            <thead>
              <tr>
                <th>Capability</th>
                <th class="th-filter">Filter Search</th>
                <th class="th-llm">LLM Chatbot</th>
                <th class="th-agent">NestAI Agent</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Natural language understanding</td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-check">✓</span></td><td><span class="cmp-check">✓</span></td></tr>
              <tr><td>Searches full dataset</td><td><span class="cmp-check">✓</span></td><td><span class="cmp-cross">✗ 60 listings only</span></td><td><span class="cmp-check">✓</span></td></tr>
              <tr><td>Multi-dimensional scoring</td><td><span class="cmp-cross">✗ Price only</span></td><td><span class="cmp-partial">~ Implicit</span></td><td><span class="cmp-check">✓ 6 dimensions</span></td></tr>
              <tr><td>Adapts when results are weak</td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-check">✓ Autonomous</span></td></tr>
              <tr><td>Live location enrichment</td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-check">✓ Google Maps</span></td></tr>
              <tr><td>Per-listing explanations</td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-partial">~ Generic</span></td><td><span class="cmp-check">✓ Factual, scored</span></td></tr>
              <tr><td>Handles vague/lifestyle queries</td><td><span class="cmp-cross">✗</span></td><td><span class="cmp-partial">~ Partial</span></td><td><span class="cmp-check">✓</span></td></tr>
            </tbody>
          </table>
        </div>

        <!-- How to use -->
        <div class="cmp-howto">
          <h2>How to see the comparison yourself</h2>
          <p>After running any search, click <strong>⚖️ Compare Baselines</strong> in the results bar to run all three methods on your exact query and see the results side by side.</p>
          <div class="cmp-steps">
            <div class="cmp-step"><div class="cmp-step-num">1</div>Type your apartment query and click Search</div>
            <div class="cmp-step"><div class="cmp-step-num">2</div>Review NestAI's AI-powered results</div>
            <div class="cmp-step"><div class="cmp-step-num">3</div>Click ⚖️ Compare Baselines to see all 3 methods</div>
          </div>
        </div>

      </div>
    `;
    return panel;
  }

  // ── Boot ──────────────────────────────────────────────────────────────────

  function init() {
    injectStyles();

    const panel     = buildPanel();
    const toggleBtn = document.createElement("button");
    const closeBtn  = document.createElement("button");

    toggleBtn.id = "cmp-toggle-btn";
    toggleBtn.innerHTML = '<span class="cmp-btn-dot"></span>⚡ Compare Methods';

    closeBtn.id = "cmp-close";
    closeBtn.innerHTML = "← Back";

    document.body.appendChild(panel);
    document.body.appendChild(toggleBtn);
    document.body.appendChild(closeBtn);

    function open() {
      panel.classList.add("visible");
      closeBtn.style.display = "flex";
      toggleBtn.style.display = "none";
      document.body.style.overflow = "hidden";
    }
    function close() {
      panel.classList.remove("visible");
      closeBtn.style.display = "none";
      toggleBtn.style.display = "flex";
      document.body.style.overflow = "";
    }

    toggleBtn.addEventListener("click", open);
    closeBtn.addEventListener("click", close);

    // Close on Escape key
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });

    console.log("[NestAI Compare] info panel loaded.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
