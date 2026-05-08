/* ─── NestAI Frontend App ─── */
(function () {
  "use strict";

  const API = "";
  let savedDataset = "matched_subset_dataset.csv";
  let lastBaseQuery = "";
  let lastAgentResult = null;

  /* ── DOM refs ── */
  const $ = (s) => document.getElementById(s);
  const queryInput = $("queryInput");
  const searchBtn = $("searchBtn");
  const btnText = searchBtn.querySelector(".btn-text");
  const btnSpinner = $("btnSpinner");
  const resultsSection = $("resultsSection");
  const cardsGrid = $("cardsGrid");
  const statusTitle = $("statusTitle");
  const statusSubtitle = $("statusSubtitle");
  const statusIcon = $("statusIcon");
  const relaxBanner = $("relaxationBanner");
  const relaxText = $("relaxationText");
  const questionBanner = $("questionBanner");
  const questionText = $("questionText");
  const querySummaryText = $("querySummaryText");
  const preferencesPanel = $("preferencesPanel");
  const preferencesGrid = $("preferencesGrid");
  const agentTracePanel = $("agentTracePanel");
  const agentTraceList = $("agentTraceList");
  const loadingOverlay = $("loadingOverlay");
  const errorToast = $("errorToast");
  const errorMessage = $("errorMessage");
  const modalOverlay = $("modalOverlay");
  const modalContent = $("modalContent");
  const settingsPanel = $("settingsPanel");
  const settingsOverlay = $("settingsOverlay");
  const compareBtn = $("compareBtn");
  const comparisonSection = $("comparisonSection");

  /* ── Settings ── */
  $("settingsToggle").addEventListener("click", () => {
    settingsPanel.classList.add("open");
    settingsOverlay.classList.add("open");
  });
  function closeSettings() {
    settingsPanel.classList.remove("open");
    settingsOverlay.classList.remove("open");
  }
  $("settingsClose").addEventListener("click", closeSettings);
  settingsOverlay.addEventListener("click", closeSettings);
  $("saveSettings").addEventListener("click", () => {
    savedDataset = $("datasetInput").value.trim() || "matched_subset_dataset.csv";
    closeSettings();
  });

  /* ── Chips ── */
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.query;
      queryInput.focus();
    });
  });

  /* ── Search ── */
  searchBtn.addEventListener("click", doSearch);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSearch(); }
  });

  async function doSearch() {
    const query = queryInput.value.trim();
    if (!query) return;
    lastBaseQuery = query;

    setLoading(true);
    resultsSection.style.display = "none";
    hideToast();

    try {
      const res = await fetch(API + "/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, dataset: savedDataset }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Search failed");
      lastAgentResult = data;
      renderResults(query, data);
      if (compareBtn) compareBtn.style.display = data.recommendations && data.recommendations.length ? "inline-flex" : "none";
      if (comparisonSection) comparisonSection.style.display = "none";
    } catch (err) {
      showToast(err.message);
    } finally {
      setLoading(false);
    }
  }

  /* ── Clarification flow ── */
  async function answerClarification(sessionId, questionKey, answer) {
    setLoading(true);
    // Keep results section visible but show a "Thinking…" state
    statusTitle.textContent = "Agent is rethinking…";
    statusSubtitle.textContent = "Adjusting search based on your answer";
    hideClarificationCard();

    try {
      const res = await fetch(API + "/api/clarify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, answer, question_key: questionKey }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Clarification failed");
      renderResults(querySummaryText.textContent, data);
    } catch (err) {
      showToast(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function searchWithFeedback(feedback) {
    const originalQuery = lastBaseQuery || querySummaryText.textContent || queryInput.value.trim();
    const revisedQuery = originalQuery + ". User feedback: " + feedback;
    queryInput.value = revisedQuery;
    closeModal();
    await doSearch();
  }

  /* ── Baseline Comparison ── */
  compareBtn.addEventListener("click", function () {
    comparisonSection.style.display = "block";
    comparisonSection.scrollIntoView({ behavior: "smooth", block: "start" });
    runComparison();
  });

  $("closeCompareBtn").addEventListener("click", function () {
    comparisonSection.style.display = "none";
  });

  async function runComparison() {
    var query = lastBaseQuery || querySummaryText.textContent || "";
    if (!query) return;

    $("compareLoading").style.display = "flex";
    $("comparisonColumns").style.display = "none";
    $("compareObservations").style.display = "none";

    try {
      var filterPromise = fetch(API + "/api/search/baseline-filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, dataset: savedDataset }),
      }).then(function (r) { return r.json(); });

      var chatbotPromise = fetch(API + "/api/search/baseline-llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, dataset: savedDataset }),
      }).then(function (r) { return r.json(); });

      var results = await Promise.allSettled([filterPromise, chatbotPromise]);
      var filterData = results[0].status === "fulfilled" ? results[0].value : { error: String(results[0].reason) };
      var chatbotData = results[1].status === "fulfilled" ? results[1].value : { error: String(results[1].reason) };

      renderComparisonColumn("agentColCards", "agentColTrace", "agentColMeta", lastAgentResult || {}, "agent");
      renderComparisonColumn("filterColCards", "filterColTrace", "filterColMeta", filterData, "filter");
      renderComparisonColumn("chatbotColCards", "chatbotColTrace", "chatbotColMeta", chatbotData, "chatbot");
      renderObservations(lastAgentResult, filterData, chatbotData);

      $("comparisonColumns").style.display = "grid";
      $("compareObservations").style.display = "block";
    } catch (err) {
      showToast("Comparison failed: " + err.message);
    } finally {
      $("compareLoading").style.display = "none";
    }
  }

  function renderComparisonColumn(cardsId, traceId, metaId, data, type) {
    var cardsEl = $(cardsId);
    var traceEl = $(traceId);
    var metaEl = $(metaId);
    cardsEl.innerHTML = "";
    traceEl.innerHTML = "";

    if (data.error) {
      cardsEl.innerHTML = "<div class=\"col-error\">&#9888; " + esc(data.error) + "</div>";
      if (metaEl) metaEl.textContent = "Error";
      return;
    }

    var recs = data.recommendations || [];
    var explanations = data.explanations || [];
    if (metaEl) metaEl.textContent = recs.length ? recs.length + " result" + (recs.length > 1 ? "s" : "") : "No results";

    var trace = data.agent_trace || [];
    if (trace.length) {
      var traceHTML = "<div class=\"col-trace-inner\">";
      trace.forEach(function (t, i) {
        traceHTML += "<div class=\"col-trace-step\"><span class=\"col-trace-n\">" + (i + 1) + "</span><span>" + esc(t.step) + "</span></div>";
      });
      traceHTML += "</div>";
      traceEl.innerHTML = traceHTML;
    }

    if (!recs.length) {
      cardsEl.innerHTML = "<div class=\"col-empty\">No listings matched this query.</div>";
      return;
    }

    recs.forEach(function (rec, idx) {
      var card = document.createElement("div");
      card.className = "compare-card";
      var priceText = rec.price != null ? "$" + rec.price.toLocaleString("en-US", { maximumFractionDigits: 0 }) + "/night" : "N/A";
      var explanation = explanations[idx] || rec.llm_rank_reason || "";
      var meta = "";
      if (rec.bedrooms != null) meta += "<span>\uD83D\uDECF " + rec.bedrooms + "bd</span>";
      if (rec.bathrooms != null) meta += "<span>\uD83D\uDEBF " + rec.bathrooms + "ba</span>";
      if (rec.review_rating != null) meta += "<span>\u2B50 " + Number(rec.review_rating).toFixed(1) + "</span>";
      if (rec.wifi) meta += "<span>\uD83D\uDCF6 WiFi</span>";
      if (rec.score > 0) meta += "<span class=\"cc-score\">Score: " + Math.round(rec.score * 100) + "%</span>";
      card.innerHTML =
        "<div class=\"cc-rank\">#" + (idx + 1) + "</div>" +
        "<div class=\"cc-title\">" + esc(rec.title) + "</div>" +
        "<div class=\"cc-neighborhood\">\uD83D\uDCCD " + esc(rec.neighborhood) + "</div>" +
        "<div class=\"cc-price\">" + priceText + "</div>" +
        "<div class=\"cc-meta\">" + meta + "</div>" +
        (explanation ? "<div class=\"cc-reason\">" + esc(explanation) + "</div>" : "");
      cardsEl.appendChild(card);
    });
  }

  function renderObservations(agentData, filterData, chatbotData) {
    var obs = [];
    var agentRecs = (agentData && agentData.recommendations) || [];
    var filterRecs = (filterData && filterData.recommendations) || [];
    var chatbotRecs = (chatbotData && chatbotData.recommendations) || [];

    if (filterRecs.length === 0 && agentRecs.length > 0) {
      obs.push("&#128270; <strong>Filter-based search returned 0 results</strong> &#8212; the regex parser could not extract a constraint that NestAI understood via LLM parsing.");
    } else if (agentRecs.length > filterRecs.length) {
      obs.push("&#128200; NestAI found <strong>" + agentRecs.length + " results</strong> vs filter-based <strong>" + filterRecs.length + "</strong>. Adaptive relaxation recovered more matches.");
    } else if (filterRecs.length >= agentRecs.length && filterRecs.length > 0) {
      obs.push("&#128203; Filter-based returned <strong>" + filterRecs.length + " results</strong> (all hard-constraint matches). NestAI returned <strong>" + agentRecs.length + "</strong> after semantic scoring.");
    }

    if (filterRecs.length >= 2) {
      var prices = filterRecs.map(function (r) { return r.price; }).filter(function (p) { return p != null; });
      var isSorted = prices.every(function (p, i, a) { return i === 0 || p >= a[i - 1]; });
      if (isSorted) obs.push("&#128178; Filter-based results are sorted by price ascending &#8212; no quality, lifestyle, or relevance scoring applied.");
    }

    if (agentRecs.length) {
      var avgScore = agentRecs.reduce(function (s, r) { return s + (r.score || 0); }, 0) / agentRecs.length;
      obs.push("&#127919; NestAI average composite score: <strong>" + avgScore.toFixed(2) + "</strong> (0&#8211;1 scale combining reviews, amenities, location, purpose fit, and price). Baselines have no composite scoring.");
    }

    var hallucinated = chatbotRecs.filter(function (r) { return r.llm_rank_reason && r.llm_rank_reason.indexOf("not found in dataset") !== -1; });
    if (hallucinated.length) {
      obs.push("&#9888; The LLM chatbot hallucinated <strong>" + hallucinated.length + " listing(s)</strong> not in the dataset &#8212; a known risk of ungrounded chatbot approaches.");
    } else if (chatbotRecs.length > 0) {
      obs.push("&#9989; LLM chatbot successfully grounded all recommendations in real dataset listings.");
    }

    var relaxHistory = (agentData && agentData.relaxation_history) || [];
    if (relaxHistory.length) {
      obs.push("&#128260; NestAI performed <strong>" + relaxHistory.length + " adaptive adjustment(s)</strong> during the search. Neither baseline supports this &#8212; they fail silently or return nothing.");
    }

    if (!obs.length) obs.push("&#8505; Run a query with specific constraints (budget, bedrooms, neighborhood) to see clear differences between the three approaches.");

    var list = $("observationsList");
    list.innerHTML = obs.map(function (o) { return "<li>" + o + "</li>"; }).join("");
  }

  /* ── Loading ── */
  function setLoading(on) {
    loadingOverlay.style.display = on ? "flex" : "none";
    searchBtn.disabled = on;
    btnText.classList.toggle("hidden", on);
    btnSpinner.classList.toggle("hidden", !on);
    if (on) animateSteps();
  }

  function animateSteps() {
    const steps = [$("step1"), $("step2"), $("step3"), $("step4")];
    steps.forEach((s) => { s.classList.remove("active"); s.querySelector(".step-dot").className = "step-dot"; });
    steps[0].classList.add("active");
    steps[0].querySelector(".step-dot").classList.add("active");
    let i = 1;
    const iv = setInterval(() => {
      if (i > 0) { steps[i - 1].querySelector(".step-dot").classList.remove("active"); steps[i - 1].querySelector(".step-dot").classList.add("done"); }
      if (i < steps.length) {
        steps[i].classList.add("active");
        steps[i].querySelector(".step-dot").classList.add("active");
        i++;
      } else clearInterval(iv);
    }, 900);
  }

  /* ── Render Results ── */
  function renderResults(query, data) {
    const recs = data.recommendations || [];
    const explanations = data.explanations || [];

    // Status bar
    if (data.need_user_input) {
      statusTitle.textContent = "Agent needs clarification";
      statusSubtitle.textContent = "Answer the question below to continue";
      statusIcon.textContent = "?";
      statusIcon.style.background = "rgba(96,165,250,.1)";
      statusIcon.style.color = "var(--blue)";
    } else {
      statusTitle.textContent = recs.length
        ? `Found ${recs.length} recommendation${recs.length > 1 ? "s" : ""}`
        : "No results found";
      statusSubtitle.textContent = recs.length ? "Ranked by AI scoring" : "Try broadening your search";
      statusIcon.textContent = recs.length ? "✓" : "—";
      statusIcon.style.background = recs.length ? "var(--green-dim)" : "var(--amber-dim)";
      statusIcon.style.color = recs.length ? "var(--green)" : "var(--amber)";
    }

    // Relaxation history banner
    const history = data.relaxation_history || [];
    const autonomousActions = history.filter(e => e.action === "relax_soft" || e.action === "relax_hard");
    if (autonomousActions.length) {
      relaxBanner.style.display = "flex";
      const latest = autonomousActions[autonomousActions.length - 1];
      relaxText.textContent = latest.reason || "The agent adjusted some preferences to find better matches.";
    } else {
      relaxBanner.style.display = "none";
    }

    // Old question banner (legacy — hidden in favour of the inline card below)
    questionBanner.style.display = "none";

    querySummaryText.textContent = query;
    renderDetectedPreferences(data.detected_preferences);
    renderAgentTrace(data.agent_trace);
    cardsGrid.innerHTML = "";

    // ── Clarification card (inline, above results) ──────────────────────────
    if (data.need_user_input && data.user_question && data.session_id) {
      cardsGrid.appendChild(
        createClarificationCard(data.user_question, data.session_id, data.question_key)
      );
    }

    recs.forEach((rec, idx) => {
      cardsGrid.appendChild(createCard(rec, idx, explanations[idx] || ""));
    });

    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderDetectedPreferences(prefs) {
    const items = buildPreferenceItems(prefs);
    if (!items.length) {
      preferencesPanel.style.display = "none";
      preferencesGrid.innerHTML = "";
      return;
    }

    preferencesGrid.innerHTML = items
      .map(([label, value]) => `
        <div class="preference-chip">
          <span>${esc(label)}</span>
          <strong>${esc(value)}</strong>
        </div>
      `)
      .join("");

    preferencesPanel.style.display = "block";
  }

  function buildPreferenceItems(prefs) {
    if (!prefs) return [];
    const hard = prefs.hard_constraints || {};
    const soft = prefs.soft_preferences || {};
    const raw = prefs.raw_preferences || {};
    const items = [];

    const firstValue = (...values) => values.find((v) => {
      if (Array.isArray(v)) return v.length;
      return v !== undefined && v !== null && v !== "";
    });

    const maxPrice = firstValue(hard.max_price, soft.target_price, raw.max_price, raw.budget);
    if (maxPrice) items.push(["Budget", typeof maxPrice === "number" ? `$${maxPrice} max` : maxPrice]);

    const bedrooms = firstValue(hard.min_bedrooms, raw.min_bedrooms, raw.bedrooms);
    if (bedrooms !== undefined && bedrooms !== null && bedrooms !== "") items.push(["Bedrooms", `${bedrooms}+`]);

    const bathrooms = firstValue(hard.min_bathrooms, raw.min_bathrooms, raw.bathrooms);
    if (bathrooms !== undefined && bathrooms !== null && bathrooms !== "") items.push(["Bathrooms", `${bathrooms}+`]);

    const neighborhoods = firstValue(
      hard.neighborhoods,
      soft.preferred_neighborhoods,
      soft.neighborhoods,
      raw.preferred_neighborhoods,
      raw.neighborhoods
    );
    if (neighborhoods) items.push(["Location", formatValue(neighborhoods)]);

    const amenities = firstValue(soft.desired_amenities, soft.amenities, raw.desired_amenities, raw.amenities);
    if (amenities) items.push(["Amenities", formatValue(amenities)]);

    const priorities = firstValue(soft.priorities, raw.priorities, raw.preferences, raw.purpose_tags);
    if (priorities) items.push(["Priorities", formatValue(priorities)]);

    const roomType = firstValue(hard.room_type, raw.room_type);
    if (roomType) items.push(["Room Type", formatValue(roomType)]);

    return items.slice(0, 8);
  }

  function renderAgentTrace(trace) {
    if (!Array.isArray(trace) || !trace.length) {
      agentTracePanel.style.display = "none";
      agentTraceList.innerHTML = "";
      return;
    }

    agentTraceList.innerHTML = trace
      .map((item, idx) => `
        <div class="trace-step">
          <div class="trace-number">${idx + 1}</div>
          <div class="trace-copy">
            <strong>${esc(item.step)}</strong>
            <p>${esc(item.detail)}</p>
          </div>
        </div>
      `)
      .join("");

    agentTracePanel.style.display = "block";
  }

  function formatValue(value) {
    if (Array.isArray(value)) return value.join(", ");
    if (value && typeof value === "object") {
      return Object.entries(value)
        .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`)
        .join("; ");
    }
    return String(value);
  }

  /* ── Clarification Card ── */
  function createClarificationCard(question, sessionId, questionKey) {
    const card = document.createElement("div");
    card.className = "clarification-card";
    card.id = "clarificationCard";

    card.innerHTML = `
      <div class="clarify-icon">🤔</div>
      <div class="clarify-body">
        <div class="clarify-label">Agent Question</div>
        <p class="clarify-question">${esc(question)}</p>
        <div class="clarify-actions">
          <button class="clarify-btn clarify-yes" id="clarifyYes">Yes, sounds good</button>
          <button class="clarify-btn clarify-no" id="clarifyNo">No, keep searching</button>
        </div>
      </div>`;

    card.querySelector("#clarifyYes").addEventListener("click", () => {
      answerClarification(sessionId, questionKey, "yes");
    });
    card.querySelector("#clarifyNo").addEventListener("click", () => {
      answerClarification(sessionId, questionKey, "no");
    });

    return card;
  }

  function hideClarificationCard() {
    const card = $("clarificationCard");
    if (card) card.remove();
  }

  /* ── Card Creation ── */
  function createCard(rec, idx, explanation) {
    const el = document.createElement("div");
    el.className = "listing-card";
    const score = rec.score || 0;
    const pct = Math.round(score * 100);
    const r = 9;
    const circ = 2 * Math.PI * r;
    const offset = circ - (score * circ);
    const scoreColor = score >= 0.7 ? "var(--green)" : score >= 0.5 ? "var(--amber)" : "var(--red)";
    const priceText = rec.price != null ? `$${rec.price.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "N/A";
    const cx = 13, cy = 13;

    el.innerHTML = `
      <div class="card-header">
        <div class="card-rank-badge">${idx + 1}</div>
        <span class="card-neighborhood-inline">📍 ${esc(rec.neighborhood)}</span>
        <div class="card-score-pill">
          <svg class="score-circle" viewBox="0 0 26 26">
            <circle class="score-bg" cx="${cx}" cy="${cy}" r="${r}"/>
            <circle class="score-fg" cx="${cx}" cy="${cy}" r="${r}"
              stroke="${scoreColor}"
              stroke-dasharray="${circ.toFixed(2)}"
              stroke-dashoffset="${offset.toFixed(2)}"
              transform="rotate(-90 ${cx} ${cy})"/>
          </svg>
          <span class="score-num" style="color:${scoreColor}">${pct}%</span>
        </div>
      </div>
      <div class="card-body">
        <div class="card-title">${esc(rec.title)}</div>
        <div class="card-details">
          ${rec.bedrooms != null ? `<span class="detail-tag">🛏 ${rec.bedrooms} bed</span>` : ""}
          ${rec.bathrooms != null ? `<span class="detail-tag">🚿 ${rec.bathrooms} bath</span>` : ""}
          ${rec.review_rating != null ? `<span class="detail-tag">⭐ ${Number(rec.review_rating).toFixed(1)}</span>` : ""}
          ${rec.wifi ? `<span class="detail-tag">📶 WiFi</span>` : ""}
          ${rec.workspace ? `<span class="detail-tag">💻 Desk</span>` : ""}
        </div>
        <div class="card-price-row">
          <span class="card-price">${priceText}</span>
          <span class="card-price-label">/night</span>
        </div>
      </div>
      <div class="card-footer">
        <span class="card-footer-label">AI Match</span>
        <span class="card-footer-action">View Details <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m9 18 6-6-6-6"/></svg></span>
      </div>`;

    el.addEventListener("click", () => openModal(rec, idx, explanation));
    return el;
  }

  /* ── Modal ── */
  function openModal(rec, idx, explanation) {
    const priceText = rec.price != null ? `$${rec.price.toLocaleString("en-US", { maximumFractionDigits: 0 })}` : "N/A";
    const bd = rec.score_breakdown || {};

    let breakdownHTML = "";
    const labelMap = {
      review_rating: "Reviews", amenity_match: "Amenities",
      purpose_alignment: "Purpose Fit", neighborhood_fit: "Location",
      price_score: "Price Value", llm_fit: "AI Fit",
    };
    for (const [k, v] of Object.entries(bd)) {
      const pct = Math.round(v * 100);
      const color = v >= 0.7 ? "var(--green)" : v >= 0.5 ? "var(--amber)" : "var(--red)";
      breakdownHTML += `
        <div class="breakdown-item">
          <div class="breakdown-label">${labelMap[k] || k}</div>
          <div class="breakdown-bar"><div class="breakdown-fill" style="width:${pct}%;background:${color}"></div></div>
          <div class="breakdown-value" style="color:${color}">${pct}%</div>
        </div>`;
    }

    let tagsHTML = "";
    (rec.amenities || []).forEach((a) => { tagsHTML += `<span class="tag">${esc(a)}</span>`; });
    (rec.purpose_tags || []).forEach((t) => { tagsHTML += `<span class="tag">${esc(t)}</span>`; });

    modalContent.innerHTML = `
      <div class="modal-inner">
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem">
          <span style="font-size:.75rem;color:var(--accent-light);font-weight:600">#${idx + 1} MATCH</span>
          <span style="font-size:.75rem;color:var(--text-dim)">Score: ${(rec.score * 100).toFixed(0)}%</span>
        </div>
        <h2 class="modal-title">${esc(rec.title)}</h2>
        <p class="modal-neighborhood">📍 ${esc(rec.neighborhood)}${rec.neighborhood_group ? " · " + esc(rec.neighborhood_group) : ""}</p>
        <div class="modal-price">${priceText}<span style="font-size:.85rem;color:var(--text-dim);font-weight:400"> /night</span></div>

        ${explanation ? `<div class="modal-section"><h4>AI Explanation</h4><div class="modal-explanation">${esc(explanation)}</div></div>` : ""}

        <div class="modal-section">
          <h4>Score Breakdown</h4>
          <div class="breakdown-grid">${breakdownHTML}</div>
        </div>

        ${tagsHTML ? `<div class="modal-section"><h4>Amenities & Tags</h4><div class="tags-row">${tagsHTML}</div></div>` : ""}

        ${rec.llm_rank_reason ? `<div class="modal-section"><h4>AI Ranking Reason</h4><p style="font-size:.85rem;color:var(--text-muted)">${esc(rec.llm_rank_reason)}</p></div>` : ""}

        <div class="feedback-box">
          <h4>Not quite right?</h4>
          <p>Give the agent feedback and it will rethink the recommendation.</p>
          <div class="feedback-actions">
            <button class="feedback-btn" data-feedback="These options are too expensive. Please find cheaper listings and prioritize price more.">Too expensive</button>
            <button class="feedback-btn" data-feedback="I want a better location or closer transit access. Please prioritize location more.">Better location</button>
            <button class="feedback-btn" data-feedback="I care more about review quality and overall comfort. Please prioritize higher-rated listings.">Better reviews</button>
          </div>
        </div>
      </div>`;

    modalContent.querySelectorAll(".feedback-btn").forEach((btn) => {
      btn.addEventListener("click", () => searchWithFeedback(btn.dataset.feedback));
    });

    modalOverlay.style.display = "flex";
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    modalOverlay.style.display = "none";
    document.body.style.overflow = "";
  }
  $("modalClose").addEventListener("click", closeModal);
  modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

  /* ── Toast ── */
  function showToast(msg) { errorMessage.textContent = msg; errorToast.classList.add("show"); }
  function hideToast() { errorToast.classList.remove("show"); }
  $("errorClose").addEventListener("click", hideToast);

  /* ── New Search ── */
  $("newSearchBtn").addEventListener("click", () => {
    resultsSection.style.display = "none";
    if (preferencesPanel) preferencesPanel.style.display = "none";
    if (agentTracePanel) agentTracePanel.style.display = "none";
    window.scrollTo({ top: 0, behavior: "smooth" });
    queryInput.focus();
  });

  /* ── Util ── */
  function esc(s) {
    if (s == null) return "";
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }
})();
