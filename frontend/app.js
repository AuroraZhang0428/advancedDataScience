/* ─── NestAI Frontend App ─── */
(function () {
  "use strict";

  const API = "";
  let savedApiKey = "";
  let savedDataset = "matched_subset_dataset.csv";

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
  const loadingOverlay = $("loadingOverlay");
  const errorToast = $("errorToast");
  const errorMessage = $("errorMessage");
  const modalOverlay = $("modalOverlay");
  const modalContent = $("modalContent");
  const settingsPanel = $("settingsPanel");
  const settingsOverlay = $("settingsOverlay");

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
    savedApiKey = $("apiKeyInput").value.trim();
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

    setLoading(true);
    resultsSection.style.display = "none";
    hideToast();

    try {
      const res = await fetch(API + "/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, api_key: savedApiKey, dataset: savedDataset }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Search failed");
      renderResults(query, data);
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
      </div>`;

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
