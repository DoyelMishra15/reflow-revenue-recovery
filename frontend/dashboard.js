const $ = (id) => document.getElementById(id);

let _txnCache = [];
let _liveMetricsCache = null;
let _leakDim = "by_failure_reason";

// The API base is resolved once, in priority order:
//  1. whatever the user typed into the API field (manual override)
//  2. GitHub Codespaces: each forwarded port gets its own subdomain
//     (https://<name>-<port>.app.github.dev), so localhost is wrong here -
//     rewrite the frontend's own forwarded host (-8080) to the backend's
//     forwarded host (-8000) instead.
//  3. this exact known local-dev layout: docker-compose serves the
//     frontend container on :8080 and the backend on :8000, both via
//     plain localhost
//  4. same origin as this page - correct when a single service (see the
//     repo-root Dockerfile) serves both the API and this frontend, which
//     is how the deployed demo is set up
function defaultApiBase() {
  const host = window.location.hostname;
  const codespaces = host.match(/^(.*)-(\d+)(\.app\.github\.dev|\.githubpreview\.dev)$/);
  if (codespaces) {
    const [, prefix, , suffix] = codespaces;
    return `${window.location.protocol}//${prefix}-8000${suffix}`;
  }
  if (window.location.protocol === "file:") return "http://localhost:8000";
  if (window.location.port === "8080") return "http://localhost:8000";
  return window.location.origin;
}

function apiBase() {
  const manual = $("apiBase").value.trim();
  return (manual || defaultApiBase()).replace(/\/$/, "");
}

$("apiBase").placeholder = defaultApiBase();

function inr(n) {
  return `₹${Number(n || 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function pct(n) {
  return `${(Number(n || 0) * 100).toFixed(1)}%`;
}

async function getJSON(path, opts) {
  const res = await fetch(`${apiBase()}${path}`, opts);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------
// view switching
// ---------------------------------------------------------------------

$("mainNav").addEventListener("click", (e) => {
  const btn = e.target.closest(".nav-btn");
  if (!btn) return;
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  btn.classList.add("active");
  $(`view-${btn.dataset.view}`).classList.add("active");
});

// ---------------------------------------------------------------------
// top-level load
// ---------------------------------------------------------------------

async function loadAll() {
  await Promise.all([
    loadMetrics(),
    loadTransactions(),
    loadAudit(),
    loadIncidents(),
    loadRevenueLeak(),
    loadReflowVsBlind(),
    loadOpportunityScore(),
    loadCalibration(),
  ]);
}

// ---------------------------------------------------------------------
// metrics / KPIs
// ---------------------------------------------------------------------

async function loadMetrics() {
  try {
    const m = await getJSON("/dashboard/metrics");
    _liveMetricsCache = m;

    $("kpiRevenueAtRisk").textContent = inr(m.revenue_at_risk);
    $("kpiRevenueAtRiskSub").textContent = `${m.active_workflows} open workflows`;
    $("kpiRevenueRecovered").textContent = inr(m.revenue_recovered);
    $("kpiRevenueRecoveredSub").textContent = `${m.by_status.recovered || 0} transactions recovered`;
    $("kpiRecoveryRate").textContent = pct(m.recovery_rate);

    renderStrategyTable(m.strategy_performance || {});
    renderFailureBars(m.failure_reason_breakdown || {});
    renderModelPerf(m.model_performance, m.baseline_comparison);
    renderBaselineCompare(m.baseline_comparison);
    renderPolicyBaseline(m); // must run after _liveMetricsCache is set, not concurrently via Promise.all
  } catch (e) {
    console.error("failed to load metrics", e);
  }
}

function renderStrategyTable(perf) {
  const rows = Object.entries(perf).sort((a, b) => b[1].attempts - a[1].attempts);
  $("strategyBody").innerHTML = rows.length
    ? rows
        .map(
          ([name, s]) => `
      <tr>
        <td style="text-transform:none">${name}</td>
        <td>${s.attempts}</td>
        <td>${s.captured}</td>
        <td>${pct(s.capture_rate)}</td>
      </tr>`
        )
        .join("")
    : `<tr><td colspan="4" class="empty">No decisions run yet.</td></tr>`;
}

function renderFailureBars(breakdown) {
  const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  $("failureBars").innerHTML = entries.length
    ? entries
        .map(
          ([reason, count]) => `
      <div class="bar-row">
        <span class="bar-name">${reason}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(count / max) * 100}%"></div></div>
        <span class="bar-count">${count}</span>
      </div>`
        )
        .join("")
    : `<p class="empty">No transactions yet.</p>`;
}

function renderModelPerf(model, baseline) {
  if (!model) {
    $("modelPerf").innerHTML = `<p class="empty">No trained model metrics found — run <code>python -m app.ml.train_model</code>.</p>`;
    return;
  }
  const cls = baseline && baseline.classifier_at_production_threshold;
  let confusion = "";
  if (cls) {
    const cm = cls.confusion_matrix;
    confusion = `
      <div class="confusion-grid">
        <div class="confusion-cell tp"><div class="cc-n">${cm.true_positive}</div>true positive</div>
        <div class="confusion-cell fp"><div class="cc-n">${cm.false_positive}</div>false positive</div>
        <div class="confusion-cell fn"><div class="cc-n">${cm.false_negative}</div>false negative</div>
        <div class="confusion-cell tn"><div class="cc-n">${cm.true_negative}</div>true negative</div>
      </div>`;
  }
  $("modelPerf").innerHTML = `
    <div class="metric-tile"><div class="m-label">ROC-AUC</div><div class="m-value">${model.roc_auc}</div></div>
    <div class="metric-tile"><div class="m-label">Precision @ prod threshold</div><div class="m-value">${cls ? cls.precision : model["precision_at_0.5"]}</div></div>
    <div class="metric-tile"><div class="m-label">Recall @ prod threshold</div><div class="m-value">${cls ? cls.recall : model["recall_at_0.5"]}</div></div>
    <div class="metric-tile"><div class="m-label">F1 @ prod threshold</div><div class="m-value">${cls ? cls.f1 : "—"}</div></div>
    <div class="metric-tile" style="grid-column:1/-1">
      <div class="m-label">Confusion matrix — held-out test set (n=${model.n_test})</div>
      ${confusion || '<p class="empty">Run evaluation.evaluate to populate.</p>'}
    </div>
  `;
}

function renderBaselineCompare(baseline) {
  if (!baseline) {
    $("baselineCompare").innerHTML = `<p class="empty" style="padding:16px 18px">No evaluation output found — run <code>python -m evaluation.evaluate</code>.</p>`;
    return;
  }
  const policies = ["retry_all", "retry_all_gated", "reflow_ml_policy"];
  const labels = { retry_all: "retry_all", retry_all_gated: "retry_all_gated", reflow_ml_policy: "reflow (ours)" };
  const rowsDef = [
    ["Retry rate", (p) => pct(p.retry_rate)],
    ["Success rate when retried", (p) => pct(p.success_rate_when_retried)],
    ["Unnecessary retries", (p) => p.unnecessary_retries],
    ["False-positive cost", (p) => inr(p.false_positive_cost_rupees)],
    ["Revenue recovered", (p) => inr(p.revenue_recovered)],
  ];
  const head = `<tr><th></th>${policies.map((p) => `<th>${labels[p]}</th>`).join("")}</tr>`;
  const body = rowsDef
    .map(([label, fn]) => `<tr><td>${label}</td>${policies.map((p) => `<td>${fn(baseline[p])}</td>`).join("")}</tr>`)
    .join("");
  $("baselineCompare").innerHTML = `<table>${head}${body}</table>`;
}

// ---------------------------------------------------------------------
// transactions table (shared renderer, used by both Command Center's
// compact queue and the full Transactions view)
// ---------------------------------------------------------------------

function statusPill(status) {
  return `<span class="pill pill-${status}">${status.replace(/_/g, " ")}</span>`;
}

async function loadTransactions() {
  try {
    const txns = await getJSON("/transactions?limit=150");
    _txnCache = txns;
    renderTxnTableCompact(txns);
    renderTxnTableFull(txns);
  } catch (e) {
    console.error("failed to load transactions", e);
  }
}

function renderTxnTableCompact(txns) {
  const body = $("txnBodyCmd");
  const open = txns.filter((t) => ["failed", "delayed", "alt_method_suggested"].includes(t.status)).slice(0, 25);
  body.innerHTML = "";
  for (const t of open) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.txn_id}</td>
      <td>${inr(t.amount)}</td>
      <td>${t.payment_method}</td>
      <td>${t.failure_reason}</td>
      <td>${statusPill(t.status)}</td>
    `;
    tr.addEventListener("click", () => openWorkflow(t, { rich: false, prefix: "Cmd" }));
    body.appendChild(tr);
  }
  if (!open.length) body.innerHTML = `<tr><td colspan="5" class="empty">No open transactions right now.</td></tr>`;
}

function renderTxnTableFull(txns) {
  const body = $("txnBody");
  body.innerHTML = "";
  for (const t of txns) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.txn_id}</td>
      <td>${inr(t.amount)}</td>
      <td>${t.payment_method}</td>
      <td>${t.failure_reason}</td>
      <td>${t.attempt_no}</td>
      <td>${t.is_flagged_risk ? '<span class="risk-flag">flagged</span>' : "—"}</td>
      <td>${statusPill(t.status)}</td>
      <td><button class="secondary" data-txn="${t.txn_id}">Run</button></td>
    `;
    tr.querySelector("button").addEventListener("click", (e) => {
      e.stopPropagation();
      runDecision(t.txn_id, { openModal: false, prefix: "" });
    });
    tr.addEventListener("click", () => openWorkflow(t, { rich: true, prefix: "" }));
    body.appendChild(tr);
  }
}

// ---------------------------------------------------------------------
// recovery workflow trace (shared, parameterized by element-id prefix so
// the same rendering logic drives both the Command Center's compact panel
// and the Transactions view's full panel)
// ---------------------------------------------------------------------

const ACTION_LABEL = {
  retry_now: "Retry now",
  delay_retry: "Delay retry",
  suggest_alt_method: "Suggest alt. method",
  escalate_human: "Escalate to human",
  stop_no_retry: "Stop — no retry",
};

const ACTION_COLOR = {
  retry_now: ["#122922", "#34d399"],
  delay_retry: ["#1c1730", "#a78bfa"],
  suggest_alt_method: ["#241c33", "#c084fc"],
  escalate_human: ["#2a2313", "#fbbf24"],
  stop_no_retry: ["#1a1a26", "#8d8da3"],
};

async function runDecision(txnId, opts = {}) {
  const prefix = opts.prefix ?? "";
  const hintEl = $(`traceHint${prefix}`);
  if (hintEl) hintEl.textContent = `Running agent on ${txnId}...`;
  try {
    const res = await fetch(`${apiBase()}/transactions/${txnId}/decide`, { method: "POST" });
    const d = await res.json();
    if (!res.ok) {
      $(`traceBody${prefix}`).innerHTML = `<p class="empty">Error: ${d.detail || res.status}</p>`;
      return;
    }
    const txn = _txnCache.find((t) => t.txn_id === txnId);
    renderTrace(txnId, txn, d, prefix);
    if (opts.openModal !== false) openModalFor(txnId, txn, d);
    await loadAll();
  } catch (e) {
    $(`traceBody${prefix}`).innerHTML = `<p class="empty">Request failed: ${e}</p>`;
  }
}

function pipelineHTML(txn, d) {
  const proba = d.predicted_success_proba != null ? `${(d.predicted_success_proba * 100).toFixed(0)}%` : "gated";
  return `
    <div class="pipeline">
      <div class="pipeline-step filled"><div class="step-label">1. Detect</div><div class="step-value">${txn ? inr(txn.amount) : "—"} at risk</div></div>
      <div class="pipeline-step filled"><div class="step-label">2. Diagnose</div><div class="step-value">${d.diagnosis}</div></div>
      <div class="pipeline-step filled"><div class="step-label">3. Intervene</div><div class="step-value">${ACTION_LABEL[d.action] || d.action}</div></div>
      <div class="pipeline-step filled"><div class="step-label">4. Model score</div><div class="step-value">${proba}</div></div>
      <div class="pipeline-step ${d.outcome ? "filled" : ""}"><div class="step-label">5. Outcome</div><div class="step-value">${d.outcome || "n/a"}</div></div>
    </div>
  `;
}

function traceHTML(txnId, txn, d) {
  const [bg, fg] = ACTION_COLOR[d.action] || ["#222", "#eee"];
  const gates = (d.gates_triggered || []).map((g) => `<span class="gate-tag">${g}</span>`).join("");
  const proba = d.predicted_success_proba != null ? `${(d.predicted_success_proba * 100).toFixed(1)}%` : "n/a (gated before model ran)";
  const outcome = d.outcome ? `<div class="trace-meta">Razorpay mock outcome: <strong>${d.outcome}</strong> (${d.razorpay_payment_id || "—"})</div>` : "";
  const cooldown = d.cooldown_minutes ? `<div class="trace-meta">Retry again in ~${d.cooldown_minutes} min</div>` : "";
  const replayed = d.replayed ? `<div class="trace-meta">⚠ idempotent replay — same client key seen before, no new action taken</div>` : "";

  return `
    ${pipelineHTML(txn, d)}
    <div class="trace-action" style="background:${bg};color:${fg}">${d.action}</div>
    ${gates}
    <p class="trace-explanation">${d.explanation}</p>
    <div class="trace-meta">Diagnosis: <strong>${d.diagnosis}</strong></div>
    <div class="trace-meta">Model predicted success probability: <strong>${proba}</strong></div>
    ${cooldown}
    ${outcome}
    ${replayed}
  `;
}

function renderTrace(txnId, txn, d, prefix = "") {
  const hintEl = $(`traceHint${prefix}`);
  if (hintEl) hintEl.textContent = txnId;
  $(`traceBody${prefix}`).innerHTML = traceHTML(txnId, txn, d);
}

async function openWorkflow(txn, opts = {}) {
  const prefix = opts.prefix ?? "";
  const rich = opts.rich !== false;
  const hintEl = $(`traceHint${prefix}`);
  if (hintEl) hintEl.textContent = txn.txn_id;
  $(`traceBody${prefix}`).innerHTML = `<p class="empty">Loading history for ${txn.txn_id}…</p>`;
  try {
    const retries = await getJSON(`/transactions/${txn.txn_id}/retries`);
    if (!retries.length) {
      $(`traceBody${prefix}`).innerHTML = `<p class="empty">No decisions run yet for this transaction.</p><button class="primary" onclick="runDecision('${txn.txn_id}', {openModal:false, prefix:'${prefix}'})">Run recovery agent</button>`;
    } else {
      const last = retries[retries.length - 1];
      renderTrace(
        txn.txn_id,
        txn,
        {
          action: last.action,
          reason_code: last.reason_code,
          diagnosis: last.diagnosis || "unclassified",
          explanation: `Last decision: ${last.reason_code}`,
          predicted_success_proba: last.predicted_success_proba,
          gates_triggered: [],
          outcome: last.outcome,
          razorpay_payment_id: last.razorpay_payment_id,
        },
        prefix
      );
    }
  } catch (e) {
    $(`traceBody${prefix}`).innerHTML = `<p class="empty">Failed to load: ${e}</p>`;
    return;
  }

  if (rich) await loadTransactionDetail(txn.txn_id);
}

// ---------------------------------------------------------------------
// per-transaction detail: Failure DNA, WhatsApp preview, Agent Replay
// (only shown in the full Transactions view)
// ---------------------------------------------------------------------

async function loadTransactionDetail(txnId) {
  await Promise.all([loadFailureDNA(txnId), loadWhatsappPreview(txnId), loadAgentReplay(txnId)]);
}

async function loadFailureDNA(txnId) {
  try {
    const dna = await getJSON(`/insights/transactions/${txnId}/failure-dna`);
    $("dnaSubpanel").style.display = "block";
    $("dnaGrid").innerHTML = `
      <div class="dna-cell"><div class="dna-label">Classification</div><div class="dna-value">${dna.classification}</div></div>
      <div class="dna-cell"><div class="dna-label">Diagnosis</div><div class="dna-value">${dna.diagnosis}</div></div>
      <div class="dna-cell"><div class="dna-label">Recovery likelihood</div><div class="dna-value">${dna.recovery_likelihood != null ? pct(dna.recovery_likelihood) : "—"}</div></div>
      <div class="dna-cell"><div class="dna-label">Bank</div><div class="dna-value">${dna.bank}</div></div>
      <div class="dna-cell"><div class="dna-label">Method</div><div class="dna-value">${dna.payment_method}</div></div>
      <div class="dna-cell"><div class="dna-label">Attempt</div><div class="dna-value">#${dna.attempt_no}</div></div>
    `;
  } catch (e) {
    $("dnaSubpanel").style.display = "none";
  }
}

async function loadWhatsappPreview(txnId) {
  try {
    const res = await getJSON(`/insights/transactions/${txnId}/whatsapp-preview`);
    if (!res.available) {
      $("whatsappSubpanel").style.display = "none";
      return;
    }
    $("whatsappSubpanel").style.display = "block";
    $("whatsappBody").innerHTML = `
      <div class="whatsapp-label">Hinglish · customer-facing</div>
      <div class="whatsapp-bubble">${res.preview.message}</div>
    `;
  } catch (e) {
    $("whatsappSubpanel").style.display = "none";
  }
}

async function loadAgentReplay(txnId) {
  try {
    const res = await getJSON(`/insights/transactions/${txnId}/replay`);
    $("replaySubpanel").style.display = "block";
    $("replayTimeline").innerHTML = res.timeline.length
      ? res.timeline
          .map(
            (step) => `
        <div class="replay-step">
          <span class="replay-step-stage">${step.stage.replace(/_/g, " ")}</span>
          <span class="replay-step-detail">${step.detail}</span>
        </div>`
          )
          .join("")
      : `<p class="empty">No timeline yet.</p>`;
  } catch (e) {
    $("replaySubpanel").style.display = "none";
  }
}

// ---------------------------------------------------------------------
// modal (kept from original, unchanged behavior)
// ---------------------------------------------------------------------

function openModalFor(txnId, txn, d) {
  $("modalTitle").textContent = txnId;
  $("modalBody").innerHTML = traceHTML(txnId, txn, d);
  $("modalBackdrop").classList.remove("hidden");
}

$("modalClose").addEventListener("click", () => $("modalBackdrop").classList.add("hidden"));
$("modalBackdrop").addEventListener("click", (e) => {
  if (e.target === $("modalBackdrop")) $("modalBackdrop").classList.add("hidden");
});

// ---------------------------------------------------------------------
// audit trail
// ---------------------------------------------------------------------

async function loadAudit() {
  try {
    const entries = await getJSON("/audit-log?limit=30");
    const body = $("auditBody");
    if (!entries.length) {
      body.innerHTML = `<p class="empty">No audit entries yet — run the agent on a transaction above.</p>`;
      return;
    }
    body.innerHTML = entries
      .map(
        (e) => `
        <div class="audit-entry">
          <span class="ts">${new Date(e.created_at).toLocaleString()}</span>
          · txn #${e.transaction_id} · <span class="event-tag">${e.event}</span>
          <div class="audit-explain">${e.explanation}</div>
        </div>`
      )
      .join("");
  } catch (e) {
    console.error("failed to load audit log", e);
  }
}

// ---------------------------------------------------------------------
// systemic incidents (Command Center summary + full Incidents view)
// ---------------------------------------------------------------------

function incidentCardHTML(inc) {
  const confClass = inc.confidence >= 0.75 ? "" : "confidence-med";
  const badgeClass = inc.confidence >= 0.75 ? "" : "med";
  const planHTML = inc.action_plan
    .map((s) => `<div class="plan-step"><span class="plan-action">${s.action}</span><span>${s.detail}</span></div>`)
    .join("");
  return `
    <div class="incident-card ${confClass}" data-incident-id="${inc.id}">
      <div class="incident-head">
        <div>
          <div class="incident-title">${inc.dimension === "bank" ? "Bank" : "Method"}: ${inc.affected} — ${inc.failure_reason.replace(/_/g, " ")}</div>
          <div class="incident-sub">${inc.diagnosis}</div>
        </div>
        <span class="incident-badge ${badgeClass}">${(inc.confidence * 100).toFixed(0)}% confidence</span>
      </div>
      <div class="incident-stats">
        <div class="incident-stat"><div class="is-label">Affected</div><div class="is-value">${inc.affected_count} txns</div></div>
        <div class="incident-stat"><div class="is-label">Share of systemic</div><div class="is-value">${inc.affected_share_pct}%</div></div>
        <div class="incident-stat"><div class="is-label">Spike vs baseline</div><div class="is-value">${inc.spike_vs_baseline}</div></div>
        <div class="incident-stat"><div class="is-label">₹ at risk</div><div class="is-value">${inr(inc.amount_at_risk)}</div></div>
      </div>
      <div class="incident-action">${inc.recommended_action}</div>
      <div class="incident-plan">${planHTML}</div>
      <div class="incident-footer">
        <span class="incident-sub">${inc.affected_txn_ids.length} txn_ids: ${inc.affected_txn_ids.slice(0, 3).join(", ")}${inc.affected_txn_ids.length > 3 ? "…" : ""}</span>
        <button class="secondary exec-btn" data-incident-exec="${inc.id}">Execute plan</button>
      </div>
    </div>
  `;
}

function bindExecButtons(container) {
  container.querySelectorAll("[data-incident-exec]").forEach((btn) => {
    btn.addEventListener("click", () => {
      btn.classList.add("executed");
      btn.textContent = "✓ Plan executed (simulated)";
      btn.disabled = true;
    });
  });
}

async function loadIncidents() {
  try {
    const res = await getJSON("/insights/incidents");
    const incidents = res.incidents;

    $("kpiIncidentCount").textContent = incidents.length;
    $("kpiIncidentSub").textContent = incidents.length ? "detected right now" : "no systemic patterns detected";

    const cmdBody = $("commandIncidentsBody");
    if (!incidents.length) {
      cmdBody.innerHTML = `<p class="empty">No systemic clusters detected — failures currently look independent, not correlated.</p>`;
    } else {
      cmdBody.innerHTML = incidents.slice(0, 2).map(incidentCardHTML).join("");
      bindExecButtons(cmdBody);
    }

    const fullBody = $("incidentsBody");
    if (!incidents.length) {
      fullBody.innerHTML = `<p class="empty">No systemic clusters detected right now. Try the chaos simulator below to see detection in action.</p>`;
    } else {
      fullBody.innerHTML = incidents.map(incidentCardHTML).join("");
      bindExecButtons(fullBody);
    }
  } catch (e) {
    console.error("failed to load incidents", e);
  }
}

// ---------------------------------------------------------------------
// revenue leak radar
// ---------------------------------------------------------------------

$("leakTabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".leak-tab");
  if (!btn) return;
  document.querySelectorAll(".leak-tab").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  _leakDim = btn.dataset.dim;
  renderLeakBars();
});

let _leakDataCache = null;

async function loadRevenueLeak() {
  try {
    _leakDataCache = await getJSON("/insights/revenue-leak");
    renderLeakBars();
  } catch (e) {
    console.error("failed to load revenue leak radar", e);
  }
}

function renderLeakBars() {
  if (!_leakDataCache) return;
  const entries = _leakDataCache[_leakDim] || [];
  const max = Math.max(1, ...entries.map((r) => r.amount));
  const isCategory = _leakDim === "by_category";
  $("leakBars").innerHTML = entries.length
    ? entries
        .map(
          (r) => `
      <div class="bar-row">
        <span class="bar-name">${r.key}</span>
        <div class="bar-track"><div class="bar-fill ${isCategory ? r.key : ""}" style="width:${(r.amount / max) * 100}%"></div></div>
        <span class="bar-count">${inr(r.amount)}</span>
      </div>`
        )
        .join("")
    : `<p class="empty">No revenue at risk right now.</p>`;
}

// ---------------------------------------------------------------------
// reflow vs blind retry
// ---------------------------------------------------------------------

async function loadReflowVsBlind() {
  try {
    const res = await getJSON("/insights/reflow-vs-blind");
    $("vsReflowRecovered").textContent = inr(res.reflow.revenue_recovered);
    $("vsReflowSub").textContent = `${pct(res.reflow.recovery_rate)} recovery rate`;
    $("vsBlindRecovered").textContent = inr(res.blind_retry.revenue_recovered);
    $("vsBlindSub").textContent = `${pct(res.blind_retry.recovery_rate)} · ${inr(res.blind_retry.penalty_cost_rupees)} penalty cost · ${res.blind_retry.unnecessary_retries} unnecessary retries`;
    const delta = res.delta.penalty_avoided_rupees;
    $("vsDelta").textContent = `Reflow avoids ${inr(delta)} in penalty/friction cost vs. blindly retrying every failure.`;
  } catch (e) {
    console.error("failed to load reflow-vs-blind", e);
  }
}

// ---------------------------------------------------------------------
// opportunity score
// ---------------------------------------------------------------------

async function loadOpportunityScore() {
  try {
    const res = await getJSON("/insights/opportunity-score");
    const rows = res.transactions.slice(0, 15);
    $("opportunityBody").innerHTML = rows.length
      ? rows
          .map(
            (r) => `
      <div class="opp-row">
        <span class="opp-txn">${r.txn_id}</span>
        <span class="opp-why">${r.why}</span>
        <span class="opp-value">${inr(r.expected_value)}</span>
        <span class="opp-value">${pct(r.predicted_success_proba)}</span>
        <span class="priority-badge priority-${r.priority}">${r.priority}</span>
      </div>`
          )
          .join("")
      : `<p class="empty" style="padding:16px 18px">No open transactions to score.</p>`;
  } catch (e) {
    console.error("failed to load opportunity score", e);
  }
}

// ---------------------------------------------------------------------
// calibration
// ---------------------------------------------------------------------

async function loadCalibration() {
  try {
    const res = await getJSON("/insights/calibration");
    $("calibBody").innerHTML = res.buckets
      .map((b) => {
        const predW = b.avg_predicted != null ? b.avg_predicted * 100 : 0;
        const actW = b.actual_success_rate != null ? b.actual_success_rate * 100 : 0;
        return `
        <div class="calib-row">
          <span class="calib-bucket">${b.bucket}</span>
          <div class="calib-bar-track"><div class="calib-bar-predicted" style="width:${predW}%"></div></div>
          <div class="calib-bar-track"><div class="calib-bar-actual" style="width:${actW}%"></div></div>
          <span class="calib-n">n=${b.n}</span>
        </div>`;
      })
      .join("");
  } catch (e) {
    console.error("failed to load calibration", e);
  }
}

// ---------------------------------------------------------------------
// chaos / incident simulator
// ---------------------------------------------------------------------

async function initChaosScenarios() {
  try {
    const res = await getJSON("/insights/chaos-scenarios");
    $("chaosSelect").innerHTML = res.scenarios.map((s) => `<option value="${s.key}">${s.label}</option>`).join("");
  } catch (e) {
    console.error("failed to load chaos scenarios", e);
  }
}

$("chaosRunBtn").addEventListener("click", async () => {
  const btn = $("chaosRunBtn");
  const key = $("chaosSelect").value;
  btn.disabled = true;
  btn.textContent = "Simulating…";
  try {
    const res = await getJSON(`/insights/chaos-simulate?scenario_key=${encodeURIComponent(key)}`, { method: "POST" });
    const triggered = res.triggered_incidents;
    $("chaosResult").innerHTML = `
      <p class="empty" style="padding-top:12px">Injected ${res.injected_transactions.length} synthetic transactions for "${res.scenario.label}".</p>
      ${
        triggered.length
          ? `<div class="incident-list">${triggered.map(incidentCardHTML).join("")}</div>`
          : `<p class="empty">No incident threshold crossed by this scenario.</p>`
      }
      <p class="chaos-note">${res.note}</p>
    `;
    bindExecButtons($("chaosResult"));
  } catch (e) {
    $("chaosResult").innerHTML = `<p class="empty">Simulation failed: ${e}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Inject scenario";
  }
});

// ---------------------------------------------------------------------
// policy lab
// ---------------------------------------------------------------------

function renderPolicyBaseline(m) {
  if (!m) {
    $("policyLiveBody").innerHTML = `<p class="empty">Loading…</p>`;
    return;
  }
  $("policyLiveBody").innerHTML = `
    <div class="compare-row"><span>Revenue recovered</span><span>${inr(m.revenue_recovered)}</span></div>
    <div class="compare-row"><span>Revenue at risk</span><span>${inr(m.revenue_at_risk)}</span></div>
    <div class="compare-row"><span>Recovery rate</span><span>${pct(m.recovery_rate)}</span></div>
    <div class="compare-row"><span>Unnecessary retries avoided</span><span>${m.avoided_unnecessary_retries}</span></div>
    <div class="compare-row"><span>Escalations</span><span>${m.escalations} (${pct(m.escalation_rate)})</span></div>
  `;
}

function renderPolicySim(sim) {
  $("policySimBody").innerHTML = `
    <div class="compare-row"><span>Revenue recovered</span><span>${inr(sim.revenue_recovered)}</span></div>
    <div class="compare-row"><span>Revenue at risk</span><span>${inr(sim.revenue_at_risk)}</span></div>
    <div class="compare-row"><span>Recovery rate</span><span>${pct(sim.recovery_rate)}</span></div>
    <div class="compare-row"><span>Unnecessary retries</span><span>${sim.unnecessary_retries}</span></div>
    <div class="compare-row"><span>Escalations</span><span>${sim.n_escalated} (${pct(sim.escalation_rate)})</span></div>
    <div class="compare-row"><span>Penalty cost</span><span>${inr(sim.penalty_cost_rupees)}</span></div>
  `;
}

let _policyDebounce = null;
async function runPolicySimulation() {
  const threshold = Number($("thresholdSlider").value) / 100;
  const cooldownMult = Number($("cooldownSlider").value) / 100;
  $("thresholdValue").textContent = `${$("thresholdSlider").value}%`;
  $("cooldownValue").textContent = `${cooldownMult.toFixed(1)}x`;

  clearTimeout(_policyDebounce);
  _policyDebounce = setTimeout(async () => {
    try {
      const sim = await getJSON(
        `/insights/policy-lab?confidence_threshold=${threshold}&cooldown_multiplier=${cooldownMult}`,
        { method: "POST" }
      );
      renderPolicySim(sim);
    } catch (e) {
      console.error("policy simulation failed", e);
    }
  }, 200);
}

$("thresholdSlider").addEventListener("input", runPolicySimulation);
$("cooldownSlider").addEventListener("input", runPolicySimulation);

// ---------------------------------------------------------------------
// run-all-pending / refresh
// ---------------------------------------------------------------------

async function runAllPending() {
  const btn = $("runAllBtn");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const pending = await getJSON("/transactions?status=failed&limit=200");
    for (const t of pending) {
      await fetch(`${apiBase()}/transactions/${t.txn_id}/decide`, { method: "POST" });
    }
    await loadAll();
  } catch (e) {
    console.error("run-all failed", e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run agent on all pending";
  }
}

$("refreshBtn").addEventListener("click", loadAll);
$("runAllBtn").addEventListener("click", runAllPending);

initChaosScenarios();
loadAll();
runPolicySimulation();
setInterval(loadMetrics, 15000);
