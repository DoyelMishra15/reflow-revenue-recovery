# Reflow — AI Revenue Recovery for Failed Payments

Not "retry everything and hope" — a gated ML decision engine that tells recoverable failures from lost causes, in real time, with every decision auditable.

Built for the **Razorpay AI Buildathon — Track 03: AI Revenue Recovery**.

## Live Demo

🔗 **[https://reflow-revenue-recovery.onrender.com/](https://reflow-revenue-recovery.onrender.com/)** — single-service deploy, API + dashboard on one origin.

> Hosted on Render's free tier — the service spins down after ~15 minutes of inactivity, so the first request may take 30–50s to cold-start.

## The Problem

A large share of "failed" payments aren't actually lost — a card blip, a 90-second bank outage, a timed-out OTP. Retrying those recovers real revenue. Retrying everything else (expired cards, fraud declines) just burns issuer trust and gets a merchant's retry privileges throttled. The hard part is telling the two apart, cheaply, in real time — without letting an automated system do anything unsafe, and while being able to prove *why* every decision was made.

## The Solution

Reflow ingests failed transactions, diagnoses the root cause, and runs each one through hard safety gates before a gradient-boosted classifier scores *should this succeed if retried*. The model never acts directly — it feeds a probability into a deterministic decision layer that picks one of five bounded actions, executes it against a mock Razorpay client, and writes the full reasoning to an audit trail.

```
detect → diagnose → gate → score (ML) → decide → act → observe → audit
```

## Why Reflow Is Different

| Most "AI recovery" demos | Reflow |
|---|---|
| LLM reads the error, LLM decides everything | Deterministic gates decide safety; a narrow classifier only scores retry probability |
| One retry policy for every failure | Diagnosis-aware routing across 5 bounded actions |
| No visibility into *why* | Every decision, gate, and model score logged to an audit trail |
| Claims a big number, no baseline | Measured against an honest gates-only baseline — see [ML Evaluation](#ml-evaluation) |
| Retries blindly until it works | Compares itself against a blind-retry counterfactual, transaction by transaction |

## Key Features

- **Systemic Failure Detection** — clusters failures by bank/method + reason to flag issuer outages, gateway degradation, or rail-wide problems before they look like thousands of unrelated customer failures.
- **Reflow vs. Blind Retry** — every transaction shows what a naive "retry immediately" policy would have done vs. what Reflow actually did, on the same hidden ground-truth outcome.
- **Failure DNA** — a per-transaction fingerprint: classification (customer / systemic / fraud), diagnosis, recovery likelihood, and current action.
- **Agent Replay** — full reconstructed timeline for any transaction: failure → feature extraction → diagnosis → gates → decision → outcome.
- **Revenue Leak Radar** — at-risk revenue broken down by failure reason, bank, payment method, and category.
- **Recovery Opportunity Score** — ranks open failures by expected recovered value (amount × model probability, attempt-friction adjusted) into high/medium/low priority.
- **Policy Lab** — drag confidence-threshold and cooldown sliders and re-simulate the whole policy in memory, no writes to the database.
- **Chaos / Incident Simulator** — inject a named failure scenario (issuer degradation, UPI outage, decline spike) and watch incident detection trigger on it live.
- **WhatsApp Recovery Preview** — templated Hinglish nudge messages for customer-side failures only (never sent for bank/issuer-side outages, since the customer can't fix those).
- **Safety gates before the model** — risk flags, non-recoverable reasons, retry caps, and large-repeat-amount transactions are routed away before the classifier ever runs.
- **Full auditability** — `/audit-log` and per-transaction replay show every gate fired, every score, every action taken.

## How the Recovery Engine Works

1. **Detect** — every ingested failed transaction is tracked as revenue at risk.
2. **Diagnose** — `failure_reason` maps to a root-cause bucket (customer-side, bank-side, issuer-side, fraud) via a deterministic lookup — not a model call.
3. **Gate** — hard rules run first: risk flags, non-recoverable reasons, retry cap, large-amount repeat attempts. The model never sees a transaction the gates have already ruled out.
4. **Score** — a gradient-boosted classifier estimates *P(retry succeeds)* from 8 tabular features.
5. **Decide** — probability + cooldown state resolve to one of five actions: `retry_now`, `delay_retry`, `suggest_alt_method`, `escalate_human`, `stop_no_retry`.
6. **Act** — the chosen action executes against a mock Razorpay client, gated by an idempotency key so a replayed request never double-acts.
7. **Observe** — the outcome is captured and written back to the transaction and retry history.
8. **Audit** — every gate, score, and action is logged with an explanation for that specific decision.

## Architecture / Tech Stack

![architecture](docs/architecture.svg)

```
frontend (vanilla JS)  →  FastAPI  →  retry_engine.decide()
                                          │
                              ┌───────────┴───────────┐
                       rule gates (no model)     GBM classifier
                       risk / expired / cap      P(retry succeeds)
                                          │
                                 razorpay_mock.RazorpayClient
                                          │
                              SQLite (transactions, retries, audit_log)
```

| Layer | Stack |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy, SQLite |
| ML | scikit-learn `GradientBoostingClassifier` on 8 tabular features |
| Frontend | Vanilla JS ops dashboard — KPIs, incidents, opportunity score, policy lab, chaos simulator, audit feed |
| Payments | Mocked Razorpay client — request/response shape matched to `razorpay-python` |
| Deployment | Single-origin Docker image (API + static frontend from one FastAPI process) |

## ML Evaluation

⚠️ **All figures below are on synthetic data** (`data/generate_synthetic_data.py`) via a hand-authored ground-truth recoverability function — there is no public dataset of real Razorpay failure/retry outcomes. The model never sees the ground-truth function directly, only sampled features and outcomes, same as it would with real production data. The mock gateway samples from the *same* function using the transaction's real features — not the model's own prediction — so the evaluation isn't the model grading its own homework.

**Held-out test set** (n=1,200, ₹14.29L revenue at risk):

| Policy | Retry rate | Success rate when retried | Unnecessary retries | False-positive cost | Revenue recovered |
|---|---|---|---|---|---|
| `retry_all` (no gates, no model) | 100% | 49.6% | 605 | ₹4,840 | ₹645,154 |
| `retry_all_gated` (honest baseline — gates only) | 86.1% | 55.5% | 460 | ₹3,680 | ₹629,206 |
| **`reflow_ml_policy` (full engine)** | 80.3% | **58.1%** | **404** | **₹3,232** | ₹623,109 |

Against the gates-only baseline, Reflow recovers **99.0%** of the same revenue while cutting unnecessary retries by **12.2%** and lifting hit-rate-when-retried from 55.5% → 58.1%. The model isn't chasing gross recovery — it's trading a small, measured amount of revenue for meaningfully fewer wasted attempts against real bank/issuer rails.

**Classifier at the production threshold (0.20, not the sklearn default of 0.5):**

| | Predicted succeeds | Predicted fails |
|---|---|---|
| **Actually succeeds** | TP 561 | FN 34 |
| **Actually fails** | FP 419 | TN 186 |

Precision 0.572 · Recall 0.943 · F1 0.712 · ROC-AUC 0.818 (full metrics in `app/ml/train_metrics.json`). The threshold is deliberately low — a missed retry (false negative) gives up real revenue outright, while a false positive only costs one cheap wasted attempt, so the model is tuned to rarely turn down a transaction that would have actually succeeded.

## Judge Demo — Recommended Walkthrough

1. **Open the dashboard** — KPI row shows revenue at risk/recovered, recovery rate, and live incident count. Fixtures auto-seed on first boot, so it's never empty.
2. **Click a transaction → Recovery workflow** — see the full detect → diagnose → gate → decide pipeline for that one payment, plus its **Failure DNA** card.
3. **Run Agent Replay** on the same transaction — the reconstructed decision timeline, gate-by-gate.
4. **Open WhatsApp Recovery Preview** on a customer-side failure (e.g. `insufficient_funds`) — templated Hinglish nudge message.
5. **Go to Revenue Leak Radar** — at-risk revenue sliced by bank, method, and reason.
6. **Check Recovery Opportunity Score** — the queue ranked by expected recovered value, not just amount.
7. **Trigger a Chaos Scenario** (e.g. "UPI rail outage") and watch **Systemic Incident Detection** flag it live with a recommended action plan.
8. **Open Policy Lab**, drag the confidence threshold — recovery rate and unnecessary retries recompute instantly, in memory.
9. **Compare Reflow vs. Blind Retry** — same transactions, naive-retry counterfactual vs. what Reflow actually did.
10. **Check the Audit Trail** — every gate, score, and action, newest first.

## Local Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # model.pkl is checked in — retraining is optional
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser (or `python -m http.server 8080` from `frontend/`) — the dashboard auto-seeds demo data and points at `localhost:8000` automatically.
 
**Or with Docker:**

```bash
docker compose up --build
```

Backend on `:8000`, frontend on `:8080`, dashboard pre-populated on first boot.

## Limitations

- **All data is synthetic**, generated from a hand-authored ground-truth function — the evaluation numbers demonstrate the approach, not a production recovery rate.
- **The payment gateway is mocked** (`razorpay_mock.py`) — no live Razorpay integration, no real card/UPI/netbanking rail, no money moves.
- **Demo environment, not production infrastructure** — no auth, SQLite, no rate limiting.
- The GBM is trained on 8 tabular features from 6,000 synthetic rows — deliberately small, not a ceiling on what a larger model/real dataset could do.
