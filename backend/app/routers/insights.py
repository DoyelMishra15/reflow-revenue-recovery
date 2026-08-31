from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction, RetryAttempt, AuditLogEntry
from app import insights

router = APIRouter(prefix="/insights", tags=["insights"])


def _retry_attempts_by_txn(db: Session) -> dict:
    grouped = defaultdict(list)
    for a in db.query(RetryAttempt).order_by(RetryAttempt.created_at.asc()).all():
        grouped[a.transaction_id].append(a)
    return grouped


@router.get("/incidents")
def incidents(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    incident_list = insights.detect_incidents(txns)
    for inc in incident_list:
        inc["action_plan"] = insights.build_action_plan(inc)
    return {"incidents": incident_list, "count": len(incident_list)}


@router.get("/reflow-vs-blind")
def reflow_vs_blind(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    attempts = _retry_attempts_by_txn(db)
    return insights.reflow_vs_blind(txns, attempts)


@router.get("/calibration")
def calibration(db: Session = Depends(get_db)):
    attempts = db.query(RetryAttempt).all()
    return {"buckets": insights.calibration_curve(attempts)}


@router.get("/revenue-leak")
def revenue_leak(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    return insights.revenue_leak_radar(txns)


@router.get("/opportunity-score")
def opportunity_score(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    return {"transactions": insights.opportunity_score(txns)}


@router.post("/policy-lab")
def policy_lab(
    confidence_threshold: float = Query(0.20, ge=0.0, le=1.0),
    cooldown_multiplier: float = Query(1.0, ge=0.1, le=5.0),
    db: Session = Depends(get_db),
):
    txns = db.query(Transaction).all()
    return insights.simulate_policy(txns, confidence_threshold, cooldown_multiplier)


@router.get("/chaos-scenarios")
def chaos_scenarios():
    return {
        "scenarios": [
            {"key": s.key, "label": s.label} for s in insights.CHAOS_SCENARIOS.values()
        ]
    }


@router.post("/chaos-simulate")
def chaos_simulate(scenario_key: str, db: Session = Depends(get_db)):
    scenario, injected = insights.inject_chaos_scenario(scenario_key)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario '{scenario_key}'")

    real_txns = db.query(Transaction).all()
    combined = list(real_txns) + injected
    incident_list = insights.detect_incidents(combined)
    for inc in incident_list:
        inc["action_plan"] = insights.build_action_plan(inc)

    # only surface incidents that involve at least one injected transaction,
    # so the response clearly demonstrates "this scenario triggered detection"
    injected_ids = {t.txn_id for t in injected}
    triggered = [i for i in incident_list if injected_ids & set(i["affected_txn_ids"])]

    injected_preview = [
        {
            "txn_id": t.txn_id,
            "amount": t.amount,
            "bank": t.bank,
            "payment_method": t.payment_method,
            "failure_reason": t.failure_reason,
        }
        for t in injected
    ]

    return {
        "scenario": {"key": scenario.key, "label": scenario.label},
        "injected_transactions": injected_preview,
        "triggered_incidents": triggered,
        "note": "Simulation only - nothing here was written to the database.",
    }


@router.get("/transactions/{txn_id}/whatsapp-preview")
def whatsapp_preview(txn_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    d = insights._txn_to_dict(txn)
    preview = insights.whatsapp_preview(d)
    return {"txn_id": txn_id, "available": preview is not None, "preview": preview}


@router.get("/transactions/{txn_id}/failure-dna")
def failure_dna(txn_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    latest = (
        db.query(RetryAttempt)
        .filter(RetryAttempt.transaction_id == txn.id)
        .order_by(RetryAttempt.created_at.desc())
        .first()
    )
    return insights.failure_dna(txn, latest)


@router.get("/transactions/{txn_id}/replay")
def agent_replay(txn_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")

    attempts = (
        db.query(RetryAttempt)
        .filter(RetryAttempt.transaction_id == txn.id)
        .order_by(RetryAttempt.created_at.asc())
        .all()
    )
    audit_entries = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.transaction_id == txn.id)
        .order_by(AuditLogEntry.created_at.asc())
        .all()
    )

    timeline = [
        {
            "stage": "payment_failure",
            "at": txn.created_at.isoformat(),
            "detail": f"₹{txn.amount:.0f} failed via {txn.payment_method} ({txn.failure_reason}).",
        },
        {
            "stage": "feature_extraction",
            "at": txn.created_at.isoformat(),
            "detail": (
                f"bank={txn.bank}, method={txn.payment_method}, attempt_no={txn.attempt_no}, "
                f"hour_of_day={txn.hour_of_day}, customer_past_success_rate={txn.customer_past_success_rate}"
            ),
        },
    ]

    audit_by_time = {a.created_at: a for a in audit_entries}
    for attempt in attempts:
        matching_audit = next(
            (a for a in audit_entries if abs((a.created_at - attempt.created_at).total_seconds()) < 2),
            None,
        )
        timeline.append(
            {
                "stage": "diagnosis",
                "at": attempt.created_at.isoformat(),
                "detail": f"Diagnosis: {attempt.diagnosis or 'unclassified'}",
            }
        )
        timeline.append(
            {
                "stage": "policy_gate",
                "at": attempt.created_at.isoformat(),
                "detail": matching_audit.explanation if matching_audit else f"reason_code={attempt.reason_code}",
                "gates_triggered": matching_audit.gates_triggered if matching_audit else None,
            }
        )
        timeline.append(
            {
                "stage": "decision",
                "at": attempt.created_at.isoformat(),
                "detail": f"action={attempt.action}, predicted_success_proba={attempt.predicted_success_proba}",
            }
        )
        if attempt.outcome:
            timeline.append(
                {
                    "stage": "outcome",
                    "at": attempt.created_at.isoformat(),
                    "detail": f"outcome={attempt.outcome}" + (f", payment_id={attempt.razorpay_payment_id}" if attempt.razorpay_payment_id else ""),
                }
            )

    return {"txn_id": txn_id, "current_status": txn.status, "timeline": timeline}
