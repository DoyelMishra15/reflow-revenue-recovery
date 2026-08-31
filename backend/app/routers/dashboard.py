import json
import os

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction, RetryAttempt, AuditLogEntry
from app.schemas import AuditLogOut
from app.retry_engine import Action

router = APIRouter(tags=["dashboard"])

# still "open" - money not yet recovered and not yet written off
OPEN_STATUSES = ("failed", "delayed", "alt_method_suggested")
AVOIDED_RETRY_ACTIONS = (Action.STOP_NO_RETRY.value, Action.SUGGEST_ALT_METHOD.value)

_TRAIN_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "train_metrics.json"
)
_EVAL_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evaluation", "metrics_output.json"
)


@router.get("/dashboard/metrics")
def metrics(db: Session = Depends(get_db)):
    total = db.query(func.count(Transaction.id)).scalar() or 0
    by_status = dict(
        db.query(Transaction.status, func.count(Transaction.id)).group_by(Transaction.status).all()
    )

    revenue_recovered = (
        db.query(func.coalesce(func.sum(Transaction.amount), 0.0))
        .filter(Transaction.status == "recovered")
        .scalar()
        or 0.0
    )
    revenue_at_risk = (
        db.query(func.coalesce(func.sum(Transaction.amount), 0.0))
        .filter(Transaction.status.in_(OPEN_STATUSES))
        .scalar()
        or 0.0
    )
    revenue_given_up = (
        db.query(func.coalesce(func.sum(Transaction.amount), 0.0))
        .filter(Transaction.status == "given_up")
        .scalar()
        or 0.0
    )

    resolved = (by_status.get("recovered", 0)) + (by_status.get("given_up", 0))
    recovery_rate = (by_status.get("recovered", 0) / resolved) if resolved else 0.0

    total_retry_attempts = db.query(func.count(RetryAttempt.id)).scalar() or 0
    successful_retries = (
        db.query(func.count(RetryAttempt.id)).filter(RetryAttempt.outcome == "captured").scalar() or 0
    )
    action_breakdown = dict(
        db.query(RetryAttempt.action, func.count(RetryAttempt.id)).group_by(RetryAttempt.action).all()
    )
    avoided_unnecessary_retries = sum(action_breakdown.get(a, 0) for a in AVOIDED_RETRY_ACTIONS)

    escalations = by_status.get("escalated", 0)
    escalation_rate = (escalations / total) if total else 0.0

    active_workflows = sum(by_status.get(s, 0) for s in OPEN_STATUSES)

    failure_reason_breakdown = dict(
        db.query(Transaction.failure_reason, func.count(Transaction.id))
        .group_by(Transaction.failure_reason)
        .all()
    )

    # per-strategy performance: of the retry attempts that actually executed a
    # RETRY_NOW, how often did that specific action end up capturing the payment
    strategy_performance = {}
    for action_name, count in action_breakdown.items():
        captured = (
            db.query(func.count(RetryAttempt.id))
            .filter(RetryAttempt.action == action_name, RetryAttempt.outcome == "captured")
            .scalar()
            or 0
        )
        strategy_performance[action_name] = {
            "attempts": count,
            "captured": captured,
            "capture_rate": round(captured / count, 4) if count else 0.0,
        }

    model_performance = None
    if os.path.exists(_TRAIN_METRICS_PATH):
        with open(_TRAIN_METRICS_PATH) as f:
            model_performance = json.load(f)

    baseline_comparison = None
    if os.path.exists(_EVAL_METRICS_PATH):
        with open(_EVAL_METRICS_PATH) as f:
            baseline_comparison = json.load(f)

    return {
        "total_transactions": total,
        "by_status": by_status,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "revenue_recovered": round(revenue_recovered, 2),
        "revenue_written_off": round(revenue_given_up, 2),
        "recovery_rate": round(recovery_rate, 4),
        "total_retry_attempts": total_retry_attempts,
        "successful_retries": successful_retries,
        "avoided_unnecessary_retries": avoided_unnecessary_retries,
        "active_workflows": active_workflows,
        "escalations": escalations,
        "escalation_rate": round(escalation_rate, 4),
        "action_breakdown": action_breakdown,
        "failure_reason_breakdown": failure_reason_breakdown,
        "strategy_performance": strategy_performance,
        "model_performance": model_performance,
        "baseline_comparison": baseline_comparison,
        # kept for older frontend builds / anything hitting this endpoint expecting the old shape
        "total_amount_recovered": round(revenue_recovered, 2),
    }


@router.get("/audit-log", response_model=list[AuditLogOut])
def audit_log(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(AuditLogEntry).order_by(AuditLogEntry.created_at.desc()).limit(limit).all()
