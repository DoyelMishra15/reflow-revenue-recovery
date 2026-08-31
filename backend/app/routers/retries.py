from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction, RetryAttempt
from app.schemas import DecisionOut
from app.retry_engine import decide, Action
from app.razorpay_mock import RazorpayClient, RazorpayMockError
from app import audit

router = APIRouter(prefix="/transactions", tags=["retries"])
_rzp = RazorpayClient()


def _minutes_since_last_failure(txn: Transaction) -> int:
    """
    Recomputes "how long ago did this transaction last fail" from the
    stored anchor timestamp instead of trusting a static number - so a
    delay_retry decision made 20 minutes ago genuinely looks different the
    next time /decide runs, rather than being stuck re-evaluating the same
    stale snapshot forever.
    """
    if txn.last_failure_at is None:
        return txn.minutes_since_last_failure
    elapsed = datetime.now(timezone.utc).replace(tzinfo=None) - txn.last_failure_at
    return max(0, int(elapsed.total_seconds() // 60))


def _attempt_to_decision_out(attempt: RetryAttempt) -> DecisionOut:
    """Rebuilds a DecisionOut from a stored attempt, for idempotent replays."""
    return DecisionOut(
        action=attempt.action,
        reason_code=attempt.reason_code,
        diagnosis=attempt.diagnosis or "unclassified",
        explanation=f"[replayed] original decision for this idempotency key: {attempt.reason_code}",
        predicted_success_proba=attempt.predicted_success_proba,
        cooldown_minutes=None,
        gates_triggered=[],
        outcome=attempt.outcome,
        razorpay_payment_id=attempt.razorpay_payment_id,
        replayed=True,
    )


@router.post("/{txn_id}/decide", response_model=DecisionOut)
def decide_and_act(
    txn_id: str,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")

    # replaying the same client-supplied key against the same transaction
    # returns the original result instead of re-running the action - this is
    # what actually protects against double-clicks / retried requests / a
    # crashed caller resubmitting, as opposed to just checking terminal status
    if idempotency_key:
        existing = (
            db.query(RetryAttempt)
            .filter(RetryAttempt.transaction_id == txn.id, RetryAttempt.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            return _attempt_to_decision_out(existing)

    if txn.status in ("recovered", "given_up"):
        raise HTTPException(status_code=400, detail=f"transaction already resolved (status={txn.status})")

    # observe: how much time has actually passed since this transaction (or
    # its most recent retry) last failed, recomputed fresh on every call so
    # a delay_retry from an earlier decide can genuinely clear its cooldown
    live_minutes_since_failure = _minutes_since_last_failure(txn)

    txn_dict = {
        "amount": txn.amount,
        "payment_method": txn.payment_method,
        "bank": txn.bank,
        "failure_reason": txn.failure_reason,
        "attempt_no": txn.attempt_no,
        "hour_of_day": txn.hour_of_day,
        "is_weekend": txn.is_weekend,
        "customer_past_success_rate": txn.customer_past_success_rate,
        "minutes_since_last_failure": live_minutes_since_failure,
        "is_flagged_risk": txn.is_flagged_risk,
    }
    txn.minutes_since_last_failure = live_minutes_since_failure

    decision = decide(txn_dict)

    outcome = None
    payment_id = None

    if decision.action == Action.RETRY_NOW:
        try:
            result = _rzp.retry_payment(
                order_id=txn.txn_id,
                amount=txn.amount,
                method=txn.payment_method,
                context=txn_dict,
            )
            outcome = result["status"]
            payment_id = result["id"]
            if outcome == "captured":
                txn.status = "recovered"
            else:
                # recover/escalate: this retry didn't clear, so it's a new
                # failure - reset the cooldown clock and bump the attempt
                # count so the gates (max-retries, large-txn-repeat) see an
                # accurate picture next time this transaction is decided
                txn.status = "failed"
                txn.attempt_no += 1
                txn.last_failure_at = datetime.now(timezone.utc).replace(tzinfo=None)
        except RazorpayMockError as e:
            outcome = "gateway_error"
            txn.attempt_no += 1
            txn.last_failure_at = datetime.now(timezone.utc).replace(tzinfo=None)
            audit.log(db, txn.id, "gateway_error", str(e))
    elif decision.action == Action.STOP_NO_RETRY:
        txn.status = "given_up"
    elif decision.action == Action.ESCALATE_HUMAN:
        txn.status = "escalated"
    elif decision.action == Action.DELAY_RETRY:
        txn.status = "delayed"
    elif decision.action == Action.SUGGEST_ALT_METHOD:
        txn.status = "alt_method_suggested"

    attempt = RetryAttempt(
        transaction_id=txn.id,
        action=decision.action.value,
        reason_code=decision.reason_code,
        diagnosis=decision.diagnosis,
        predicted_success_proba=decision.predicted_success_proba,
        outcome=outcome,
        razorpay_payment_id=payment_id,
        idempotency_key=idempotency_key,
    )
    db.add(attempt)
    try:
        db.commit()
    except IntegrityError:
        # lost a race with another request carrying the same idempotency key -
        # the action already ran under that request, so return its result
        # instead of double-executing (and undo the status/attempt_no changes
        # we made above, since this request's write didn't actually land)
        db.rollback()
        existing = (
            db.query(RetryAttempt)
            .filter(RetryAttempt.transaction_id == txn.id, RetryAttempt.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            return _attempt_to_decision_out(existing)
        raise

    audit.log(
        db,
        txn.id,
        event=decision.action.value,
        explanation=decision.explanation,
        gates_triggered=decision.gates_triggered,
    )

    db.commit()
    db.refresh(txn)

    return DecisionOut(
        action=decision.action.value,
        reason_code=decision.reason_code,
        diagnosis=decision.diagnosis,
        explanation=decision.explanation,
        predicted_success_proba=decision.predicted_success_proba,
        cooldown_minutes=decision.cooldown_minutes,
        gates_triggered=decision.gates_triggered,
        outcome=outcome,
        razorpay_payment_id=payment_id,
    )
