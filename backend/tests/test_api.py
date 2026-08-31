import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# point at a throwaway db before importing the app, so tests never touch reflow.db
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"
# tests want a clean slate to make their own assertions on, not demo fixtures
os.environ["DEMO_SEED_ON_STARTUP"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

SAMPLE_TXN = {
    "txn_id": "txn_test_001",
    "amount": 1500.0,
    "payment_method": "upi",
    "bank": "hdfc",
    "failure_reason": "network_timeout",
    "attempt_no": 1,
    "hour_of_day": 15,
    "is_weekend": 0,
    "customer_past_success_rate": 0.7,
    "minutes_since_last_failure": 45,
    "is_flagged_risk": 0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_and_fetch_transaction():
    r = client.post("/transactions", json=SAMPLE_TXN)
    assert r.status_code == 200
    body = r.json()
    assert body["txn_id"] == SAMPLE_TXN["txn_id"]
    assert body["status"] == "failed"

    r2 = client.get(f"/transactions/{SAMPLE_TXN['txn_id']}")
    assert r2.status_code == 200


def test_duplicate_txn_id_rejected():
    r = client.post("/transactions", json=SAMPLE_TXN)
    assert r.status_code == 409


def test_decide_endpoint_returns_valid_action():
    r = client.post(f"/transactions/{SAMPLE_TXN['txn_id']}/decide")
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {
        "retry_now",
        "delay_retry",
        "suggest_alt_method",
        "escalate_human",
        "stop_no_retry",
    }
    assert body["explanation"]
    assert body["diagnosis"]


def test_idempotency_key_replays_instead_of_double_executing():
    txn = {**SAMPLE_TXN, "txn_id": "txn_test_idem_001"}
    r1 = client.post("/transactions", json=txn)
    assert r1.status_code == 200

    headers = {"Idempotency-Key": "client-retry-abc123"}
    first = client.post(f"/transactions/{txn['txn_id']}/decide", headers=headers)
    assert first.status_code == 200

    before = client.get(f"/transactions/{txn['txn_id']}/retries").json()
    assert len(before) == 1

    # same key again - must not create a second retry attempt or mutate state again
    second = client.post(f"/transactions/{txn['txn_id']}/decide", headers=headers)
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert second.json()["action"] == first.json()["action"]

    after = client.get(f"/transactions/{txn['txn_id']}/retries").json()
    assert len(after) == 1  # still one attempt, not two


def test_different_idempotency_keys_are_independent():
    txn = {**SAMPLE_TXN, "txn_id": "txn_test_idem_002", "is_flagged_risk": 1, "failure_reason": "risk_block"}
    client.post("/transactions", json=txn)

    r1 = client.post(f"/transactions/{txn['txn_id']}/decide", headers={"Idempotency-Key": "key-a"})
    assert r1.status_code == 200
    # a resolved (escalated is not terminal) txn can still be re-decided under a fresh key
    r2 = client.post(f"/transactions/{txn['txn_id']}/decide", headers={"Idempotency-Key": "key-b"})
    assert r2.status_code == 200
    assert r2.json().get("replayed", False) is False


def test_audit_log_recorded_after_decision():
    r = client.get("/audit-log")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_decide_on_unknown_txn_404s():
    r = client.post("/transactions/nope_not_real/decide")
    assert r.status_code == 404


def test_dashboard_metrics_shape():
    r = client.get("/dashboard/metrics")
    assert r.status_code == 200
    body = r.json()
    for key in ["total_transactions", "by_status", "total_amount_recovered", "action_breakdown"]:
        assert key in body


def test_delayed_retry_resolves_once_cooldown_actually_passes():
    """
    A delay_retry decision shouldn't be a dead end - once enough real time
    has passed, re-running /decide on the same transaction should stop
    delaying and actually act, instead of repeating the identical decision
    forever because minutes_since_last_failure never moved.
    """
    from datetime import datetime, timedelta, timezone
    from app.db import SessionLocal
    from app.models import Transaction

    txn_id = "txn_test_delay_resolves_001"
    payload = {
        **SAMPLE_TXN,
        "txn_id": txn_id,
        "failure_reason": "bank_server_down",
        "payment_method": "netbanking",
        "customer_past_success_rate": 0.9,
        "minutes_since_last_failure": 1,  # fresh failure, well inside the cooldown
    }
    r = client.post("/transactions", json=payload)
    assert r.status_code == 200

    first = client.post(f"/transactions/{txn_id}/decide", headers={"Idempotency-Key": "k1"})
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["action"] == "delay_retry"
    assert first_body["cooldown_minutes"] and first_body["cooldown_minutes"] > 0

    # fast-forward: pretend the cooldown window (30min for bank_server_down)
    # has actually elapsed, the same way it would if a client just waited
    db = SessionLocal()
    try:
        txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
        txn.last_failure_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=35)
        db.commit()
    finally:
        db.close()

    second = client.post(f"/transactions/{txn_id}/decide", headers={"Idempotency-Key": "k2"})
    assert second.status_code == 200
    second_body = second.json()
    # cooldown cleared, so it should no longer be stuck delaying - it should
    # have actually acted (retried, and either recovered or failed again)
    assert second_body["action"] != "delay_retry"
    assert second_body["action"] == "retry_now"
    assert second_body["outcome"] in ("captured", "failed")


@pytest.fixture(autouse=True, scope="session")
def cleanup():
    yield
    try:
        os.unlink(_tmp_db.name)
    except OSError:
        pass
