"""Tests for app/insights.py - the new intelligence layer added on top of
the existing retry engine. These test the pure functions directly against
constructed fixtures rather than the live seeded DB, so they're deterministic
regardless of what's currently in transactions_sample.csv."""
from types import SimpleNamespace

from app import insights


def _txn(**kwargs):
    defaults = dict(
        id=1,
        txn_id="t1",
        amount=1000.0,
        payment_method="card",
        bank="hdfc",
        failure_reason="issuer_declined",
        attempt_no=1,
        hour_of_day=12,
        is_weekend=0,
        customer_past_success_rate=0.5,
        minutes_since_last_failure=10,
        is_flagged_risk=0,
        status="failed",
        created_at=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_classify_customer_vs_systemic_vs_fraud():
    assert insights._classify("insufficient_funds") == "customer"
    assert insights._classify("card_expired") == "customer"
    assert insights._classify("issuer_declined") == "systemic"
    assert insights._classify("bank_server_down") == "systemic"
    assert insights._classify("risk_block") == "fraud"


def test_detect_incidents_flags_a_real_cluster():
    txns = [
        _txn(id=i, txn_id=f"t{i}", bank="hdfc", failure_reason="issuer_declined", amount=1000.0)
        for i in range(5)
    ] + [
        _txn(id=100, txn_id="lonely", bank="sbi", failure_reason="card_expired", amount=200.0),
    ]
    incidents = insights.detect_incidents(txns)
    assert len(incidents) >= 1
    hdfc_incident = next(i for i in incidents if i["affected"] == "hdfc")
    assert hdfc_incident["affected_count"] == 5
    assert hdfc_incident["amount_at_risk"] == 5000.0
    assert "affected_txn_ids" in hdfc_incident
    assert len(hdfc_incident["affected_txn_ids"]) == 5


def test_detect_incidents_ignores_small_or_diffuse_groups():
    txns = [_txn(id=i, txn_id=f"t{i}", bank=f"bank{i}", failure_reason="issuer_declined") for i in range(5)]
    incidents = insights.detect_incidents(txns)
    # every bank is unique here - no single bank cluster should qualify
    assert not any(i["dimension"] == "bank" for i in incidents)


def test_detect_incidents_never_clusters_customer_side_reasons():
    txns = [
        _txn(id=i, txn_id=f"t{i}", bank="hdfc", failure_reason="insufficient_funds") for i in range(5)
    ]
    incidents = insights.detect_incidents(txns)
    assert incidents == []


def test_build_action_plan_has_at_least_one_step():
    incident = {"failure_reason": "bank_server_down", "affected_count": 3}
    plan = insights.build_action_plan(incident)
    assert len(plan) >= 1
    assert all("action" in step and "detail" in step for step in plan)


def test_blind_retry_outcome_is_deterministic_per_txn_id():
    d = {
        "txn_id": "stable_id",
        "amount": 500.0,
        "payment_method": "upi",
        "failure_reason": "network_timeout",
        "hour_of_day": 10,
        "attempt_no": 1,
        "customer_past_success_rate": 0.6,
        "minutes_since_last_failure": 20,
    }
    r1 = insights.blind_retry_outcome(d)
    r2 = insights.blind_retry_outcome(d)
    assert r1 == r2


def test_reflow_vs_blind_produces_matching_row_count():
    txns = [_txn(id=i, txn_id=f"t{i}") for i in range(5)]
    result = insights.reflow_vs_blind(txns, retry_attempts_by_txn={})
    assert result["n_transactions"] == 5
    assert len(result["rows"]) == 5
    assert "revenue_recovered" in result["reflow"]
    assert "revenue_recovered" in result["blind_retry"]


def test_whatsapp_preview_only_for_customer_side_reasons():
    customer_txn = {"txn_id": "t1", "amount": 500.0, "failure_reason": "insufficient_funds"}
    systemic_txn = {"txn_id": "t2", "amount": 500.0, "failure_reason": "bank_server_down"}
    assert insights.whatsapp_preview(customer_txn) is not None
    assert insights.whatsapp_preview(systemic_txn) is None


def test_whatsapp_preview_is_hinglish_and_includes_amount():
    d = {"txn_id": "t1", "amount": 1234.0, "failure_reason": "card_expired"}
    preview = insights.whatsapp_preview(d)
    assert preview["language"] == "hinglish"
    assert "1234" in preview["message"]


def test_simulate_policy_higher_threshold_retries_less_or_equal():
    txns = [
        _txn(id=i, txn_id=f"t{i}", failure_reason="gateway_error", attempt_no=1, minutes_since_last_failure=60)
        for i in range(10)
    ]
    loose = insights.simulate_policy(txns, confidence_threshold=0.05, cooldown_multiplier=1.0)
    strict = insights.simulate_policy(txns, confidence_threshold=0.95, cooldown_multiplier=1.0)
    assert strict["n_retried"] <= loose["n_retried"]


def test_simulate_policy_never_mutates_input_transactions():
    txns = [_txn(id=1, txn_id="t1", status="failed")]
    insights.simulate_policy(txns, confidence_threshold=0.5, cooldown_multiplier=1.0)
    assert txns[0].status == "failed"  # untouched


def test_calibration_curve_buckets_have_expected_shape():
    attempts = [
        SimpleNamespace(predicted_success_proba=0.1, outcome="failed"),
        SimpleNamespace(predicted_success_proba=0.85, outcome="captured"),
        SimpleNamespace(predicted_success_proba=None, outcome=None),  # should be ignored
    ]
    buckets = insights.calibration_curve(attempts)
    assert len(buckets) == 5
    assert buckets[0]["n"] == 1
    assert buckets[-1]["n"] == 1


def test_revenue_leak_radar_only_counts_open_and_written_off():
    txns = [
        _txn(id=1, txn_id="t1", status="failed", amount=100.0, failure_reason="issuer_declined"),
        _txn(id=2, txn_id="t2", status="recovered", amount=999.0, failure_reason="issuer_declined"),
        _txn(id=3, txn_id="t3", status="given_up", amount=50.0, failure_reason="card_expired"),
    ]
    radar = insights.revenue_leak_radar(txns)
    assert radar["total_at_risk"] == 150.0  # recovered txn excluded


def test_opportunity_score_only_scores_open_transactions_and_sorts_desc():
    txns = [
        _txn(id=1, txn_id="a", status="failed", amount=100.0),
        _txn(id=2, txn_id="b", status="recovered", amount=99999.0),
        _txn(id=3, txn_id="c", status="failed", amount=5000.0),
    ]
    scored = insights.opportunity_score(txns)
    assert {s["txn_id"] for s in scored} == {"a", "c"}
    assert scored[0]["expected_value"] >= scored[-1]["expected_value"]


def test_failure_dna_classification_matches_reason():
    txn = _txn(txn_id="t1", failure_reason="card_expired")
    dna = insights.failure_dna(txn)
    assert dna["classification"] == "customer"
    assert dna["txn_id"] == "t1"


def test_inject_chaos_scenario_returns_synthetic_never_touches_db():
    scenario, injected = insights.inject_chaos_scenario("issuer_degradation")
    assert scenario is not None
    assert len(injected) == scenario.count
    assert all(t.txn_id.startswith("chaos_") for t in injected)
    assert all(t.bank == "hdfc" for t in injected)


def test_inject_chaos_scenario_unknown_key_returns_none():
    scenario, injected = insights.inject_chaos_scenario("not_a_real_scenario")
    assert scenario is None
    assert injected == []


def test_chaos_scenario_transactions_trigger_incident_detection():
    _, injected = insights.inject_chaos_scenario("issuer_degradation")
    incidents = insights.detect_incidents(injected)
    assert len(incidents) >= 1
    assert any(i["affected"] == "hdfc" for i in incidents)
