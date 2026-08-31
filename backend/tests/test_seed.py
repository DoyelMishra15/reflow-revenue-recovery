import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")  # only matters if app.db isn't loaded yet

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import Base  # noqa: E402
from app.models import Transaction  # noqa: E402
from app import seed as seed_module  # noqa: E402
from app.seed import GUARANTEED_DEMO_TXNS  # noqa: E402
from app.retry_engine import decide, Action  # noqa: E402
from app.ml.features import to_feature_dict  # noqa: E402

# app.db.engine/SessionLocal are module-level singletons bound at import
# time, and other test files in this same pytest session may already have
# imported app.db against their own temp database. Rather than depend on
# collection order, give this file its own isolated engine and point the
# seed module at it directly.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_engine = create_engine(f"sqlite:///{_tmp_db.name}", connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
Base.metadata.create_all(bind=_engine)
seed_module.SessionLocal = _TestSession


def test_seed_populates_empty_db():
    inserted = seed_module.seed_if_empty(sample_size=10)
    assert inserted > 0

    db = _TestSession()
    try:
        total = db.query(Transaction).count()
        assert total == inserted
    finally:
        db.close()


def test_seed_is_a_noop_when_data_already_exists():
    seed_module.seed_if_empty(sample_size=10)
    db = _TestSession()
    try:
        before = db.query(Transaction).count()
    finally:
        db.close()

    inserted_again = seed_module.seed_if_empty(sample_size=10)
    assert inserted_again == 0

    db = _TestSession()
    try:
        after = db.query(Transaction).count()
    finally:
        db.close()
    assert after == before


def test_guaranteed_demo_txns_actually_hit_every_action():
    """
    The whole point of the hand-picked fixtures is that a judge can
    reliably see all four decision paths. Verify each one actually
    produces the action it's there to demonstrate, by calling the real
    engine on it - not asserting against a hardcoded expectation that
    could drift from retry_engine.py silently.
    """
    by_txn_id = {t["txn_id"]: t for t in GUARANTEED_DEMO_TXNS}

    expected = {
        "demo_retry_now": Action.RETRY_NOW,
        "demo_stop_no_retry": Action.STOP_NO_RETRY,
        "demo_escalate_human": Action.ESCALATE_HUMAN,
        "demo_suggest_alt_method": Action.SUGGEST_ALT_METHOD,
    }

    for txn_id, expected_action in expected.items():
        txn = dict(by_txn_id[txn_id])
        to_feature_dict(txn)  # sanity: all feature cols present
        decision = decide(txn)
        assert decision.action == expected_action, (
            f"{txn_id} was expected to demonstrate {expected_action}, "
            f"got {decision.action} instead"
        )
