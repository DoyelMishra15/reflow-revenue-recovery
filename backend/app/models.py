from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    txn_id = Column(String, unique=True, index=True)
    amount = Column(Float)
    payment_method = Column(String)
    bank = Column(String)
    failure_reason = Column(String)
    attempt_no = Column(Integer, default=1)
    hour_of_day = Column(Integer)
    is_weekend = Column(Integer)
    customer_past_success_rate = Column(Float)
    minutes_since_last_failure = Column(Integer)
    is_flagged_risk = Column(Integer, default=0)
    status = Column(String, default="failed")  # failed | recovered | given_up | escalated
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # anchor timestamp for "minutes since last failure" - set at ingestion
    # (offset by the caller-supplied minutes_since_last_failure) and reset
    # every time a new failure actually happens (a retry attempt that
    # doesn't clear). This is what lets a delay_retry decision genuinely
    # resolve later: /decide recomputes minutes_since_last_failure from this
    # anchor each time it's called, instead of trusting a number that would
    # otherwise never change no matter how much real time passes.
    last_failure_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    retries = relationship("RetryAttempt", back_populates="transaction", cascade="all, delete-orphan")
    audit_entries = relationship("AuditLogEntry", back_populates="transaction", cascade="all, delete-orphan")


class RetryAttempt(Base):
    __tablename__ = "retry_attempts"
    __table_args__ = (
        # same idempotency key replayed against the same txn returns the cached
        # result instead of firing the action twice - see routers/retries.py
        UniqueConstraint("transaction_id", "idempotency_key", name="uq_txn_idempotency_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    action = Column(String)
    reason_code = Column(String)
    diagnosis = Column(String, nullable=True)
    predicted_success_proba = Column(Float, nullable=True)
    outcome = Column(String, nullable=True)  # captured | failed | pending | n/a
    razorpay_payment_id = Column(String, nullable=True)
    idempotency_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    transaction = relationship("Transaction", back_populates="retries")


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    event = Column(String)
    explanation = Column(Text)
    gates_triggered = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    transaction = relationship("Transaction", back_populates="audit_entries")
