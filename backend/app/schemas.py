from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    txn_id: str
    amount: float = Field(gt=0)
    payment_method: str
    bank: str
    failure_reason: str
    attempt_no: int = 1
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: int = 0
    customer_past_success_rate: float = Field(ge=0, le=1)
    minutes_since_last_failure: int = 0
    is_flagged_risk: int = 0


class TransactionOut(BaseModel):
    id: int
    txn_id: str
    amount: float
    payment_method: str
    bank: str
    failure_reason: str
    attempt_no: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RetryAttemptOut(BaseModel):
    id: int
    action: str
    reason_code: str
    diagnosis: Optional[str] = None
    predicted_success_proba: Optional[float]
    outcome: Optional[str]
    razorpay_payment_id: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DecisionOut(BaseModel):
    action: str
    reason_code: str
    diagnosis: str
    explanation: str
    predicted_success_proba: Optional[float]
    cooldown_minutes: Optional[int]
    gates_triggered: list[str]
    outcome: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    replayed: bool = False


class AuditLogOut(BaseModel):
    id: int
    transaction_id: int
    event: str
    explanation: str
    gates_triggered: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
