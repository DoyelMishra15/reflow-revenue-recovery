from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import TransactionIn, TransactionOut, RetryAttemptOut

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("", response_model=TransactionOut)
def ingest_transaction(payload: TransactionIn, db: Session = Depends(get_db)):
    existing = db.query(Transaction).filter(Transaction.txn_id == payload.txn_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"txn_id '{payload.txn_id}' already exists")

    txn = Transaction(**payload.model_dump())
    # anchor last_failure_at so it lines up with the caller-supplied
    # minutes_since_last_failure at the moment of ingestion - see the field
    # comment on Transaction for why this exists
    txn.last_failure_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=payload.minutes_since_last_failure
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return txn


@router.get("", response_model=list[TransactionOut])
def list_transactions(status: str | None = None, limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(Transaction)
    if status:
        q = q.filter(Transaction.status == status)
    return q.order_by(Transaction.created_at.desc()).limit(limit).all()


@router.get("/{txn_id}", response_model=TransactionOut)
def get_transaction(txn_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    return txn


@router.get("/{txn_id}/retries", response_model=list[RetryAttemptOut])
def get_retries(txn_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.txn_id == txn_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    return txn.retries
