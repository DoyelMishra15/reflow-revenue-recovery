from sqlalchemy.orm import Session
from app.models import AuditLogEntry


def log(db: Session, transaction_id: int, event: str, explanation: str, gates_triggered: list[str] | None = None):
    entry = AuditLogEntry(
        transaction_id=transaction_id,
        event=event,
        explanation=explanation,
        gates_triggered=",".join(gates_triggered) if gates_triggered else None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
