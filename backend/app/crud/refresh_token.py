from datetime import datetime
from sqlalchemy.orm import Session
from app.models.refresh_token import RefreshToken


def create_refresh_token(db: Session, *, user_id: int, token: str, expires_at: datetime) -> RefreshToken:
    record = RefreshToken(user_id=user_id, token=token, expires_at=expires_at, revoked=False)
    db.add(record)
    return record


def get_refresh_token(db: Session, token: str) -> RefreshToken | None:
    return db.query(RefreshToken).filter(RefreshToken.token == token).first()


def revoke_refresh_token(db: Session, token: str) -> None:
    record = db.query(RefreshToken).filter(RefreshToken.token == token).first()
    if record:
        record.revoked = True
