from datetime import datetime

from fastapi import Depends, Header, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_access_token, generate_user_tokens
from app.crud import refresh_token as crud_refresh_token
from app.deps.db import get_db
from app.models.user import User

gpu_api_key_header = APIKeyHeader(name="X-GPU-API-Key", auto_error=False)


def verify_gpu_api_key(api_key: str | None = Security(gpu_api_key_header)) -> None:
    if not api_key or api_key != settings.GPU_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing GPU API key"
        )


bearer_scheme = HTTPBearer(auto_error=False)


# old method, currently unused
def get_current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        db: Session = Depends(get_db)
) -> User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    user_id, _ = decode_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User inactive or not found"
        )

    return user


def get_current_user_id(
        response: Response,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        x_refresh_token: str | None = Header(default=None),
        db: Session = Depends(get_db),
) -> str:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    user_id, is_expired = decode_access_token(credentials.credentials)

    if user_id and not is_expired:
        return user_id

    if is_expired and x_refresh_token:
        record = crud_refresh_token.get_refresh_token(db, x_refresh_token)
        expires = record.expires_at.replace(tzinfo=None) if record and record.expires_at.tzinfo else (record.expires_at if record else None)
        if record and not record.revoked and expires > datetime.utcnow():
            crud_refresh_token.revoke_refresh_token(db, x_refresh_token)
            new_tokens = generate_user_tokens(db, record.user_id)
            db.commit()
            response.headers["X-New-Access-Token"] = new_tokens.access_token
            response.headers["X-New-Refresh-Token"] = new_tokens.refresh_token
            return str(record.user_id)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token"
    )
