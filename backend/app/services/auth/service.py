import os

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.apple_oauth import verify_apple_token
from app.core.google_oauth import verify_google_id_token
from app.crud import user as crud_user
from app.schemas.auth import SignUpIn, Token
from app.core.security import hash_password, verify_password, create_access_token, generate_user_tokens
from app.crud import refresh_token as crud_refresh_token
from app.models.user import User



def register_user(db: Session, *, payload: SignUpIn) -> User:
    if crud_user.get_user_by_email(db, email=payload.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    if len(payload.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot exceed 72 bytes"
        )

    try:
        hashed_pwd = hash_password(payload.password)
        user = crud_user.create_user(
            db,
            email=payload.email,
            hashed_password=hashed_pwd
        )
        db.commit()
        db.refresh(user)
        return user

    except Exception as e:
        db.rollback()
        # Log the actual error 'e' here for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


def authenticate_user(db: Session, *, payload: SignUpIn) -> Token:
    user = crud_user.get_user_by_email(db, email=payload.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    tokens = generate_user_tokens(db, user.id)
    db.commit()
    return tokens


def process_google_auth(db: Session, token: str) -> Token:
    info = verify_google_id_token(token)

    email = info.get("email")
    google_sub = info.get("sub")

    # 2. Check if user exists
    user = crud_user.get_user_by_email(db, email=email)

    if not user:
        user = crud_user.create_social_user(
            db,
            email=email,
            sub=google_sub,
            provider="google"
        )
        db.commit()
    elif not user.google_sub:
        user.google_sub = google_sub
        db.commit()

    tokens = generate_user_tokens(db, user.id)
    db.commit()
    return tokens


def process_apple_auth(db: Session, identity_token: str, email_hint: str = None) -> Token:
    audience = os.getenv("APPLE_BUNDLE_ID")
    try:
        claims = verify_apple_token(identity_token, audience)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Apple Auth Failed: {str(e)}")

    apple_sub = claims["sub"]
    email = email_hint or claims.get("email")

    user = crud_user.get_by_apple_sub(db, apple_sub=apple_sub)

    if not user:
        if not email:
            raise HTTPException(status_code=400, detail="Email required for first-time sign-in")

        user = crud_user.get_user_by_email(db, email=email)
        if user:
            user.apple_sub = apple_sub
        else:
            user = crud_user.create_social_user(db, email=email, sub=apple_sub, provider="apple")

        db.commit()

    tokens = generate_user_tokens(db, user.id)
    db.commit()
    return tokens

def refresh_access_token(db: Session, refresh_token: str) -> Token:
    from datetime import datetime

    record = crud_refresh_token.get_refresh_token(db, refresh_token)
    expires = record.expires_at.replace(tzinfo=None) if record and record.expires_at.tzinfo else (record.expires_at if record else None)
    if not record or record.revoked or expires < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user = crud_user.get_user_by_id(db, user_id=record.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive or does not exist")

    crud_refresh_token.revoke_refresh_token(db, refresh_token)
    tokens = generate_user_tokens(db, user.id)
    db.commit()
    return tokens