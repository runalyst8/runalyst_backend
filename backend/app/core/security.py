import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from jose import jwt, JWTError
import bcrypt
from starlette import status

from app.core.config import settings
from app.schemas.auth import Token


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
        
    Raises:
        ValueError: If password exceeds 72 bytes (bcrypt limit)
    """
    # bcrypt has a 72-byte limit
    if len(password.encode('utf-8')) > 72:
        raise ValueError("Password cannot exceed 72 bytes")
    
    # Hash the password and decode the bytes result to a string for storage
    return bcrypt.hashpw(
        password.encode('utf-8'), 
        bcrypt.gensalt(rounds=12)
    ).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain: Plain text password to verify
        hashed: Hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        # Check length before verification to avoid issues with bcrypt 5.0.0 ValueError
        if len(plain.encode('utf-8')) > 72:
            return False
        
        # Verify the password. bcrypt.checkpw expects bytes for both arguments.
        return bcrypt.checkpw(
            plain.encode('utf-8'), 
            hashed.encode('utf-8')
        )
    except ValueError:
        # Catch the ValueError that bcrypt 5.0.0 raises if the password is too long
        return False
    except Exception:
        # Handle any other verification errors gracefully
        return False


def create_access_token(sub: str, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        sub: Subject (typically user ID)
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        exp = datetime.now(timezone.utc) + expires_delta
    else:
        exp = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": sub,
        "exp": exp,
        "iat": datetime.now(timezone.utc)  # Issued at time
    }
    
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


def decode_access_token(token: str) -> tuple[str | None, bool]:
    """
    Decode and verify a JWT access token.

    Returns:
        (user_id, is_expired) — is_expired=True means the token had a valid
        signature but is past its expiry time and can be refreshed.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str):
            return None, False
        return sub, False
    except JWTError:
        # Could be expired or truly invalid — decode without exp check to distinguish
        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG],
                options={"verify_exp": False}
            )
            sub = payload.get("sub")
            if not sub or not isinstance(sub, str):
                return None, False
            return sub, True  # valid signature, just expired
        except JWTError:
            return None, False

import secrets

def create_refresh_token() -> str:
    return secrets.token_urlsafe(64)

def refresh_token_expiry() -> datetime:
    from datetime import timedelta
    return datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

def create_password_reset_token(email: str) -> str:
    """
    Creates a password reset token.
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": email}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALG)
    return encoded_jwt

def decode_password_reset_token(token: str) -> str | None:
    """
    Decodes the password reset token to get the user's email.
    """
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALG])
        return decoded_token.get("sub")
    except JWTError:
        return None

def generate_user_tokens(db, user_id: int) -> Token:
    from app.crud.refresh_token import create_refresh_token as crud_create_refresh_token

    access_token = create_access_token(sub=str(user_id), expires_delta=timedelta(minutes=15))
    raw_refresh_token = create_refresh_token()
    expires_at = refresh_token_expiry()
    crud_create_refresh_token(db, user_id=user_id, token=raw_refresh_token, expires_at=expires_at)
    return Token(access_token=access_token, refresh_token=raw_refresh_token)