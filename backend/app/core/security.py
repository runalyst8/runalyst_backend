from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
import bcrypt
from app.core.config import settings

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


def decode_access_token(token: str) -> str | None:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string to decode
        
    Returns:
        Subject (user ID) if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALG]
        )
        sub = payload.get("sub")
        
        # Ensure sub exists and is a string
        if sub is None or not isinstance(sub, str):
            return None
            
        return sub
    except JWTError:
        return None
    except Exception:
        # Catch any other unexpected errors
        return None

import secrets

def create_refresh_token() -> str:
    return secrets.token_urlsafe(64)

def refresh_token_expiry() -> datetime:
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

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