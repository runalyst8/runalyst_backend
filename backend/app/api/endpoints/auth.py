from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.deps.db import get_db
from app.schemas.auth import SignUpIn, Token, GoogleAuthIn, AppleAuthIn, TokenRefreshRequest
from app.schemas.user import UserOut
from app.services.auth import service as auth_service
from app.services import otp_store
from app.services.email import send_verification_email
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: SignUpIn, db: Session = Depends(get_db)):
    logger.info(f"New signup attempt for email: {payload.email}")
    user = auth_service.register_user(db, payload=payload)
    logger.info(f"User successfully registered: {user.id}")
    return user

@router.post("/login", response_model=Token)
def login(payload: SignUpIn, db: Session = Depends(get_db)):
    logger.info(f"Login attempt for email: {payload.email}")
    token = auth_service.authenticate_user(db, payload=payload)
    logger.info(f"Successful login for email: {payload.email}")
    return token

@router.post("/google", response_model=Token)
def google_login(payload: GoogleAuthIn, db: Session = Depends(get_db)):
    logger.info("Processing Google OAuth login attempt")
    token = auth_service.process_google_auth(db, token=payload.token)
    logger.info("Google OAuth login successful")
    return token

@router.post("/apple", response_model=Token)
def apple_login(payload: AppleAuthIn, db: Session = Depends(get_db)):
    logger.info(f"Processing Apple OAuth login attempt for email hint: {payload.email}")
    token = auth_service.process_apple_auth(
        db,
        identity_token=payload.identity_token,
        email_hint=payload.email
    )
    logger.info("Apple OAuth login successful")
    return token


class EmailIn(BaseModel):
    email: EmailStr

class VerifyEmailIn(BaseModel):
    email: EmailStr
    code: str

@router.post("/send-verification-email", status_code=status.HTTP_200_OK)
def send_verification(payload: EmailIn):
    code = otp_store.save_otp(payload.email)
    try:
        send_verification_email(payload.email, code)
    except Exception as e:
        logger.error(f"Failed to send verification email to {payload.email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send verification email")
    return {"detail": "Verification email sent"}

@router.post("/verify-email", status_code=status.HTTP_200_OK)
def verify_email(payload: VerifyEmailIn):
    if not otp_store.verify_otp(payload.email, payload.code):
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")
    return {"detail": "Email verified successfully"}

@router.post(
    "/refresh",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Submit a long-lived refresh token to obtain a brand new access and refresh token pair."
)
def refresh_token(
    payload: TokenRefreshRequest,
    db: Session = Depends(get_db)
):
    new_tokens = auth_service.refresh_access_token(db, refresh_token=payload.refresh_token)
    return new_tokens