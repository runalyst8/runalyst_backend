from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session
from app.deps.db import get_db
from app.schemas.auth import SignUpIn, Token, GoogleAuthIn, AppleAuthIn
from app.schemas.user import UserOut
from app.services.auth import service as auth_service
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
    return Token(access_token=token)

@router.post("/apple", response_model=Token)
def apple_login(payload: AppleAuthIn, db: Session = Depends(get_db)):
    logger.info(f"Processing Apple OAuth login attempt for email hint: {payload.email}")
    token = auth_service.process_apple_auth(
        db,
        identity_token=payload.identity_token,
        email_hint=payload.email
    )
    logger.info("Apple OAuth login successful")
    return Token(access_token=token)