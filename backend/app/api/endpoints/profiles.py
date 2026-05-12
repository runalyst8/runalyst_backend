from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.deps.db import get_db
from app.deps.auth import get_current_user_id
from app.services.profile import service as profile_service
from app.schemas.profile import ProfileUpdateIn, ProfileOut

router = APIRouter()

@router.get("/me", response_model=ProfileOut)
def get_my_profile(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    return profile_service.get_user_profile(db, user_id=user_id)

@router.patch("/me", response_model=ProfileOut)
def update_my_profile(
    payload: ProfileUpdateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    return profile_service.upsert_profile(
        db,
        user_id=user_id,
        payload=payload
    )