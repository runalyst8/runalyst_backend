from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.deps.db import get_db
from app.deps.auth import get_current_user_id
from app.services.user import service as user_service
from app.schemas.user import  UserUpdateIn, UserOut

router = APIRouter()

@router.get("/me", response_model=UserOut, status_code=status.HTTP_200_OK)
def get_me(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    return user_service.get_user(db, user_id=user_id)

@router.delete("/me", status_code=status.HTTP_200_OK)
def delete_my_account(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    user_service.remove_account(db, user_id=user_id)
    return None

@router.patch("/me", response_model=UserOut, status_code=status.HTTP_200_OK)
def update_my_account(
    payload: UserUpdateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    return user_service.update_user_account(
        db,
        user_id=user_id,
        payload=payload
    )