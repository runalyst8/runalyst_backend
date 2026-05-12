from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.crud import user as crud_user
from app.schemas.user import  UserUpdateIn
from app.core.security import hash_password
from app.models.user import User

def get_user(db: Session, *, user_id: int) -> User:
    user = crud_user.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account is disabled"
        )
    return user

def get_user_with_profile(db: Session, *, user_id: int) -> User:
    user = crud_user.get_user_with_profile(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

def remove_account(db: Session, *, user_id: int) -> None:
    user = crud_user.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    try:
        crud_user.delete_user(db, db_obj=user)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete account"
        )

def update_user_account(db: Session, *, user_id: int, payload: UserUpdateIn) -> User:
    db_obj = crud_user.get_user_by_id(db, user_id=user_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    update_data = payload.model_dump(exclude_unset=True)

    if "password" in update_data:
        if len(update_data["password"].encode('utf-8')) > 72:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password cannot exceed 72 bytes"
            )

        update_data["hashed_password"] = hash_password(update_data["password"])
        del update_data["password"]

    if "email" in update_data and update_data["email"] != db_obj.email:
        if crud_user.get_user_by_email(db, email=update_data["email"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )

    try:
        updated_user = crud_user.update_user(db, db_obj=db_obj, obj_in=update_data)
        db.commit()
        db.refresh(updated_user)
        return updated_user
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user account"
        )