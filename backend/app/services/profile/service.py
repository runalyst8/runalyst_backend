from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.crud import profile_info as crud_profile
from app.schemas.profile import ProfileUpdateIn
from app.models.profile_info import ProfileInfo


def get_user_profile(db: Session, *, user_id: int) -> ProfileInfo:
    profile = crud_profile.get_profile_by_user_id(db, user_id=user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Please complete onboarding."
        )
    return profile


def upsert_profile(db: Session, *, user_id: int, payload: ProfileUpdateIn) -> ProfileInfo:
    existing_profile = crud_profile.get_profile_by_user_id(db, user_id=user_id)
    update_data = payload.model_dump(exclude_unset=True)

    if not existing_profile and not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create a profile with no data."
        )

    try:
        if existing_profile:
            updated_profile = crud_profile.update_profile_info(
                db,
                db_obj=existing_profile,
                obj_in=update_data
            )
        else:
            updated_profile = crud_profile.create_profile_info(
                db,
                user_id=user_id,
                profile_data=update_data
            )

        db.commit()
        db.refresh(updated_profile)
        return updated_profile

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save profile information"
        )