from sqlalchemy.orm import Session
from app.models.profile_info import ProfileInfo


def get_profile_by_user_id(db: Session, user_id: int) -> ProfileInfo | None:
    return db.query(ProfileInfo).filter(ProfileInfo.user_id == user_id).first()


def create_profile_info(db: Session, *, user_id: int, profile_data: dict) -> ProfileInfo:
    db_obj = ProfileInfo(
        **profile_data,
        user_id=user_id
    )
    db.add(db_obj)
    return db_obj


def update_profile_info(db: Session, *, db_obj: ProfileInfo, obj_in: dict) -> ProfileInfo:
    for field in obj_in:
        if hasattr(db_obj, field):
            setattr(db_obj, field, obj_in[field])

    db.add(db_obj)
    return db_obj