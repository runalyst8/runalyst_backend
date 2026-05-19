from sqlalchemy.orm import Session
from app.models.run import Run

def get_run(db: Session, *, run_id: int) -> Run | None:
    return db.query(Run).filter(Run.id == run_id).first()

def get_multi_by_owner(db: Session, *, user_id: int, skip: int = 0, limit: int = 100):
    return (
        db.query(Run)
        .filter(Run.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

def create_run(db: Session, *, user_id: int, video_path: str, status: str, title: str | None = None, thumbnail_path: str | None = None) -> Run:
    db_obj = Run(
        user_id=user_id,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        status=status,
        title=title
    )
    db.add(db_obj)
    return db_obj

def update_run_status(db: Session, *, db_obj: Run, status: str) -> Run:
    db_obj.status = status
    db.add(db_obj)
    return db_obj

def delete_run(db: Session, *, run_id: int) -> bool:
    obj = db.query(Run).get(run_id)
    if obj:
        db.delete(obj)
        return True
    return False