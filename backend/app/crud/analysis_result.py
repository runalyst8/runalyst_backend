from typing import Any, List, Optional

from sqlalchemy.orm import Session

from app.models.analysis_result import AnalysisResult
from app.models.run import Run


def create(db: Session, *, obj_in: dict) -> AnalysisResult:
    db_obj = AnalysisResult(**obj_in)
    db.add(db_obj)
    return db_obj


def get_by_run_id(db: Session, *, run_id: int) -> Optional[AnalysisResult]:
    return db.query(AnalysisResult).filter(AnalysisResult.run_id == run_id).first()


def get_multi_by_user(
        db: Session, *, user_id: int, skip: int = 0, limit: int = 100
) -> List[AnalysisResult]:
    results = (
        db.query(AnalysisResult)
        .join(Run)
        .filter(Run.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    if not results:
        return []

    return results


def update(db: Session, *, db_obj: AnalysisResult, obj_in: dict) -> AnalysisResult:
    for field in obj_in:
        if hasattr(db_obj, field):
            setattr(db_obj, field, obj_in[field])

    db.add(db_obj)
    return db_obj


def save_recommendations(
    db: Session, *, db_obj: AnalysisResult, recommendations: dict[str, Any]
) -> AnalysisResult:
    db_obj.recommendations = recommendations
    db.add(db_obj)
    return db_obj


def remove(db: Session, *, analysis_id: int) -> Optional[AnalysisResult]:
    obj = db.query(AnalysisResult).get(analysis_id)
    if obj:
        db.delete(obj)
    return obj