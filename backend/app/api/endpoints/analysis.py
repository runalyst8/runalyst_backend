import logging
from typing import Any, List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.deps.auth import get_current_user_id, verify_gpu_api_key
from app.deps.db import get_db
from app.schemas.analysis import AnalysisCreateIn, AnalysisOut, RecommendationsOut
from app.services.analysis import service as analysis_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/save-result", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
def save_result(
        payload: AnalysisCreateIn,
        db: Session = Depends(get_db),
        _: None = Depends(verify_gpu_api_key)
):
    # Log the metadata to track which run just finished processing
    logger.info(f"Received analysis results for Run ID: {payload.run_id} from GPU API")

    result = analysis_service.create_analysis_result(db, payload=payload)

    logger.info(f"Analysis result successfully saved for Run ID: {result.run_id}")
    return result


@router.get("/get", response_model=AnalysisOut)
def get_analysis(
        run_id: int,
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user_id)
):
    logger.debug(f"User {user_id} fetching analysis for run {run_id}")
    return analysis_service.get_run_analysis(db, run_id=run_id, user_id=user_id)


@router.get("/history", response_model=List[AnalysisOut])
def get_history(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user_id)
):
    logger.info(f"User {user_id} requesting full analysis history")
    history = analysis_service.get_user_history(db, user_id=user_id)
    logger.info(f"Retrieved {len(history)} historical analyses for user {user_id}")
    return history


@router.get("/recommendations")
async def get_recommendations(
        run_id: int,
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user_id)
):
    logger.info(f"User {user_id} requesting recommendations for run {run_id}")
    result = await analysis_service.get_or_generate_recommendations(db, run_id=run_id, user_id=user_id)
    logger.info(f"Returning recommendations for run {run_id} (issues: {len(result.get('issues', []))})")
    return {"run_id": run_id, "recommendations": result}