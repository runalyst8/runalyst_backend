from app.services.run import service as run_service
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.deps.db import get_db
from app.deps.auth import get_current_user_id, verify_gpu_api_key
from app.schemas.run import RunCreateIn, RunOut, RunAllOut
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-url", status_code=status.HTTP_200_OK)
def generate_run_upload_url(user_id: int = Depends(get_current_user_id)):
    logger.info(f"User {user_id} requested a new video upload URL")
    response = run_service.get_upload_url(user_id=user_id)
    return response

@router.post("/create-record", response_model=RunOut, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    logger.info(f"Creating run record for user {user_id} with video: {payload.video_path}")
    run = run_service.create_run_record(db, user_id=user_id, payload=payload)
    logger.info(f"Run {run.id} successfully created and queued for user {user_id}")
    return run

@router.get("/get", response_model=RunOut)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    logger.debug(f"User {user_id} fetching details for run {run_id}")
    return run_service.get_run_details(db, run_id=run_id, user_id=user_id)

@router.get("/all", response_model=RunAllOut)
def get_all_runs(
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user_id)
):
    logger.info(f"Fetching all runs for user {user_id}")
    runs = run_service.get_all_runs_mapped(db, user_id=user_id)
    logger.info(f"Retrieved {len(runs.runs)} runs for user {user_id}")
    return runs

@router.patch("/update-status", response_model=RunOut)
def update_status(
    run_id: int,
    new_status: str,
    db: Session = Depends(get_db),
     _: None = Depends(verify_gpu_api_key)
):
    # This is an internal call from the GPU worker
    logger.info(f"GPU Worker updating status for run {run_id} to: {new_status}")
    updated_run = run_service.update_run_status(db, run_id=run_id, new_status=new_status)
    logger.info(f"Status successfully updated for run {run_id}")
    return updated_run