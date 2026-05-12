import uuid
import logging
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.supabase_client import supabase_client
from app.schemas.run import RunCreateIn, RunOut, RunAllOut
from app.crud import run as crud_run
from app.services.queue.service import send_message_to_queue

# Get logger for this specific module
logger = logging.getLogger(__name__)


def get_upload_url(*, user_id: int):
    bucket_name = "user_videos_test"
    unique_filename = f"{user_id}/{uuid.uuid4()}.mp4"

    try:
        logger.debug(f"Generating signed upload URL for user {user_id} at path {unique_filename}")
        response = supabase_client.storage.from_(bucket_name).create_signed_upload_url(
            path=unique_filename
        )

        return {
            "upload_url": response['signed_url'],
            "path": response['path']
        }

    except Exception as e:
        logger.error(f"Supabase storage error for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to interface with storage provider"
        )


def create_run_record(db: Session, *, user_id: int, payload: RunCreateIn):
    try:
        logger.info(f"Initiating run record creation for user {user_id}")
        new_run = crud_run.create_run(
            db,
            user_id=user_id,
            video_path=payload.video_path,
            status="queued",
            title=payload.title
        )

        db.commit()
        db.refresh(new_run)
        logger.debug(f"DB record created for run {new_run.id}")

        message_to_send = {
            "run_id": new_run.id,
            "video_path": new_run.video_path
        }

        logger.info(f"Sending run {new_run.id} to SQS queue")
        send_message_to_queue(message_body=message_to_send)

        return new_run

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create run/queue for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the run record to the database."
        )


def get_run_details(db: Session, *, run_id: int, user_id: int):
    logger.debug(f"Fetching run {run_id} for user {user_id}")
    run = crud_run.get_run(db, run_id=run_id)

    if not run:
        logger.warning(f"Run {run_id} not found in database")
        raise HTTPException(status_code=404, detail="Run not found")

    if run.user_id != user_id:
        logger.warning(f"Unauthorized access attempt: User {user_id} requested run {run_id} owned by {run.user_id}")
        raise HTTPException(status_code=403, detail="Access denied")

    return run


def get_all_runs_mapped(db: Session, *, user_id: int):
    try:
        logger.debug(f"Fetching and mapping all runs for user {user_id}")
        runs = crud_run.get_multi_by_owner(db, user_id=user_id)

        run_map = {
            run.id: RunOut.model_validate(run) for run in runs
        }

        return RunAllOut(runs=run_map)

    except SQLAlchemyError as e:
        logger.error(f"Database query error in get_all_runs_mapped for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Database query failed")
    except Exception as e:
        logger.error(f"Mapping error in get_all_runs_mapped for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Data processing error")


def update_run_status(db: Session, *, run_id: int, new_status: str):
    logger.info(f"Service: Attempting status update for run {run_id} to '{new_status}'")
    run = crud_run.get_run(db, run_id=run_id)

    if not run:
        logger.error(f"Update failed: Run {run_id} does not exist")
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        updated_run = crud_run.update_run_status(db, db_obj=run, status=new_status)
        db.commit()
        db.refresh(updated_run)
        logger.info(f"Run {run_id} status successfully transitioned to '{new_status}'")
        return updated_run
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update status for run {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update run status"
        )