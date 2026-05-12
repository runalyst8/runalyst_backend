import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from typing import List

from app.crud import analysis_result as crud_analysis
from app.crud import run as crud_run
from app.schemas.analysis import AnalysisCreateIn, AnalysisOut

# Setup logger for the analysis logic
logger = logging.getLogger(__name__)


def create_analysis_result(db: Session, *, payload: AnalysisCreateIn) -> AnalysisOut:
    run_id = payload.run_id
    logger.info(f"Attempting to persist analysis results for run_id: {run_id}")

    run_obj = crud_run.get_run(db, run_id=run_id)
    if not run_obj:
        logger.warning(f"Analysis creation failed: Run {run_id} does not exist in DB")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated run not found"
        )

    try:
        # model_dump() is the Pydantic v2 way to get the dict
        analysis_data = payload.model_dump()

        logger.debug(f"Checking for existing analysis for run_id: {run_id}")
        existing_result = crud_analysis.get_by_run_id(db, run_id=run_id)

        if existing_result:
            logger.info(f"Overwriting existing analysis result for run_id: {run_id}")
            new_result = crud_analysis.update(db, db_obj=existing_result, obj_in=analysis_data)
        else:
            logger.info(f"Creating new analysis result for run_id: {run_id}")
            new_result = crud_analysis.create(db, obj_in=analysis_data)

        # Crucial state transition: update the Run status to completed
        logger.debug(f"Updating run {run_id} status to 'completed'")
        crud_run.update_run_status(db, db_obj=run_obj, status="completed")

        db.commit()
        db.refresh(new_result)
        logger.info(f"Analysis cycle complete for run_id: {run_id}. Database committed.")
        return new_result

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during analysis persistence for run {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database failure while saving analysis"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error saving analysis for run {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist analysis results"
        )


def get_run_analysis(db: Session, *, run_id: int, user_id: int) -> AnalysisOut:
    logger.debug(f"Fetching analysis result for run_id: {run_id}, user_id: {user_id}")
    result = crud_analysis.get_by_run_id(db, run_id=run_id)

    # Note: result.owner refers to the Run object because of your relationship setup
    if not result:
        logger.warning(f"No analysis result found for run_id: {run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result not found"
        )

    #logger.info(f"result.owner.id: {result.owner.id}")
    #logger.info(f"result.owner.user_id: {result.owner.user_id}")
    #logger.info(f"DEBUG TYPE - result.owner.user_id: {type(result.owner.user_id)}")
    #logger.info(f"DEBUG TYPE - user_id: {type(user_id)}")
    if int(result.owner.user_id) != int(user_id):
        logger.error(f"Unauthorized access: User {user_id} tried to access analysis for run {run_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this analysis"
        )

    return result


def get_user_history(db: Session, *, user_id: int) -> List[AnalysisOut]:
    logger.info(f"Service: Retrieving analysis history for user_id: {user_id}")
    try:
        results = crud_analysis.get_multi_by_user(db, user_id=user_id)
        logger.debug(f"Found {len(results)} historical results for user {user_id}")
        return results
    except Exception as e:
        logger.error(f"Error fetching history for user {user_id}: {str(e)}")
        raise