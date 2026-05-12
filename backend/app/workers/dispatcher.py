import asyncio
import json
import logging

from app.core.aws_client import sqs_client, SQS_QUEUE_URL
from app.deps.db import SessionLocal
from app.crud import run as crud_run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_jobs_from_sqs():

    response = sqs_client.receive_message(
        QueueUrl=SQS_QUEUE_URL,
        MaxNumberOfMessages=5,
        WaitTimeSeconds=5
    )

    messages = response.get("Messages", [])
    if not messages:
        return

    # Open a single DB session for this batch
    db = SessionLocal()
    try:
        for message in messages:
            receipt_handle = message['ReceiptHandle']
            try:
                body = json.loads(message['Body'])
                run_id = body.get("run_id")
                video_path = body.get("video_path")

                logger.info(f"Worker: Processing run_id {run_id}")
                logger.info(f"Worker: Processing video_path {video_path}")

                # 1. Update DB status to 'processing'
                run_obj = crud_run.get_run(db, run_id=run_id)
                if run_obj:
                    crud_run.update_run_status(db, db_obj=run_obj, status="processing")
                    db.commit()

                # 2. DISPATCH LOGIC: Send the task to your GPU server here
                # (e.g., via a HTTP request to the GPU server's API)
                logger.info(f"--- DISPATCHING run_id {run_id} TO GPU SERVER ---")

                # 3. If dispatch successful, delete from SQS
                sqs_client.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )
                logger.info(f"Worker: Successfully dispatched and deleted message {run_id}")

            except Exception as e:
                logger.error(f"Worker: Failed to process individual message: {e}")
                db.rollback()
                # Message will reappear in SQS automatically due to visibility timeout

    finally:
        db.close()


async def run_dispatcher_periodically():
    logger.info("Runalyst Dispatcher started. Polling for jobs...")
    while True:
        try:
            # We run the synchronous SQS polling in a thread or just call it
            # if the logic is fast enough.
            process_jobs_from_sqs()
        except Exception as e:
            logger.error(f"Dispatcher loop error: {e}")

        # Poll every 10 seconds to save on AWS costs/CPU
        await asyncio.sleep(10)
