import json
import logging
from botocore.exceptions import BotoCoreError, ClientError
from app.core.aws_client import sqs_client, SQS_QUEUE_URL

# Standard logger setup
logger = logging.getLogger(__name__)


def send_message_to_queue(message_body: dict):
    """
    Sends a message to the SQS queue for the GPU worker.
    Includes logging to track message handoff and AWS errors.
    """
    try:
        # We log the run_id specifically to make grepping logs easier
        run_id = message_body.get("run_id", "unknown")
        logger.debug(f"Attempting to queue SQS message for run_id: {run_id}")

        response = sqs_client.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message_body)
        )

        # Log the MessageId returned by AWS as proof of receipt
        message_id = response.get("MessageId")
        logger.info(f"Successfully queued run {run_id}. SQS MessageId: {message_id}")

        return response

    except (BotoCoreError, ClientError) as e:
        # This catches network issues or AWS credential/permission errors
        logger.error(
            f"AWS SQS Error for run {message_body.get('run_id')}: {str(e)}",
            exc_info=True
        )
        raise  # Re-raise so the service layer knows the queue failed
    except Exception as e:
        logger.error(f"Unexpected error in send_message_to_queue: {str(e)}", exc_info=True)
        raise