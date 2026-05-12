import boto3
import os

# These run once when the app (or worker) starts
AWS_REGION = os.environ.get("AWS_REGION")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")

if not SQS_QUEUE_URL:
    raise ValueError("SQS_QUEUE_URL must be set in environment variables.")

# Create the client ONCE at the top level
sqs_client = boto3.client("sqs", region_name=AWS_REGION)