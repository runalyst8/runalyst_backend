import os
import uuid

from app.services.storage import supabase_client
from fastapi import APIRouter, HTTPException, status, Request

router = APIRouter(prefix="/model-results", tags=["model-results"])

#GPU_SERVER_IP = os.getenv("GPU_SERVER_IP")

@router.get("/get_result_upload_url", status_code=status.HTTP_200_OK)
async def create_upload_url(request: Request):
    # Get the client IP address from headers
    #client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0]

    # Check if the IP matches the GPU server's IP
    """
    if client_ip != GPU_SERVER_IP:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Unauthorized IP"
        )
    """

    # Generate unique filename
    bucket_name = "analysis_results"
    unique_filename = f"{uuid.uuid4()}.mp4"

    try:
        signed_url_response = supabase_client.storage.from_(bucket_name).create_signed_upload_url(
            path=unique_filename
        )

        return {
            "upload_url": signed_url_response['signed_url'],
            "path": signed_url_response['path']
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create upload URL: {str(e)}"
        )