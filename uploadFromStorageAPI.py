from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import tempfile
import os
from inference import run_algorithm

app = FastAPI(title="Insecure GPU Video Downloader")

# -----------------------------
# Request model
# -----------------------------
class VideoDownloadIn(BaseModel):
    video_url: str

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/download")
def download_video(payload: VideoDownloadIn):
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            response = requests.get(
                payload.video_url,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)

            saved_path = tmp.name

        run_algorithm(saved_path)

        return {
            "status": "downloaded",
            "saved_path": saved_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )

