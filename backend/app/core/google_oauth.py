import os
from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi import HTTPException, status

_http = requests.Request()

def _get_client_ids() -> list[str]:
    ids = [
        os.getenv("GOOGLE_ANDROID_CLIENT_ID"),
        os.getenv("GOOGLE_IOS_CLIENT_ID"),
        os.getenv("GOOGLE_WEB_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_ID"),
    ]
    return [cid for cid in ids if cid]


def verify_google_id_token(token: str) -> dict:
    client_ids = _get_client_ids()
    if not client_ids:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured"
        )

    last_error = None
    for cid in client_ids:
        try:
            idinfo = id_token.verify_oauth2_token(token, _http, cid)
            if idinfo.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
                raise ValueError("Wrong issuer.")
            return idinfo
        except Exception as e:
            last_error = e

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Invalid Google token: {last_error}"
    )