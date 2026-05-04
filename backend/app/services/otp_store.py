import random
import string
from datetime import datetime, timedelta, timezone
from threading import Lock

_store: dict[str, dict] = {}
_lock = Lock()

OTP_TTL_MINUTES = 10
OTP_LENGTH = 6


def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=OTP_LENGTH))


def save_otp(email: str) -> str:
    code = generate_otp()
    with _lock:
        _store[email] = {
            "code": code,
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES),
        }
    return code


def verify_otp(email: str, code: str) -> bool:
    with _lock:
        entry = _store.get(email)
        if not entry:
            return False
        if datetime.now(timezone.utc) > entry["expires_at"]:
            del _store[email]
            return False
        if entry["code"] != code:
            return False
        del _store[email]
        return True
