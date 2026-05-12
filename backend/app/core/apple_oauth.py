import requests
import time
from jose import jwt

APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"

_JWKS_CACHE = None
_JWKS_CACHE_TS = 0
_JWKS_TTL_SECONDS = 3600


def get_apple_jwks():
    global _JWKS_CACHE, _JWKS_CACHE_TS
    now = time.time()
    if _JWKS_CACHE and (now - _JWKS_CACHE_TS) < _JWKS_TTL_SECONDS:
        return _JWKS_CACHE
    resp = requests.get(APPLE_JWKS_URL, timeout=10)
    resp.raise_for_status()
    _JWKS_CACHE = resp.json()
    _JWKS_CACHE_TS = now
    return _JWKS_CACHE


def verify_apple_token(identity_token: str, audience: str) -> dict:
    jwks = get_apple_jwks()
    headers = jwt.get_unverified_header(identity_token)
    kid = headers.get("kid")
    alg = headers.get("alg")

    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if not key:
        raise ValueError("Apple public key not found.")

    return jwt.decode(
        identity_token,
        key,
        algorithms=[alg],
        audience=audience,
        issuer=APPLE_ISSUER,
        options={"verify_aud": True, "verify_iss": True, "verify_exp": True},
    )