"""Shared API-key authentication.

The key is read from the ``API_KEY`` environment variable at request time.
There is deliberately no built-in default: if ``API_KEY`` is unset every
protected endpoint returns 503, instead of silently accepting a key that is
published in the repository (v1 defect).
"""

import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(API_KEY_HEADER)) -> str:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Server API_KEY is not configured.")
    if key is None or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass X-API-Key header.")
    return key
