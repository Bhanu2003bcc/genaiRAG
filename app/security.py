import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.schemas import CurrentUser

logger = logging.getLogger("app.security")

ALGORITHM = "HS256"
security_scheme = HTTPBearer(auto_error=False)

# Limit concurrent CPU-intensive bcrypt hashing/verification to prevent thread pool saturation
bcrypt_semaphore = asyncio.Semaphore(settings.bcrypt_max_concurrent)


# =========================
# PASSWORD HASHING
# =========================

def hash_password(password: str) -> str:
    """Synchronous bcrypt hash."""
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Synchronous bcrypt verification."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


async def hash_password_async(password: str) -> str:
    """Asynchronous wrapper for hash_password using semaphore and thread pool."""
    async with bcrypt_semaphore:
        return await asyncio.to_thread(hash_password, password)


async def verify_password_async(plain_password: str, hashed_password: str) -> bool:
    """Asynchronous wrapper for verify_password using semaphore and thread pool."""
    async with bcrypt_semaphore:
        return await asyncio.to_thread(verify_password, plain_password, hashed_password)


# =========================
# JWT TOKEN GENERATION
# =========================

def generate_token(
    user_id: UUID,
    username: str,
    role: str
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(milliseconds=settings.jwt_expiration_ms)

    authority = f"ROLE_{role}" if not role.startswith("ROLE_") else role

    payload = {
        "sub": str(user_id),
        "username": username,
        "role": authority,
        "iat": now,
        "exp": expire,
        "type": "access"
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


# =========================
# CURRENT USER DEPENDENCY
# =========================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> CurrentUser:
    """
    Decodes the JWT token and extracts user identity info.
    No database lookup is performed to prevent connection pool starvation.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials missing"
        )

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM]
        )

        user_id_str = payload.get("sub")
        username = payload.get("username")
        role = payload.get("role")

        if not user_id_str or not username or not role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token claims"
            )

        user_id = UUID(user_id_str)
        # Replicate Spring Security role convention: strip ROLE_ prefix if present
        role_name = role[5:] if role.startswith("ROLE_") else role

        return CurrentUser(
            id=user_id,
            username=username,
            role=role_name
        )

    except (JWTError, ValueError) as e:
        logger.warning(f"JWT decode failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )
