"""
Security module: JWT authentication, API key validation, rate limiting.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from loguru import logger

from backend.config import get_settings

settings = get_settings()

# --- Password hashing ---
# CryptContext is defined here but NOT called at module load time.
# Calling pwd_context.hash() at import time causes a bcrypt compatibility
# crash on Windows with bcrypt >= 4.1. Hashes below are pre-computed.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# --- Models ---
class TokenData(BaseModel):
    username: Optional[str] = None
    exp: Optional[datetime] = None


class UserInDB(BaseModel):
    username: str
    hashed_password: str
    disabled: bool = False


# --- Simple in-memory user store (replace with DB in production) ---
# Passwords are PRE-HASHED — never hash at module load time to avoid
# bcrypt/passlib version conflicts on startup.
#   admin  → admin123
#   demo   → demo123
FAKE_USERS_DB: dict[str, UserInDB] = {
    "admin": UserInDB(
        username="admin",
        hashed_password="$2b$12$/bn65NnDKEn/niEdPVhCl.fz8885e5KwoVd4B5zdXoLMTfgoqPJ.O",
        disabled=False,
    ),
    "demo": UserInDB(
        username="demo",
        hashed_password="$2b$12$DB/l4IdpqER0wHsIfjmyCO1K9uUH2KQQEgJTNIzT.N0oOQwR9Gu0y",
        disabled=False,
    ),
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = FAKE_USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return api_key


def validate_file_extension(filename: str) -> bool:
    """Validate that the file has an allowed extension."""
    allowed = {".pdf", ".docx", ".txt"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in allowed


def validate_file_size(size_bytes: int) -> bool:
    """Validate that the file is within allowed size."""
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    return size_bytes <= max_bytes
