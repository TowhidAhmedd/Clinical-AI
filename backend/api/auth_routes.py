"""
Authentication routes: login, token refresh.
"""
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from backend.security.auth import authenticate_user, create_access_token
from backend.api.schemas import LoginRequest, TokenResponse
from backend.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT access token."""
    user = authenticate_user(request.username, request.password)
    if not user:
        logger.warning(f"Failed login attempt for username: {request.username!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    logger.info(f"User logged in: {user.username}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
